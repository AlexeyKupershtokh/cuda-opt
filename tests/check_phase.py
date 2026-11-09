"""Compare either phase example with a corrected archive FP64 library."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import tempfile
import numpy as np


class Library:
    def __init__(self, path, dims, single=False):
        real = ct.c_float if single else ct.c_double

        class Array(ct.Structure):
            _fields_ = [("p", ct.POINTER(real))] + [(s, ct.c_int) for s in ("nx", "ny", "nz")]

        self.dll = ct.CDLL(str(path))
        self.dll.init.restype = self.dll.cuda_calc.restype = None
        self.dll.init()
        actual = tuple(ct.c_int.in_dll(self.dll, s).value for s in ("Nx", "Ny", "Nz"))
        if actual != dims:
            raise ValueError(f"Expected {dims}, loaded {actual}")
        self.fields = [np.ctypeslib.as_array(Array.in_dll(self.dll, s).p, shape=(int(np.prod(dims)),))
                       .reshape(dims[::-1]) for s in ("rho", "ux", "uy", "uz")]
        self.ticks = ct.c_float.in_dll(self.dll, "LBE_elapsedTime")
        self.to_ms = float(np.prod(dims))/(480*1140e6*1e-3)

    def batch(self):
        self.dll.cuda_calc()
        return self.ticks.value*self.to_ms


def compare(actual, expected, region, single):
    a = np.stack([f[region] for f in actual.fields]).astype(np.float64)
    b = np.stack([f[region] for f in expected.fields])
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError("Nonfinite macroscopic field")
    delta = a-b
    equal = np.array_equal(a.view(np.uint64), b.view(np.uint64))
    if not single and not equal:
        raise AssertionError(f"FP64 differs, max absolute error {np.abs(delta).max()}")
    return {"bitwise_equal": equal, "rho_relative_l2": float(np.linalg.norm(delta[0])/np.linalg.norm(b[0])),
            "rho_max_abs": float(np.abs(delta[0]).max()),
            "velocity_max_vector_norm": float(np.sqrt((delta[1:]**2).sum(axis=0)).max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=("periodic", "walls"), required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    reference, candidate = args.reference.resolve(), args.candidate.resolve()
    if reference == candidate:
        ap.error("Choose distinct libraries")
    output = args.output.resolve() if args.output else None
    walls = args.case == "walls"
    dims = (512,256,3) if walls else (256,256,8)
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    # Archive DLL constructors allocate device memory under the Windows loader lock.
    cuda = ct.WinDLL("nvcuda") if os.name == "nt" else ct.CDLL("libcuda.so.1")
    context = ct.c_void_p()
    for fn, values in ((cuda.cuInit,(0,)), (cuda.cuDevicePrimaryCtxRetain,(ct.byref(context),0)),
                       (cuda.cuCtxSetCurrent,(context,))):
        if fn(*values):
            raise RuntimeError("CUDA initialization failed")
    region = np.s_[1:-1,:,1:-1] if walls else np.s_[1:-1,1:-1,1:-1]
    report = {"case": args.case, "steps_per_batch": 10, "errors": {}, "gpu_ms_per_step": {}}
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="cuda-opt-") as tmp:
        os.chdir(tmp)
        try:
            if walls:
                Path("ini").write_text("Nx 512\nNy 256\nNz 3\ntau 0.55\nT 0.85\ngx 0\ngy 5e-6\nmp_parallel 0\n")
            libraries = [Library(reference,dims), Library(candidate,dims,args.fp32)]
            samples = [[],[]]
            for batch in range(1,501):
                for i in ((0,1) if batch % 2 else (1,0)):
                    elapsed = libraries[i].batch()
                    if 100 < batch <= 131:
                        samples[i].append(elapsed)
                if batch in (20,100,500):
                    errors = compare(libraries[1],libraries[0],region,args.fp32)
                    report["errors"][batch*10] = errors
                    print(batch*10,errors,flush=True)
            report["gpu_ms_per_step"] = dict(zip(("reference","candidate"),map(lambda x: float(np.median(x)),samples)))
            report["samples_gpu_ms_per_step"] = samples
        finally:
            os.chdir(cwd)
    if output:
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(report["gpu_ms_per_step"],flush=True)


if __name__ == "__main__":
    main()
