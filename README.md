# cuda-opt

CUDA optimization experiments for the two D3Q19 phase-transition examples in CUP 0.208.

The `main` branch contains all 217 original distribution files, including the shipped build artifacts. Its tree matches the archive byte-for-byte. Upstream authors are A. L. Kupershtokh, D. A. Medvedev, D. I. Karpov and E. R. Pruuel of the Lavrentyev Institute of Hydrodynamics, SB RAS. [Project and upstream usage terms](https://ancient.hydro.nsc.ru/labsimflow/index.html)

Archive SHA256 is `d6559e95cb458a17f6b4c52d64919920f5ee61cb4732f7a47dcdbfafc3da3008`. [Original archive](https://ancient.hydro.nsc.ru/labsimflow/download/cup-0.208.zip)

`lossless` applies correctness fixes followed by FP64 optimizations. `lossy` starts from its final commit and uses FP32. Commit messages identify the affected example, the mechanism and validation.

Build from an x64 Visual Studio developer terminal on Windows, or a C++/CUDA toolchain on Linux.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build
```

The periodic example uses a 256×256×8 padded lattice, tau 0.6 and effective temperature 0.85. The wall example reads `ini` from the process working directory. Its stock padded lattice is 512×256×3, with tau 0.55 and gy 5e-6.

`init()` resets the state. `cuda_calc()` advances ten iterations and downloads `rho`, `ux`, `uy`, `uz` once per call. Its CUDA-event timer excludes this download.

Compare a candidate library with the corrected archive FP64 library using Python and NumPy.

```sh
git worktree add ../cuda-opt-reference 059e107
cmake -S ../cuda-opt-reference -B reference -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build reference
python tests/check_phase.py --case periodic --reference reference/d3q19_phase_tr/cuda_core.dll --candidate build/d3q19_phase_tr/cuda_core.dll
python tests/check_phase.py --case walls --reference reference/d3q19_phase_tr_force/cuda_core.dll --candidate build/d3q19_phase_tr_force/cuda_core.dll
```

The check covers all four macroscopic fields at 200, 1000 and 5000 iterations. A zero-error FP64 result must be bitwise equal. Device time is the median of 31 ten-step batches at iterations 1001–1310. Pass `--fp32` for a single-precision candidate and `--output result.json` to save samples and errors.

On RTX 3070 with CUDA 12.6, the final FP64 checkpoint run measured 1.047 ms/step for periodic and 0.375 ms/step for walls. The paired corrected archive reference measured 2.732 and 2.009 ms/step. All four fields matched bitwise at 200, 1000 and 5000 steps.

Additional FP64 checks matched bitwise after 100 steps on a 64×32×7 wall lattice, a 35×19×5 wall lattice, and two logical Z slabs sharing one GPU. The last check exercises decomposition and halo ordering on the available RTX 3070.

On `lossy`, `LBE_T` and the lattice constants use FP32. This also changes the public fields and scalar parameters to `float`. Rebuild callers against `cuda_core.h`. Use a separate FP64 reference build and pass `--fp32` to the checker.

The paired RTX 3070 run measured these device times and relative L2 density errors against the parent `lossless` FP64 implementation, which is bitwise equal to the corrected archive.

| Example | FP64 ms/step | FP32 ms/step | Speedup | Error at 200 steps | Error at 1000 steps | Error at 5000 steps |
|---|---:|---:|---:|---:|---:|---:|
| Periodic | 1.044 | 0.272 | 3.84× | 0.000595% | 0.0472% | 0.4746% |
| Walls | 0.377 | 0.094 | 4.02× | 0.000701% | 0.0494% | 0.2252% |
