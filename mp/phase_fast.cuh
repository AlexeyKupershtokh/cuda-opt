#pragma once

struct PhaseFaces { LBE_T* p[14]; };

__global__ void phase_copy_z_faces(PhaseFaces fields, int plane, int nz) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= plane) return;
    #pragma unroll
    for (int q = 0; q < 14; ++q) {
        LBE_T* f = fields.p[q];
        f[i] = f[(nz-2)*plane+i];
        f[(nz-1)*plane+i] = f[plane+i];
    }
}

inline bool phase_periodicz(d_LB3d& lb) {
    if (MP_parallel.GPU_pool.size() != 1) return false;
    cudaSetDevice(MP_parallel.GPU_pool[0]);
    PhaseFaces fields = {{lb.rho.p[0], lb.ux.p[0], lb.uy.p[0], lb.uz.p[0],
        lb.f5.p[0], lb.f1u.p[0], lb.f2u.p[0], lb.f3u.p[0], lb.f4u.p[0],
        lb.f6.p[0], lb.f1d.p[0], lb.f2d.p[0], lb.f3d.p[0], lb.f4d.p[0]}};
    const int plane = lb.Nx*lb.Ny;
    const int nz = lb.Nz;
    phase_copy_z_faces<<<(plane+255)/256,256>>>(fields,plane,nz);
    return true;
}
