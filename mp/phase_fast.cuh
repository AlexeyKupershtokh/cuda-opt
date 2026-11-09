#pragma once
#include <utility>

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

struct PhaseMoving { LBE_T* p[18]; };

__global__ void phase_stream(PhaseMoving src, PhaseMoving dst, int nx, int ny, int nz) {
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= nx || y >= ny) return;
    const int site = (z*ny+y)*nx+x;
    const int dx[] = {1,0,-1,0,0,0,1,-1,-1,1,1,0,-1,0,1,0,-1,0};
    const int dy[] = {0,1,0,-1,0,0,1,1,-1,-1,0,1,0,-1,0,1,0,-1};
    const int dz[] = {0,0,0,0,1,-1,0,0,0,0,1,1,1,1,-1,-1,-1,-1};
    #pragma unroll
    for (int q = 0; q < 18; ++q) {
        const int xx = x-dx[q], yy = y-dy[q], zz = z-dz[q];
        const bool inside = xx >= 0 && xx < nx && yy >= 0 && yy < ny && zz >= 0 && zz < nz;
        dst.p[q][site] = src.p[q][inside ? (zz*ny+yy)*nx+xx : site];
    }
}

inline bool phase_transfer(d_LB3d& lb) {
    if (MP_parallel.GPU_pool.size() != 1) return false;
    cudaSetDevice(MP_parallel.GPU_pool[0]);
    static d_Array3d<LBE_T> next[18];
    d_Array3d<LBE_T>* current[] = {&lb.f1,&lb.f2,&lb.f3,&lb.f4,&lb.f5,&lb.f6,&lb.f7,
        &lb.f8,&lb.f9,&lb.f10,&lb.f1u,&lb.f2u,&lb.f3u,&lb.f4u,&lb.f1d,&lb.f2d,&lb.f3d,&lb.f4d};
    PhaseMoving src, dst;
    for (int q = 0; q < 18; ++q) {
        if (next[q].Nx != lb.Nx || next[q].Ny != lb.Ny || next[q].Nz != lb.Nz)
            next[q].resize(lb.Nx,lb.Ny,lb.Nz);
        src.p[q] = current[q]->p[0];
        dst.p[q] = next[q].p[0];
        std::swap(current[q]->p[0],next[q].p[0]);
    }
    phase_stream<<<dim3((lb.Nx+15)/16,(lb.Ny+15)/16,lb.Nz),dim3(16,16)>>>(src,dst,lb.Nx,lb.Ny,lb.Nz);
    return true;
}
