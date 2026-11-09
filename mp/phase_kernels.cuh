#pragma once

#include <cuda_runtime.h>
#include <mp/lb3d.h>

namespace phase {

struct Grid {
    int nx, ny, nz;  // local dimensions, including two ghost planes
};

struct Populations {
    LBE_T* f[19];  // CUP direction order: f0..f10, f1u..f4u, f1d..f4d
};

struct Fields {
    LBE_T *rho, *ux, *uy, *uz, *psi;
};

struct Velocity {
    LBE_T x, y, z;
};

struct Offsets {
    int xm, xp, ym, yp, zm, zp;
};

__device__ __forceinline__ LBE_T weight(int q) {
    return q == 0 ? LBE_w0 : (q <= 6 ? LBE_w1 : LBE_w2);
}

__global__ void initialize(Grid grid, Populations state, Fields fields, int rank) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= grid.nx * grid.ny * grid.nz) return;
    // The original CUDA generator squares in the low 32 bits and reduces the unsigned product.
    unsigned seed = static_cast<unsigned>(index + rank * grid.nz);
    for (int n = 0; n < 100; ++n)
        seed = (seed * seed) % (101u * 127u);
    const LBE_T rho = 0.8 + (seed / (101.0 * 127.0) - 0.5) * 0.2;
    fields.rho[index] = rho;
    fields.ux[index] = fields.uy[index] = fields.uz[index] = 0;
    #pragma unroll
    for (int q = 0; q < 19; ++q) state.f[q][index] = weight(q) * rho;
}

// On the first pull the original initial condition includes independent ghost values.
// Later, x/y wrap to physical cells and z either wraps locally or reads a neighbour's slab.
template<bool Walls, bool Flat>
__device__ __forceinline__ Offsets offsets(Grid g, int x, int y, int z, bool first, bool splitZ) {
    const int plane = g.nx * g.ny;
    Offsets d;
    d.xm = !first && x == 1        ? g.nx - 3 : -1;
    d.xp = !first && x == g.nx - 2 ? 3 - g.nx :  1;
    d.ym = (!Walls && !first && y == 1        ? g.ny - 3 : -1) * g.nx;
    d.yp = (!Walls && !first && y == g.ny - 2 ? 3 - g.ny :  1) * g.nx;
    if (Flat) {
        d.zm = first ? -plane : 0;
        d.zp = first ?  plane : 0;
    } else {
        d.zm = (!splitZ && !first && z == 1        ? g.nz - 3 : -1) * plane;
        d.zp = (!splitZ && !first && z == g.nz - 2 ? 3 - g.nz :  1) * plane;
    }
    return d;
}

__device__ __forceinline__ void pull(Populations src, int i, Offsets d, bool lowY, bool highY,
                                     LBE_T (&f)[19]) {
    // At a wall, an incoming link from outside keeps its local population, including diagonals.
    const int from[19] = {
        0, d.xm, lowY ? 0 : d.ym, d.xp, highY ? 0 : d.yp, d.zm, d.zp,
        lowY ? 0 : d.xm+d.ym, lowY ? 0 : d.xp+d.ym, highY ? 0 : d.xp+d.yp, highY ? 0 : d.xm+d.yp,
        d.xm+d.zm, lowY ? 0 : d.ym+d.zm, d.xp+d.zm, highY ? 0 : d.yp+d.zm,
        d.xm+d.zp, lowY ? 0 : d.ym+d.zp, d.xp+d.zp, highY ? 0 : d.yp+d.zp
    };
    #pragma unroll
    for (int q = 0; q < 19; ++q) f[q] = src.f[q][i + from[q]];
}

__device__ __forceinline__ LBE_T potential(LBE_T rho, LBE_T temperature) {
    const LBE_T pressure = 8 * temperature * rho / (3 - rho) - 3 * rho * rho;
    return sqrt(fabs(rho / 3 - LBE_T(0.02) * pressure));
}

template<bool Walls, bool Flat>
__global__ void moments(Grid g, Populations src, Fields out, LBE_T temperature, bool first, bool splitZ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int firstY = blockIdx.y * blockDim.y + threadIdx.y + (Walls ? 0 : 1);
    if (x >= g.nx - 1) return;
    for (int y = firstY; y < g.ny - (Walls ? 0 : 1); y += blockDim.y * gridDim.y) {
        for (int z = blockIdx.z + 1; z < g.nz - 1; z += gridDim.z) {
            const int i = (z * g.ny + y) * g.nx + x;
            LBE_T f[19];
            pull(src, i, offsets<Walls, Flat>(g,x,y,z,first,splitZ), Walls && y == 0, Walls && y == g.ny-1, f);
            LBE_T rho = f[0];
            #pragma unroll
            for (int q = 1; q < 19; ++q) rho += f[q];
            out.rho[i] = rho;
            out.ux[i] = (f[1]-f[3]+f[7]-f[8]-f[9]+f[10]+f[11]-f[13]+f[15]-f[17]) / rho;
            out.uy[i] = (f[2]-f[4]+f[7]+f[8]-f[9]-f[10]+f[12]-f[14]+f[16]-f[18]) / rho;
            out.uz[i] = (f[5]-f[6]+f[11]+f[12]+f[13]+f[14]-f[15]-f[16]-f[17]-f[18]) / rho;
            out.psi[i] = potential(rho, temperature);
        }
    }
}

__device__ __forceinline__ Velocity gradient(const LBE_T* p, Offsets d) {
    Velocity v = {p[d.xp]-p[d.xm], p[d.yp]-p[d.ym], p[d.zp]-p[d.zm]};
    LBE_T pair = LBE_T(0.5) * (p[d.zp+d.xp]-p[d.zm+d.xm]);
    v.x += pair; v.z += pair;
    pair = LBE_T(0.5) * (p[d.zp+d.xm]-p[d.zm+d.xp]);
    v.x -= pair; v.z += pair;
    pair = LBE_T(0.5) * (p[d.yp+d.xp]-p[d.ym+d.xm]);
    v.x += pair; v.y += pair;
    pair = LBE_T(0.5) * (p[d.ym+d.xp]-p[d.yp+d.xm]);
    v.x += pair; v.y -= pair;
    pair = LBE_T(0.5) * (p[d.zp+d.yp]-p[d.zm+d.ym]);
    v.y += pair; v.z += pair;
    pair = LBE_T(0.5) * (p[d.zm+d.yp]-p[d.zp+d.ym]);
    v.y += pair; v.z -= pair;
    return v;
}

__device__ __forceinline__ void equilibrium_polynomial(Velocity v, LBE_T (&eq)[19]) {
    const LBE_T base = 1 - LBE_d * (v.x*v.x + v.y*v.y + v.z*v.z);
    const LBE_T xy = v.x+v.y, dx = v.x-v.y, zx = v.z+v.x, dzx = v.z-v.x;
    const LBE_T zy = v.z+v.y, dzy = v.z-v.y;
    const LBE_T projection[19] = {0, v.x, v.y, -v.x, -v.y, v.z, -v.z, xy, -dx,
                                   -xy, dx, zx, zy, dzx, dzy, -dzx, -dzy, -zx, -zy};
    eq[0] = base;
    #pragma unroll
    for (int q = 1; q < 19; ++q) {
        const LBE_T s = projection[q];
        eq[q] = base + LBE_a * s + LBE_b * s * s;
    }
}

__device__ __forceinline__ double rounded_product(double a, double b) { return __dmul_rn(a,b); }
__device__ __forceinline__ float rounded_product(float a, float b) { return __fmul_rn(a,b); }

template<bool Walls, bool Flat>
__global__ void collide(Grid g, Populations src, Populations dst, Fields fields, LBE_T omega,
                        LBE_T gx, LBE_T gy, bool first, bool splitZ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int firstY = blockIdx.y * blockDim.y + threadIdx.y + (Walls ? 0 : 1);
    if (x >= g.nx - 1) return;
    for (int y = firstY; y < g.ny - (Walls ? 0 : 1); y += blockDim.y * gridDim.y) {
        for (int z = blockIdx.z + 1; z < g.nz - 1; z += gridDim.z) {
            const int i = (z * g.ny + y) * g.nx + x;
            if (Walls && (y == 0 || y == g.ny - 1)) {
                const int adjacent = y == 0 ? i + g.nx : i - g.nx;
                const LBE_T wallRho = fields.rho[adjacent];
                #pragma unroll
                for (int q = 0; q < 19; ++q) dst.f[q][i] = weight(q) * wallRho;
                continue;
            }
            const Offsets neighbours = offsets<Walls, Flat>(g,x,y,z,false,splitZ);
            LBE_T f[19], eq[19], shiftedEq[19];
            pull(src, i, first ? offsets<Walls, Flat>(g,x,y,z,true,splitZ) : neighbours, false, false, f);
            const LBE_T rho = fields.rho[i], psi = fields.psi[i];
            const Velocity u = {fields.ux[i], fields.uy[i], fields.uz[i]};
            Velocity du = gradient(fields.psi+i, neighbours);
            if (Walls) {
                // The archived fy buffer rounds gy*rho before the interaction force is added.
                du.x = (gx + du.x*psi/3) / rho;
                du.y = (rounded_product(gy,rho) + du.y*psi/3) / rho;
                du.z = (du.z*psi/3) / rho;
            } else {
                du.x = du.x*psi / (rho*3);
                du.y = du.y*psi / (rho*3);
                du.z = du.z*psi / (rho*3);
            }
            const Velocity shifted = {u.x+du.x, u.y+du.y, u.z+du.z};
            equilibrium_polynomial(u, eq);
            equilibrium_polynomial(shifted, shiftedEq);
            #pragma unroll
            for (int q = 0; q < 19; ++q) {
                const LBE_T mass = weight(q)*rho;
                const LBE_T relaxed = f[q] + (mass*eq[q] - f[q])*omega;
                dst.f[q][i] = relaxed + mass*(shiftedEq[q] - eq[q]);
            }
        }
    }
}

} // namespace phase
