#pragma once

#include <mp/mp.h>
#include <mp/h_array3d.h>
#include <mp/phase_kernels.cuh>
#include <algorithm>
#include <climits>
#include <cstring>
#include <memory>
#include <vector>

namespace phase {

inline void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw MP_Exeption(std::string(operation) + ": " + cudaGetErrorString(error));
}

struct Slab {
    int device, firstZ;
    Grid grid;
    Populations banks[2] = {};
    Fields fields = {};
    LBE_T* populationStorage = nullptr;
    LBE_T* fieldStorage = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;

    Slab(int id, int first, Grid g) : device(id), firstZ(first), grid(g) {}
    Slab(const Slab&) = delete;
    Slab& operator=(const Slab&) = delete;

    void allocate() {
        check(cudaSetDevice(device), "select phase GPU");
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create phase stream");
        check(cudaEventCreate(&start), "create start event");
        check(cudaEventCreate(&stop), "create stop event");
        const size_t sites = size_t(grid.nx) * grid.ny * grid.nz;
        check(cudaMalloc(&populationStorage, 2*19*sites*sizeof(LBE_T)), "allocate D3Q19 populations");
        check(cudaMalloc(&fieldStorage, 5*sites*sizeof(LBE_T)), "allocate D3Q19 fields");
        LBE_T* next = populationStorage;
        for (auto& bank : banks)
            for (auto& direction : bank.f) {
                direction = next;
                next += sites;
            }
        fields = {fieldStorage, fieldStorage+sites, fieldStorage+2*sites, fieldStorage+3*sites, fieldStorage+4*sites};
    }

    ~Slab() {
        if (cudaSetDevice(device) != cudaSuccess) return;
        if (stream) cudaStreamDestroy(stream);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (populationStorage) cudaFree(populationStorage);
        if (fieldStorage) cudaFree(fieldStorage);
    }
};

// Copy a full xy plane. Only crossing populations and the potential need z halos.
template<typename T>
inline void copy_face(Slab& dst, T* to, const Slab& src, const T* from, bool low) {
    const size_t plane = size_t(dst.grid.nx) * dst.grid.ny;
    to += (low ? 0 : dst.grid.nz-1)*plane;
    from += (low ? src.grid.nz-2 : 1)*plane;
    if (dst.device == src.device) {
        check(cudaMemcpyAsync(to,from,plane*sizeof(T),cudaMemcpyDeviceToDevice,dst.stream), "copy local z face");
    } else {
        check(cudaMemcpyPeerAsync(to,dst.device,from,src.device,plane*sizeof(T),dst.stream), "copy peer z face");
    }
}

template<bool Walls>
class Solver {
public:
    void reset(int nx, int ny, int nz, const MP_Parallel& parallel) {
        if (nx < 3 || ny < 3 || nz < 3 || size_t(nx)*ny > size_t(INT_MAX)/nz)
            throw MP_Exeption("Invalid D3Q19 dimensions including the ghost layers");
        if (parallel.GPU_pool.empty()) throw MP_Exeption("No GPU selected for D3Q19");
        const size_t count = std::min(parallel.GPU_pool.size(), size_t(nz-2));
        bool reuse = domain.nx == nx && domain.ny == ny && domain.nz == nz && slabs.size() == count;
        for (size_t i = 0; reuse && i < count; ++i)
            reuse = size_t(slabs[i]->device) == parallel.GPU_pool[i];
        if (!reuse) {
            std::vector<std::unique_ptr<Slab>> next;
            int z = 1;
            for (size_t i = 0; i < count; ++i) {
                const int depth = (nz-2)/int(count) + (int(i) < (nz-2)%int(count));
                next.emplace_back(new Slab(int(parallel.GPU_pool[i]), z, {nx,ny,depth+2}));
                next.back()->allocate();
                z += depth;
            }
            slabs.swap(next);
            domain = {nx,ny,nz};
        }
        bank = 0;
        first = true;
        for (size_t rank = 0; rank < count; ++rank) {
            Slab& s = *slabs[rank];
            check(cudaSetDevice(s.device), "select initialization GPU");
            const size_t sites = size_t(nx)*ny*s.grid.nz;
            initialize<<<unsigned((sites+255)/256),256,0,s.stream>>>(s.grid,s.banks[bank],s.fields,int(rank));
            check(cudaGetLastError(), "launch D3Q19 initialization");
        }
        wait_all();
    }

    float advance(int steps, LBE_T tau, LBE_T temperature, LBE_T gx, LBE_T gy) {
        if (slabs.empty()) throw MP_Exeption("Call init before cuda_calc");
        if (!(tau > 0)) throw MP_Exeption("D3Q19 tau must be positive");
        const LBE_T omega = 1/tau;
        const bool splitZ = slabs.size() > 1;
        const bool flat = false;
        for (const auto& s : slabs) {
            check(cudaSetDevice(s->device), "select timing GPU");
            check(cudaEventRecord(s->start,s->stream), "record batch start");
        }
        for (int i = 0; i < steps; ++i) {
            for (const auto& s : slabs) {
                if (splitZ) check(cudaSetDevice(s->device), "select moments GPU");
                if (flat) launch_moments<true>(*s,temperature,splitZ);
                else launch_moments<false>(*s,temperature,splitZ);
            }
            if (splitZ) exchange(true);
            for (const auto& s : slabs) {
                if (splitZ) check(cudaSetDevice(s->device), "select collision GPU");
                if (flat) launch_collision<true>(*s,omega,gx,gy,splitZ);
                else launch_collision<false>(*s,omega,gx,gy,splitZ);
            }
            bank ^= 1;
            first = false;
            if (splitZ) exchange(false);
        }
        for (const auto& s : slabs) {
            check(cudaSetDevice(s->device), "select timing GPU");
            check(cudaEventRecord(s->stop,s->stream), "record batch stop");
        }
        float elapsed = 0;
        for (const auto& s : slabs) {
            check(cudaSetDevice(s->device), "select timing GPU");
            check(cudaEventSynchronize(s->stop), "complete D3Q19 batch");
            float ms;
            check(cudaEventElapsedTime(&ms,s->start,s->stop), "time D3Q19 batch");
            elapsed = std::max(elapsed,ms);
        }
        return elapsed;
    }

    void download(h_Array3d<LBE_T>& rho, h_Array3d<LBE_T>& ux,
                  h_Array3d<LBE_T>& uy, h_Array3d<LBE_T>& uz) const {
        h_Array3d<LBE_T>* outputs[] = {&rho,&ux,&uy,&uz};
        for (auto* h : outputs)
            if (h->Nx != domain.nx || h->Ny != domain.ny || h->Nz != domain.nz)
                h->resize(domain.nx,domain.ny,domain.nz);
        const size_t plane = size_t(domain.nx)*domain.ny;
        for (const auto& s : slabs) {
            check(cudaSetDevice(s->device), "select output GPU");
            const LBE_T* src[] = {s->fields.rho,s->fields.ux,s->fields.uy,s->fields.uz};
            const size_t bytes = (s->grid.nz-2)*plane*sizeof(LBE_T);
            for (int n = 0; n < 4; ++n)
                check(cudaMemcpy(outputs[n]->p+s->firstZ*plane,src[n]+plane,bytes,cudaMemcpyDeviceToHost),
                      "download D3Q19 field");
        }
        // The public h_Array3d includes all faces, edges and corners for the GUI and file output.
        for (auto* h : outputs) {
            for (int z = 1; z < domain.nz-1; ++z) {
                LBE_T* layer = h->p+z*plane;
                if (!Walls) {
                    std::memcpy(layer,layer+(domain.ny-2)*domain.nx,domain.nx*sizeof(LBE_T));
                    std::memcpy(layer+(domain.ny-1)*domain.nx,layer+domain.nx,domain.nx*sizeof(LBE_T));
                }
                for (int y = 0; y < domain.ny; ++y) {
                    LBE_T* row = layer+y*domain.nx;
                    row[0] = row[domain.nx-2];
                    row[domain.nx-1] = row[1];
                }
            }
            std::memcpy(h->p,h->p+(domain.nz-2)*plane,plane*sizeof(LBE_T));
            std::memcpy(h->p+(domain.nz-1)*plane,h->p+plane,plane*sizeof(LBE_T));
        }
    }

private:
    friend struct SolverProbe;
    Grid domain = {};
    std::vector<std::unique_ptr<Slab>> slabs;
    int bank = 0;
    bool first = true;

    static dim3 blocks(Grid g) {
        return dim3((g.nx-2+15)/16, std::min((g.ny-(Walls ? 0 : 2)+15)/16,65535), std::min(g.nz-2,65535));
    }
    template<bool Flat> void launch_moments(Slab& s, LBE_T temperature, bool splitZ) {
        moments<Walls,Flat><<<blocks(s.grid),dim3(16,16),0,s.stream>>>(
            s.grid,s.banks[bank],s.fields,temperature,first,splitZ);
        check(cudaGetLastError(), "launch D3Q19 moments");
    }
    template<bool Flat> void launch_collision(Slab& s, LBE_T omega, LBE_T gx, LBE_T gy, bool splitZ) {
        collide<Walls,Flat><<<blocks(s.grid),dim3(16,16),0,s.stream>>>(
            s.grid,s.banks[bank],s.banks[bank^1],s.fields,omega,gx,gy,first,splitZ);
        check(cudaGetLastError(), "launch D3Q19 collision");
    }
    void wait_all() const {
        for (const auto& s : slabs) {
            check(cudaSetDevice(s->device), "select source GPU");
            check(cudaStreamSynchronize(s->stream), "finish writing source fields");
        }
    }
    void exchange(bool potentialOnly) {
        wait_all();
        const size_t count = slabs.size();
        const int up[] = {5,11,12,13,14}, down[] = {6,15,16,17,18};
        for (size_t i = 0; i < count; ++i) {
            Slab& dst = *slabs[i];
            const Slab& prev = *slabs[(i+count-1)%count];
            const Slab& next = *slabs[(i+1)%count];
            check(cudaSetDevice(dst.device), "select halo GPU");
            if (potentialOnly) {
                copy_face(dst,dst.fields.psi,prev,prev.fields.psi,true);
                copy_face(dst,dst.fields.psi,next,next.fields.psi,false);
            } else {
                for (int q : up) copy_face(dst,dst.banks[bank].f[q],prev,prev.banks[bank].f[q],true);
                for (int q : down) copy_face(dst,dst.banks[bank].f[q],next,next.banks[bank].f[q],false);
            }
        }
    }
};

inline float legacy_ticks(float ms, int nx, int ny, int nz, int steps) {
    return float(ms / (double(nx)*ny*nz*steps) * LBE_ncores * (LBE_gpuf) * 1e-3);
}

} // namespace phase
