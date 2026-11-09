#include <mp/mp.h>
#include <mp/lb3d.h>
#include <mp/d_lb3d.h>
#include <mp/ini.h>

#include "cuda_core.h"

const char * fnameini="./ini";
const MP_Parallel MP_parallel=ini_read<MP_Parallel>(fnameini, "mp_parallel");
int Nx=ini_read<int>(fnameini, "Nx", 1024);
int Ny=ini_read<int>(fnameini, "Ny", 16);
int Nz=ini_read<int>(fnameini, "Nz", 6);
LBE_T tau=ini_read<LBE_T>(fnameini, "tau", 0.55);
LBE_T   T=ini_read<LBE_T>(fnameini, "T", 0.85);
LBE_T   gy=ini_read<LBE_T>(fnameini, "gy", 5e-6);
LBE_T   gx=ini_read<LBE_T>(fnameini, "gx", 0.0);

float LBE_elapsedTime;

__device__ double p_rho(LBE_T rho, LBE_T T){
    return 8*T*rho/(3-rho)-3*rho*rho;
}

__device__ LBE_T drand(int s){
	for (int i=0; i<100; i++)
		s=(s*s)%(101*127);
	return s/(101.0*127.0);
}

d_LB3d lb(Nx, Ny, Nz);
d_Array3d<LBE_T> psi(Nx, Ny, Nz);   // Массив для вычисления сил
d_Array3d<LBE_T> fx(Nx, Ny, Nz), fy(Nx, Ny, Nz), fz(Nx, Ny, Nz);  

h_Array3d<LBE_T> rho(Nx,Ny,Nz), ux(Nx,Ny,Nz), uy(Nx, Ny, Nz), uz(Nx, Ny, Nz);

//    CUP FORALL_3D_IJK lb lb Nz lb.Nz # 
template<typename T1, typename T2> __global__ void CUP_3D_2_core0(int gpuIdx, T1 lb, T2 Nz){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t j=blockIdx.y*blockDim.y+threadIdx.y;
const size_t k=blockIdx.z*blockDim.z+threadIdx.z;
const size_t ind=k*lb.Nx*lb.Ny+j*lb.Nx+i;

int k1=ind+gpuIdx*lb.Nz; 
lb.rho[ind]=0.8 + (drand(k1)-0.5)*0.2; 
}

//		CUP FORALL_3DXZ lb lb Nxy lb.Nx*(lb.Ny-1) #
template<typename T1, typename T2> __global__ void CUP_3DXZ_2_core1(T1 lb, T2 Nxy){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t k=blockIdx.y*blockDim.y+threadIdx.y;
const size_t ind=k*lb.Nx*lb.Ny+i;

set_value(lb, ind, lb.rho[ind+lb.Nx], 0.0, 0.0, 0.0); 
set_value(lb, ind+Nxy, lb.rho[ind+Nxy-lb.Nx], 0.0, 0.0, 0.0); 
}

//		CUP FORALL_3D rho lb.rho psi psi fx fx fy fy fz fz gx gx gy gy T T # 
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __global__ void CUP_3D_8_core2(int gpuIdx, T1 rho, T2 psi, T3 fx, T4 fy, T5 fz, T6 gx, T7 gy, T8 T){const size_t ind=(blockIdx.z*blockDim.z+threadIdx.z)*rho.Nx*rho.Ny+(blockIdx.y*blockDim.y+threadIdx.y)*rho.Nx +   blockIdx.x * blockDim.x + threadIdx.x;

psi[ind]=sqrt(fabs(rho[ind]/3.0-0.02*p_rho(rho[ind], T))); 
fx[ind]=gx; 
fy[ind]=gy*rho[ind]; 
fz[ind]=0.0; 
}

//		CUP FORALL_3DXZ lb lb Nxy lb.Nx*(lb.Ny-1) #
template<typename T1, typename T2> __global__ void CUP_3DXZ_2_core3(T1 lb, T2 Nxy){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t k=blockIdx.y*blockDim.y+threadIdx.y;
const size_t ind=k*lb.Nx*lb.Ny+i;

set_value(lb, ind, lb.rho[ind+lb.Nx], 0.0, 0.0, 0.0); 
set_value(lb, ind+Nxy, lb.rho[ind+Nxy-lb.Nx], 0.0, 0.0, 0.0); 
}


void init(){
    setrhou(lb, 1.0, 0.0, 0.0, 0.0);

{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(lb.Nx/threads.x, lb.Ny/threads.y, ((lb.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
s[i] = 0;
CUP_3D_2_core0<<<grid, threads, 0, s[i]>>>(i, split(i, lb), split(i, lb.Nz));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);

}
}



    rhou2f(lb);
}

void cuda_calc(){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    int NN=10;
    for(int i=0; i<NN; i++){
		transfer(lb);
		f2rhou(lb);
		relax(lb, 0.6);
		sync(lb);
		periodicz(lb); periodicx(lb); //periodicy(lb);
{
dim3 threads(MP_parallel.Nx, MP_parallel.Nz_bound);
dim3 grid(lb.Nx/threads.x, ((lb.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.y);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
s[i] = 0;
CUP_3DXZ_2_core1<<<grid, threads, 0, s[i]>>>(split(i, lb), split(i, lb.Nx*(lb.Ny-1)));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);

}
}



	
		// Вычисление сил для уравнения состояния
{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(lb.rho.Nx/threads.x, lb.rho.Ny/threads.y, ((lb.rho.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
s[i] = 0;
CUP_3D_8_core2<<<grid, threads, 0, s[i]>>>(i, split(i, lb.rho), split(i, psi), split(i, fx), split(i, fy), split(i, fz), split(i, gx), split(i, gy), split(i, T));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);

}
}





		psi2force(fx, fy, fz, psi);
		force(lb, fx, fy, fz);
		sync(lb);
		periodicz(lb); periodicx(lb); // periodicy(lb);
{
dim3 threads(MP_parallel.Nx, MP_parallel.Nz_bound);
dim3 grid(lb.Nx/threads.x, ((lb.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.y);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
s[i] = 0;
CUP_3DXZ_2_core3<<<grid, threads, 0, s[i]>>>(split(i, lb), split(i, lb.Nx*(lb.Ny-1)));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);

}
}



	
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime_;
    cudaEventElapsedTime(&elapsedTime_, start, stop);

    LBE_elapsedTime=elapsedTime_/(Nx*Ny*Nz*NN)*LBE_ncores*LBE_gpuf*1e-3;
 
    cpy(rho, lb.rho);
    cpy(ux, lb.ux);
    cpy(uy, lb.uy);
    cpy(uz, lb.uz);
}
