// LBE с силами
#include <mp/mp.h>
#include <mp/lb3d.h>
#include <mp/d_lb3d.h>
#include <mp/ini.h>

#include "cuda_core.h"

const char * fnameini="./ini";

float LBE_elapsedTime;

const MP_Parallel MP_parallel=ini_read<MP_Parallel>(fnameini, "mp_parallel");
int Nx=ini_read<int>(fnameini, "Nx", 128);
int Ny=ini_read<int>(fnameini, "Ny", 128);
int Nz=ini_read<int>(fnameini, "Nz", 128);
LBE_T tau=ini_read<LBE_T>(fnameini, "tau", 0.55);
LBE_T betta=ini_read<LBE_T>(fnameini, "betta", 0.1);

d_LB3d lb(Nx, Ny, Nz);
h_Array3d<LBE_T> rho(Nx, Ny, Nz), ux(Nx, Ny, Nz), uy(Nx, Ny, Nz), uz(Nx, Ny, Nz);
d_Array3d<LBE_T> fx(Nx, Ny, Nz), fy(Nx, Ny, Nz), fz(Nx, Ny, Nz);
d_Array3d<LBE_T> rho2(Nx, Ny, Nz), rho_tmp(Nx, Ny, Nz);

//	CUP FORALL_3D rho2 rho2 rho_tmp rho_tmp # 
template<typename T1, typename T2> __global__ void CUP_3D_2_core0(int gpuIdx, T1 rho2, T2 rho_tmp){const size_t ind=(blockIdx.z*blockDim.z+threadIdx.z)*rho2.Nx*rho2.Ny+(blockIdx.y*blockDim.y+threadIdx.y)*rho2.Nx +   blockIdx.x * blockDim.x + threadIdx.x;

rho_tmp[ind]=rho2[ind]=0.0; 
}

//		CUP FORALL_3D_IN A rho2 B rho_tmp Nx rho.Nx Ny rho.Ny Nxy rho.Nx*rho.Ny # B[ind]=1.0/6*(A[ind-1]+A[ind+1]+A[ind-Nx]+A[ind+Nx]+A[ind-Nxy]+A[ind+Nxy]); #
template<typename T1, typename T2, typename T3, typename T4, typename T5> __global__ void CUP_3D_5_core1(int gpuIdx, T1 A, T2 B, T3 Nx, T4 Ny, T5 Nxy){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t j=blockIdx.y*blockDim.y+threadIdx.y;
const size_t k=blockIdx.z*blockDim.z+threadIdx.z;
if (i>0 && i < A.Nx-1 && j>0 && j < A.Ny-1 && k>0 && k < A.Nz-1){
const size_t ind=k*A.Nx*A.Ny+j*A.Nx+i;
B[ind]=1.0/6*(A[ind-1]+A[ind+1]+A[ind-Nx]+A[ind+Nx]+A[ind-Nxy]+A[ind+Nxy]); }
}

//		CUP FORALL_3D_IN A rho2 B rho_tmp Nx rho.Nx Ny rho.Ny Nxy rho.Nx*rho.Ny # A[ind]=1.0/6*(B[ind-1]+B[ind+1]+B[ind-Nx]+B[ind+Nx]+B[ind-Nxy]+B[ind+Nxy]); #
template<typename T1, typename T2, typename T3, typename T4, typename T5> __global__ void CUP_3D_5_core2(int gpuIdx, T1 A, T2 B, T3 Nx, T4 Ny, T5 Nxy){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t j=blockIdx.y*blockDim.y+threadIdx.y;
const size_t k=blockIdx.z*blockDim.z+threadIdx.z;
if (i>0 && i < A.Nx-1 && j>0 && j < A.Ny-1 && k>0 && k < A.Nz-1){
const size_t ind=k*A.Nx*A.Ny+j*A.Nx+i;
A[ind]=1.0/6*(B[ind-1]+B[ind+1]+B[ind-Nx]+B[ind+Nx]+B[ind-Nxy]+B[ind+Nxy]); }
}

//	CUP FORALL_3D lb lb rho2 rho2 #
template<typename T1, typename T2> __global__ void CUP_3D_2_core3(int gpuIdx, T1 lb, T2 rho2){const size_t ind=(blockIdx.z*blockDim.z+threadIdx.z)*lb.Nx*lb.Ny+(blockIdx.y*blockDim.y+threadIdx.y)*lb.Nx +   blockIdx.x * blockDim.x + threadIdx.x;

lb.rho[ind]=1.0-rho2[ind]+0.002; 
}

//		CUP FORALL_3D rho lb.rho rho2 rho2 ux lb.ux uy lb.uy uz lb.uz fx fx fy fy fz fz betta betta # 
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __global__ void CUP_3D_9_core4(int gpuIdx, T1 rho, T2 rho2, T3 ux, T4 uy, T5 uz, T6 fx, T7 fy, T8 fz, T9 betta){const size_t ind=(blockIdx.z*blockDim.z+threadIdx.z)*rho.Nx*rho.Ny+(blockIdx.y*blockDim.y+threadIdx.y)*rho.Nx +   blockIdx.x * blockDim.x + threadIdx.x;

const LBE_T tmp=betta*rho[ind]*rho2[ind]; 
fx[ind]=-ux[ind]*tmp; 
fy[ind]=-uy[ind]*tmp; 
fz[ind]=-uz[ind]*tmp; 
}


void init(){
	ini_read(fnameini, tau, "tau");

	setrhou(lb, 1.0, 0.05, 0.0, 0.0);
		
{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(rho2.Nx/threads.x, rho2.Ny/threads.y, ((rho2.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3D_2_core0<<<grid, threads, 0, s[i]>>>(i, split(i, rho2), split(i, rho_tmp));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamDestroy(s[i]);
}
}



	ini_read(fnameini, rho2, "rho2");
	ini_read(fnameini, lb.ux, "ux");

	for(size_t i=0; i<5; i++){
{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(rho2.Nx/threads.x, rho2.Ny/threads.y, ((rho2.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3D_5_core1<<<grid, threads, 0, s[i]>>>(i, split(i, rho2), split(i, rho_tmp), split(i, rho.Nx), split(i, rho.Ny), split(i, rho.Nx*rho.Ny));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamDestroy(s[i]);
}
}
		sync(rho_tmp);
		periodicz(rho_tmp); periodicx(rho_tmp); periodicy(rho_tmp);
{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(rho2.Nx/threads.x, rho2.Ny/threads.y, ((rho2.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3D_5_core2<<<grid, threads, 0, s[i]>>>(i, split(i, rho2), split(i, rho_tmp), split(i, rho.Nx), split(i, rho.Ny), split(i, rho.Nx*rho.Ny));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamDestroy(s[i]);
}
}
		sync(rho2);
		periodicz(rho2); periodicx(rho2); periodicy(rho2);
	}

{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(lb.Nx/threads.x, lb.Ny/threads.y, ((lb.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3D_2_core3<<<grid, threads, 0, s[i]>>>(i, split(i, lb), split(i, rho2));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamDestroy(s[i]);
}
}


  	rhou2f(lb);
}

void cuda_calc(){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    int NN=30;
    for(int i=0; i<NN; i++){	
		transfer(lb);
		f2rhou(lb);
		relax(lb, tau);		
{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(lb.rho.Nx/threads.x, lb.rho.Ny/threads.y, ((lb.rho.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3D_9_core4<<<grid, threads, 0, s[i]>>>(i, split(i, lb.rho), split(i, rho2), split(i, lb.ux), split(i, lb.uy), split(i, lb.uz), split(i, fx), split(i, fy), split(i, fz), split(i, betta));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamDestroy(s[i]);
}
}





		sync(lb.rho); periodicz(lb); periodicx(lb); periodicy(lb);
		psipsi2force(fx, fy, fz, lb.rho, rho2, -0.4); // add force
		force(lb, fx, fy, fz);
		f2rhou(lb);
		sync(lb);
		periodicz(lb); periodicx(lb); periodicy(lb);
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

