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

double rhol=1.0, ul=0.0, rhor=0.5, ur=0.0;
double T=0.3;


float LBE_elapsedTime;

d_LB3d lb;
d_Array3d<LBE_T> psi(Nx, Ny, Nz);   // Массив для вычисления сил

h_Array3d<LBE_T> rho(Nx,Ny,Nz), ux(Nx,Ny,Nz), uy(Nx, Ny, Nz), uz(Nx, Ny, Nz);

//    CUP FORALL_3D_IJK lb lb rhol rhol ul ul rhor rhor ur ur # 		
template<typename T1, typename T2, typename T3, typename T4, typename T5> __global__ void CUP_3D_5_core0(int gpuIdx, T1 lb, T2 rhol, T3 ul, T4 rhor, T5 ur){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t j=blockIdx.y*blockDim.y+threadIdx.y;
const size_t k=blockIdx.z*blockDim.z+threadIdx.z;
const size_t ind=k*lb.Nx*lb.Ny+j*lb.Nx+i;

if( i < lb.Nx/2 ) { 
lb.rho[ind]=rhol; 
lb.ux[ind]=ul; 
}else{ 
lb.rho[ind]=rhor; 
lb.ux[ind]=ur; 
}; 
}

//		CUP FORALL_3DYZ lb lb rhol rhol rhor rhor ul ul ur ur #
template<typename T1, typename T2, typename T3, typename T4, typename T5> __global__ void CUP_3DYZ_5_core1(T1 lb, T2 rhol, T3 rhor, T4 ul, T5 ur){const size_t j=blockIdx.x*blockDim.x+threadIdx.x;
const size_t k=blockIdx.y*blockDim.y+threadIdx.y;
const size_t ind=k*lb.Nx*lb.Ny+j*lb.Nx;

set_value(lb, ind, rhol, ul, 0.0, 0.0); 
set_value(lb, ind+lb.Nx-1, rhor, ur, 0.0, 0.0); 
}

//		CUP FORALL_3D rho lb.rho psi psi T T # 
template<typename T1, typename T2, typename T3> __global__ void CUP_3D_3_core2(int gpuIdx, T1 rho, T2 psi, T3 T){const size_t ind=(blockIdx.z*blockDim.z+threadIdx.z)*rho.Nx*rho.Ny+(blockIdx.y*blockDim.y+threadIdx.y)*rho.Nx +   blockIdx.x * blockDim.x + threadIdx.x;

psi[ind]=sqrt(fabs((rho[ind]*(1.0/3.0-T)))); 
}

//		CUP FORALL_3DYZ lb lb rhol rhol rhor rhor ul ul ur ur #
template<typename T1, typename T2, typename T3, typename T4, typename T5> __global__ void CUP_3DYZ_5_core3(T1 lb, T2 rhol, T3 rhor, T4 ul, T5 ur){const size_t j=blockIdx.x*blockDim.x+threadIdx.x;
const size_t k=blockIdx.y*blockDim.y+threadIdx.y;
const size_t ind=k*lb.Nx*lb.Ny+j*lb.Nx;

set_value(lb, ind, rhol, ul, 0.0, 0.0); 
set_value(lb, ind+lb.Nx-1, rhor, ur, 0.0, 0.0); 
}


void resize(){
	lb.resize(Nx, Ny, Nz);
	psi.resize(Nx, Ny, Nz);
	rho.resize(Nx,Ny,Nz);
	ux.resize(Nx,Ny,Nz);
	uy.resize(Nx, Ny, Nz);
	uz.resize(Nx, Ny, Nz);
}

void init(){
	setrhou(lb, 1.0, 0.0, 0.0, 0.0);

{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(lb.Nx/threads.x, lb.Ny/threads.y, ((lb.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3D_5_core0<<<grid, threads, 0, s[i]>>>(i, split(i, lb), split(i, rhol), split(i, ul), split(i, rhor), split(i, ur));
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
    int NN=10;
    for(int i=0; i<NN; i++){
		transfer(lb);
	   	f2rhou(lb);
    	relax(lb, tau);
		sync(lb);

{
dim3 threads(MP_parallel.Ny, MP_parallel.Nz_bound);
dim3 grid(lb.Ny/threads.x, ((lb.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.y);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3DYZ_5_core1<<<grid, threads, 0, s[i]>>>(split(i, lb), split(i, rhol), split(i, rhor), split(i, ul), split(i, ur));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamDestroy(s[i]);
}
}



		periodicz(lb); periodicy(lb);

		// Вычисление сил для уравнения состояния
{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(lb.rho.Nx/threads.x, lb.rho.Ny/threads.y, ((lb.rho.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3D_3_core2<<<grid, threads, 0, s[i]>>>(i, split(i, lb.rho), split(i, psi), split(i, T));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamDestroy(s[i]);
}
}


		forcepsi(lb, psi, T < 1.0/3.0? 1.0 : -1.0 );
		sync(lb);

{
dim3 threads(MP_parallel.Ny, MP_parallel.Nz_bound);
dim3 grid(lb.Ny/threads.x, ((lb.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.y);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3DYZ_5_core3<<<grid, threads, 0, s[i]>>>(split(i, lb), split(i, rhol), split(i, rhor), split(i, ul), split(i, ur));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamDestroy(s[i]);
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

