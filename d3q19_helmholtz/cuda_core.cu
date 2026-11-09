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


d_LB3d lb(Nx, Ny, Nz);

h_Array3d<LBE_T> rho(Nx,Ny,Nz), ux(Nx,Ny,Nz), uy(Nx, Ny, Nz), uz(Nx, Ny, Nz);

LBE_T UX0=0.15;

//    CUP FORALL_3D_IJK lb lb Nz Nz UX0 UX0 # 
template<typename T1, typename T2, typename T3> __global__ void CUP_3D_3_core0(int gpuIdx, T1 lb, T2 Nz, T3 UX0){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t j=blockIdx.y*blockDim.y+threadIdx.y;
const size_t k=blockIdx.z*blockDim.z+threadIdx.z;
const size_t ind=k*lb.Nx*lb.Ny+j*lb.Nx+i;

int k1=k+gpuIdx*lb.Nz; 
//if( Nz/2 < k1 ) 
//lb.ux[ind]=UX0; 
// else 
// lb.ux[ind]=-UX0; 

if( k1 >Nz/2-10 && k1 <Nz/2+10) 
lb.uz[ind]=0.1*sin(i*2*3.1415*0.01); 
}

//		CUP FORALL_3DXY lb lb UX0 UX0 GPU_UP MP_parallel.GPU_pool.size()-1 #
template<typename T1, typename T2, typename T3> __global__ void CUP_3DXY_3_core1(int gpuIdx, T1 lb, T2 UX0, T3 GPU_UP){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t j=blockIdx.y*blockDim.y+threadIdx.y;
const size_t ind=i+j*lb.Nx;

if (gpuIdx==0) set_value(lb, ind, 1.0, -UX0, 0.0, 0.0); 
if (gpuIdx==GPU_UP) set_value(lb, ind+lb.Nx*lb.Ny*(lb.Nz-1), 1.0, UX0, 0.0, 0.0); 
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
CUP_3D_3_core0<<<grid, threads, 0, s[i]>>>(i, split(i, lb), split(i, Nz), split(i, UX0));
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
    	relax(lb,0.55);    	
		sync(lb);
		periodicx(lb); periodicy(lb);
{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny);
dim3 grid(lb.Nx/threads.x, lb.Ny/threads.y);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3DXY_3_core1<<<grid, threads, 0, s[i]>>>(i, split(i, lb), split(i, UX0), split(i, MP_parallel.GPU_pool.size()-1));
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

