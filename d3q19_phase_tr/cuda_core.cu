#include <mp/mp.h>
#include <mp/lb3d.h>
#include <mp/d_lb3d.h>

#include "cuda_core.h"

int Nx=256*1;
int Ny=256*1;
int Nz=6+2;
//int Nz=256+128+64+32;

LBE_T T=0.3;
float LBE_elapsedTime;


__device__ double p_rho(LBE_T rho, LBE_T T){
    return 8*T*rho/(3-rho)-3*rho*rho;
}

__device__ LBE_T drand(int s){
	for (int i=0; i<100; i++)
		s=(s*s)%(101*127);
	return s/(101.0*127.0);
}

//int a[1]={0};
//MP_Parallel MP_parallel(a, 1);
const MP_Parallel MP_parallel;


d_LB3d *lbp;// (Nx, Ny, Nz);
d_Array3d<LBE_T> *psip;   // Массив для вычисления сил

h_Array3d<LBE_T> rho(Nx,Ny,Nz), ux(Nx,Ny,Nz), uy(Nx, Ny, Nz), uz(Nx, Ny, Nz);

//    CUP FORALL_3D_IJK lb lb Nz lb.Nz # 
template<typename T1, typename T2> __global__ void CUP_3D_2_core0(int gpuIdx, T1 lb, T2 Nz){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t j=blockIdx.y*blockDim.y+threadIdx.y;
const size_t k=blockIdx.z*blockDim.z+threadIdx.z;
const size_t ind=k*lb.Nx*lb.Ny+j*lb.Nx+i;

int k1=ind+gpuIdx*lb.Nz; 
lb.rho[ind]=0.8 + (drand(k1)-0.5)*0.2; 
}

//	CUP FORALL_3D rho lb.rho psi psi T T # 
template<typename T1, typename T2, typename T3> __global__ void CUP_3D_3_core1(int gpuIdx, T1 rho, T2 psi, T3 T){const size_t ind=(blockIdx.z*blockDim.z+threadIdx.z)*rho.Nx*rho.Ny+(blockIdx.y*blockDim.y+threadIdx.y)*rho.Nx +   blockIdx.x * blockDim.x + threadIdx.x;

psi[ind]=sqrt(fabs(rho[ind]/3.0-0.02*p_rho(rho[ind], 0.85))); 
}


void init(){
    lbp=new d_LB3d(Nx, Ny, Nz);
    psip=new d_Array3d<LBE_T>(Nx, Ny, Nz);

    d_LB3d &lb=*lbp;
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
    d_LB3d &lb=*lbp;
    d_Array3d<LBE_T> &psi=*psip;

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
	periodicz(lb); periodicx(lb); periodicy(lb);
	
	// Вычисление сил для уравнения состояния
{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(lb.rho.Nx/threads.x, lb.rho.Ny/threads.y, ((lb.rho.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
s[i] = 0;
CUP_3D_3_core1<<<grid, threads, 0, s[i]>>>(i, split(i, lb.rho), split(i, psi), split(i, T));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);

}
}


	forcepsi(lb, psi);
	sync(lb);
	periodicx(lb); periodicy(lb); periodicz(lb);
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
