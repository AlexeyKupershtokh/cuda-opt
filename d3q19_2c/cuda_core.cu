#include <mp/lb3d.h>
#include <mp/d_lb3d.h>
#include <mp/ini.h>

#include "cuda_core.h"

const char * fnameini="./ini.txt";

float LBE_elapsedTime;

__device__ double p_rho(LBE_T rho, LBE_T T){
    return 8*T*rho/(3-rho)-3*rho*rho;
}

__device__ LBE_T drand(int s){
	for (int i=0; i<100; i++)
		s=(s*s)%(101*127);
	return s/(101.0*127.0);
}

const MP_Parallel MP_parallel=ini_read<MP_Parallel>(fnameini, "mp_parallel");
int Nx=ini_read<int>(fnameini, "Nx", 128);
int Ny=ini_read<int>(fnameini, "Ny", 128);
int Nz=ini_read<int>(fnameini, "Nz", 128);

LBE_T T1   =ini_read<LBE_T>(fnameini, "T1", 0.85);
LBE_T T2   =ini_read<LBE_T>(fnameini, "T2", 0.8);
LBE_T tau1  =ini_read<LBE_T>(fnameini, "tau1", 0.55);
LBE_T tau2  =ini_read<LBE_T>(fnameini, "tau2", 0.55);
LBE_T betta=ini_read<LBE_T>(fnameini, "betta", 0.1);
LBE_T G    =ini_read<LBE_T>(fnameini, "G", -0.005);
LBE_T gy   =ini_read<LBE_T>(fnameini, "gy", 5e-6);
LBE_T gx   =ini_read<LBE_T>(fnameini, "gx", 0.0);

LBE_T Uxu   =ini_read<LBE_T>(fnameini, "Uxu", 0.0);
LBE_T Uxd   =ini_read<LBE_T>(fnameini, "Uxd", 0.0);

d_LB3d lb1(Nx, Ny, Nz);
d_LB3d lb2(Nx, Ny, Nz);

h_Array3d<LBE_T> rho3d1(Nx,Ny,Nz), rho3d2(Nx,Ny,Nz), ux1(Nx, Ny, Nz), uy1(Nx, Ny, Nz), uz1(Nx, Ny, Nz), ux2(Nx, Ny, Nz), uy2(Nx, Ny, Nz), uz2(Nx, Ny, Nz);

d_Array3d<LBE_T> fx(Nx,Ny,Nz), fy(Nx,Ny,Nz), fz(Nx,Ny,Nz);
d_Array3d<LBE_T> psi(Nx, Ny, Nz);

//    CUP FORALL_3D_IJK lb1 lb1 lb2 lb2 Nz Nz ux0 0.0 # 
template<typename T1, typename T2, typename T3, typename T4> __global__ void CUP_3D_4_core0(int gpuIdx, T1 lb1, T2 lb2, T3 Nz, T4 ux0){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t j=blockIdx.y*blockDim.y+threadIdx.y;
const size_t k=blockIdx.z*blockDim.z+threadIdx.z;
const size_t ind=k*lb1.Nx*lb1.Ny+j*lb1.Nx+i;

int k1=k+gpuIdx*lb1.Nz; 
lb1.rho[ind] *=(1.0 + 0.2*(drand(k1+ind)-0.5)); 
lb2.rho[ind] *=(1.0 + 0.2*(drand(k1+ind+1234567)-0.5)); 
}

//		CUP FORALL_3D lb1 lb1 lb2 lb2 fx fx fy fy fz fz psi psi betta betta T T1 gx gx gy gy # 
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10> __global__ void CUP_3D_10_core1(int gpuIdx, T1 lb1, T2 lb2, T3 fx, T4 fy, T5 fz, T6 psi, T7 betta, T8 T, T9 gx, T10 gy){const size_t ind=(blockIdx.z*blockDim.z+threadIdx.z)*lb1.Nx*lb1.Ny+(blockIdx.y*blockDim.y+threadIdx.y)*lb1.Nx +   blockIdx.x * blockDim.x + threadIdx.x;

psi[ind]=sqrt(fabs(lb1.rho[ind]/3.0-0.02*p_rho(lb1.rho[ind], T))); 
const LBE_T tmp=betta*lb1.rho[ind]*lb2.rho[ind]; 
fx[ind]=-(lb1.ux[ind]-lb2.ux[ind])*tmp + lb1.rho[ind]*gx; 
fy[ind]=-(lb1.uy[ind]-lb2.uy[ind])*tmp + lb1.rho[ind]*gy; 
fz[ind]=-(lb1.uz[ind]-lb2.uz[ind])*tmp; 
}

//		CUP FORALL_3DXZ lb1 lb1 lb2 lb2 Nxy lb1.Nx*(lb1.Ny-1) Uxu Uxu Uxd Uxd #
template<typename T1, typename T2, typename T3, typename T4, typename T5> __global__ void CUP_3DXZ_5_core2(T1 lb1, T2 lb2, T3 Nxy, T4 Uxu, T5 Uxd){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t k=blockIdx.y*blockDim.y+threadIdx.y;
const size_t ind=k*lb1.Nx*lb1.Ny+i;

set_value(lb1, ind, lb1.rho[ind+lb1.Nx], Uxd, 0.0, 0.0); 
set_value(lb1, ind+Nxy, lb1.rho[ind+Nxy-lb1.Nx], Uxu, 0.0, 0.0); 
set_value(lb2, ind, lb2.rho[ind+lb2.Nx], Uxd, 0.0, 0.0); 
set_value(lb2, ind+Nxy, lb2.rho[ind+Nxy-lb2.Nx], Uxu, 0.0, 0.0); 
}

//		CUP FORALL_3D lb1 lb1 lb2 lb2 fx fx fy fy fz fz psi psi betta betta T T2 gx gx gy gy # 
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10> __global__ void CUP_3D_10_core3(int gpuIdx, T1 lb1, T2 lb2, T3 fx, T4 fy, T5 fz, T6 psi, T7 betta, T8 T, T9 gx, T10 gy){const size_t ind=(blockIdx.z*blockDim.z+threadIdx.z)*lb1.Nx*lb1.Ny+(blockIdx.y*blockDim.y+threadIdx.y)*lb1.Nx +   blockIdx.x * blockDim.x + threadIdx.x;

psi[ind]=sqrt(fabs(lb2.rho[ind]/3.0-0.02*p_rho(lb2.rho[ind], T))); 
const LBE_T tmp=betta*lb1.rho[ind]*lb2.rho[ind]; 
fx[ind]=(lb1.ux[ind]-lb2.ux[ind])*tmp + lb2.rho[ind]*gx; 
fy[ind]=(lb1.uy[ind]-lb2.uy[ind])*tmp + lb2.rho[ind]*gy; 
fz[ind]=(lb1.uz[ind]-lb2.uz[ind])*tmp; 
}

//		CUP FORALL_3DXZ lb1 lb1 lb2 lb2 Nxy lb1.Nx*(lb1.Ny-1) Uxu Uxu Uxd Uxd #
template<typename T1, typename T2, typename T3, typename T4, typename T5> __global__ void CUP_3DXZ_5_core4(T1 lb1, T2 lb2, T3 Nxy, T4 Uxu, T5 Uxd){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t k=blockIdx.y*blockDim.y+threadIdx.y;
const size_t ind=k*lb1.Nx*lb1.Ny+i;

set_value(lb1, ind, lb1.rho[ind+lb1.Nx], Uxd, 0.0, 0.0); 
set_value(lb1, ind+Nxy, lb1.rho[ind+Nxy-lb1.Nx], Uxu, 0.0, 0.0); 
set_value(lb2, ind, lb2.rho[ind+lb2.Nx], Uxd, 0.0, 0.0); 
set_value(lb2, ind+Nxy, lb2.rho[ind+Nxy-lb2.Nx], Uxu, 0.0, 0.0); 
}


void init(){
    setrhou(lb1, 0.5, 0.0, 0.0, 0.0);
    setrhou(lb2, 0.5, 0.0, 0.0, 0.0);
    
    ini_read(fnameini, lb1.rho, "rho1");
    ini_read(fnameini, lb2.rho, "rho2");

{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(lb1.Nx/threads.x, lb1.Ny/threads.y, ((lb1.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3D_4_core0<<<grid, threads, 0, s[i]>>>(i, split(i, lb1), split(i, lb2), split(i, Nz), split(i, 0.0));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamDestroy(s[i]);
}
}




    rhou2f(lb1);
    rhou2f(lb2);
}

void cuda_calc(){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    int NN=10;
    for(int i=0; i<NN; i++){	
		transfer(lb1);    transfer(lb2);
		f2rhou(lb1);      f2rhou(lb2);
		relax(lb1, tau1);  relax(lb2, tau2);

{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(lb1.Nx/threads.x, lb1.Ny/threads.y, ((lb1.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3D_10_core1<<<grid, threads, 0, s[i]>>>(i, split(i, lb1), split(i, lb2), split(i, fx), split(i, fy), split(i, fz), split(i, psi), split(i, betta), split(i, T1), split(i, gx), split(i, gy));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamDestroy(s[i]);
}
}






				
		sync(lb1);	sync(lb2);
		periodicz(lb1); periodicx(lb1);//periodicy(lb1); 
		periodicz(lb2); periodicx(lb2);//periodicy(lb2);
{
dim3 threads(MP_parallel.Nx, MP_parallel.Nz_bound);
dim3 grid(lb1.Nx/threads.x, ((lb1.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.y);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3DXZ_5_core2<<<grid, threads, 0, s[i]>>>(split(i, lb1), split(i, lb2), split(i, lb1.Nx*(lb1.Ny-1)), split(i, Uxu), split(i, Uxd));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamDestroy(s[i]);
}
}





		
		psi2force(fx, fy, fz, psi);
		psipsi2force(fx, fy, fz, lb1.rho, lb2.rho, G); // add force
		force(lb1, fx, fy, fz);		
		
{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(lb1.Nx/threads.x, lb1.Ny/threads.y, ((lb1.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3D_10_core3<<<grid, threads, 0, s[i]>>>(i, split(i, lb1), split(i, lb2), split(i, fx), split(i, fy), split(i, fz), split(i, psi), split(i, betta), split(i, T2), split(i, gx), split(i, gy));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamDestroy(s[i]);
}
}






		
		psi2force(fx, fy, fz, psi);
		psipsi2force(fx, fy, fz, lb2.rho, lb1.rho, G); // add force
		force(lb2, fx, fy, fz);

		sync(lb1);     sync(lb2);
		periodicz(lb1); periodicx(lb1);//eriodicy(lb1); 
		periodicz(lb2); periodicx(lb2);//eriodicy(lb2);
{
dim3 threads(MP_parallel.Nx, MP_parallel.Nz_bound);
dim3 grid(lb1.Nx/threads.x, ((lb1.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.y);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3DXZ_5_core4<<<grid, threads, 0, s[i]>>>(split(i, lb1), split(i, lb2), split(i, lb1.Nx*(lb1.Ny-1)), split(i, Uxu), split(i, Uxd));
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
 
    cpy(rho3d1, lb1.rho);
    cpy(rho3d2, lb2.rho);

    cpy(ux1, lb1.ux);
    cpy(uy1, lb1.uy);
    cpy(uz1, lb1.uz);
    cpy(ux2, lb2.ux);
    cpy(uy2, lb2.uy);
    cpy(uz2, lb2.uz);

}

