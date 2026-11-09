// #define MP_2GPU
 #define MP_1GPU

#include <mp/lb3d.h>

#include <mp/d_lb3d.h>
#include <mp/d_lb3d.h>

#include "cuda_core.h"

int Nx=128*2;
int Ny=128*2;
int Nz=16;

//int Nz=256+128+64+32;

float LBE_elapsedTime;

d_LB3d *lbp1;// (Nx, Ny, Nz);
d_LB3d *lbp2;// (Nx, Ny, Nz);

h_Array3d<LBE_T> rho(Nx,Ny,Nz), rho3d1(Nx,Ny,Nz), rho3d2(Nx,Ny,Nz), ux(Nx,Ny,Nz), uy(Nx, Ny, Nz), uz(Nx, Ny, Nz);
d_Array3d<LBE_T> rho12(Nx,Ny,Nz), ux12(Nx,Ny,Nz), uy12(Nx, Ny, Nz), uz12(Nx, Ny, Nz);
d_Array3d<LBE_T> alpha(Nx,Ny,Nz);

//     CUP FORALL_3D_IJK lb1 lb1 lb2 lb2 ux0 0.1 # 
template<typename T1, typename T2, typename T3> __global__ void CUP_3D_3_core0(int gpuIdx, T1 lb1, T2 lb2, T3 ux0){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t j=blockIdx.y*blockDim.y+threadIdx.y;
const size_t k=blockIdx.z*blockDim.z+threadIdx.z;
const size_t ind=k*lb1.Nx*lb1.Ny+j*lb1.Nx+i;

int k1=k, Nz=lb1.Nz; if (gpuIdx == 1){ k1+=lb1.Nz; Nz*=2;} if( lb1.Ny/4< j && j < 3*lb1.Ny/4 && lb1.Nx/4< i && i < 3*lb1.Nx/4 /* && Nz/4 < k1 && k1 < 3*Nz/4*/ ) {lb1.rho[ind]=0.99; lb2.rho[ind]=0.01; /*lb1.ux[ind]=lb2.ux[ind]=ux0; */ }else{ lb1.rho[ind]=0.01; lb2.rho[ind]=0.99; /*lb1.ux[ind]=lb2.ux[ind]=-ux0;*/ }; }

//    	CUP FORALL_3D rho rho12 ux ux12 uy uy12 uz uz12 rho1 lb1.rho rho2 lb2.rho ux1 lb1.ux uy1 lb1.uy uz1 lb1.uz ux2 lb2.ux uy2 lb2.uy uz2 lb2.uz  # 
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12> __global__ void CUP_3D_12_core1(int gpuIdx, T1 rho, T2 ux, T3 uy, T4 uz, T5 rho1, T6 rho2, T7 ux1, T8 uy1, T9 uz1, T10 ux2, T11 uy2, T12 uz2){const size_t ind=(blockIdx.z*blockDim.z+threadIdx.z)*rho.Nx*rho.Ny+(blockIdx.y*blockDim.y+threadIdx.y)*rho.Nx +   blockIdx.x * blockDim.x + threadIdx.x;

rho[ind]=rho1[ind]+rho2[ind]; 
ux[ind]=(rho1[ind]*ux1[ind]+rho2[ind]*ux2[ind])/rho[ind]; 
uy[ind]=(rho1[ind]*uy1[ind]+rho2[ind]*uy2[ind])/rho[ind]; 
uz[ind]=(rho1[ind]*uz1[ind]+rho2[ind]*uz2[ind])/rho[ind]; 
}


void init(){
    lbp1=new d_LB3d(Nx, Ny, Nz);
    lbp2=new d_LB3d(Nx, Ny, Nz);
    d_LB3d &lb1=*lbp1;
    d_LB3d &lb2=*lbp2;
    setrhou(lb1, 0.5, 0.0, 0.0, 0.0);
    setrhou(lb2, 0.5, 0.0, 0.0, 0.0);
    
     // CUP FORALL_3D_IJK lb lb # int k1=k; if (gpuIdx == 1) k1+=256/2; if( lb.Nz/2 < k1 && k1 < 4*lb.Nz/3 && lb.Nx/2 < i && i < 3*lb.Nx/4 && lb.Ny/3 < j && j < 2*lb.Ny/3 ) {lb.rho[ind]=1.0;}else{lb.rho[ind]=0.5;};
{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(lb1.Nx/threads.x, lb1.Ny/threads.y, ((lb1.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3D_3_core0<<<grid, threads, 0, s[i]>>>(i, split(i, lb1), split(i, lb2), split(i, 0.1));
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
    d_LB3d &lb1=*lbp1;
    d_LB3d &lb2=*lbp2;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    int NN=10;
    for(int i=0; i<NN; i++){	
	transfer(lb1);  transfer(lb2);

    	f2rhou(lb1);    f2rhou(lb2);
	periodicx(lb1); periodicy(lb1); periodicz(lb1);  periodicx(lb2); periodicy(lb2); periodicz(lb2);
	force(lb1, lb2.rho, -0.002); force(lb2, lb1.rho, -0.002);

	f2rhou(lb1);    f2rhou(lb2);
	periodicx(lb1); periodicy(lb1); periodicz(lb1);  periodicx(lb2); periodicy(lb2); periodicz(lb2);
	force(lb1, 0.1, -0.0); force(lb2, 0.1, 0.0);
	f2rhou(lb1);    f2rhou(lb2);

{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(rho12.Nx/threads.x, rho12.Ny/threads.y, ((rho12.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3D_12_core1<<<grid, threads, 0, s[i]>>>(i, split(i, rho12), split(i, ux12), split(i, uy12), split(i, uz12), split(i, lb1.rho), split(i, lb2.rho), split(i, lb1.ux), split(i, lb1.uy), split(i, lb1.uz), split(i, lb2.ux), split(i, lb2.uy), split(i, lb2.uz));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamDestroy(s[i]);
}
}





	
    	relax(lb1, ux12, uy12, uz12, 0.6);  relax(lb2, ux12, uy12, uz12, 0.6);
	sync(lb1);     sync(lb2);
	periodicx(lb1); periodicy(lb1); periodicz(lb1);  periodicx(lb2); periodicy(lb2); periodicz(lb2);
    	
	

	sync(lb1);     sync(lb2);
	periodicx(lb1); periodicy(lb1); periodicz(lb1);  periodicx(lb2); periodicy(lb2); periodicz(lb2);
    		   
	   
	// periodicxyz(lb1); periodicxyz(lb2);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime_;	
    cudaEventElapsedTime(&elapsedTime_, start, stop);	

    LBE_elapsedTime=elapsedTime_/(Nx*Ny*Nz*NN)*LBE_ncores*LBE_gpuf*1e-3;
    
    #ifdef MP_2GPU
	if (MP_DEVICE1 !=MP_DEVICE2) LBE_elapsedTime*=2;
    #endif

    cpy(rho, lb1.rho);
    cpy(rho3d1, lb1.rho);
    cpy(rho3d2, lb2.rho);
    cpy(ux, ux12);
    cpy(uy, uy12);
    cpy(uz, uz12);

}

