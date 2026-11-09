#include <iostream>
#include <fstream>

#include <mp/mp.h>
#include <mp/rdtsc.h>
#include <mp/d_array3d.h>

const int Nx=256;  // кратно 16
const int Ny=256;  // кратно 16
const int Nz=256;  // // Nz-2 кратно количеству GPU
const int Nxy=Nx*Ny;

using namespace std;

typedef double T;
#define m 0.16666666666666666666666666666667

int a[3]={0,1,2};   // массив с номерами устройств GPU (device)
MP_Parallel MP_parallel(a, 3);

// Обрабатывается препроцессором. В этом меcте появятся определения функций
//	CUP FORALL_3D_IJK a a tmp tmp #
template<typename T1, typename T2> __global__ void CUP_3D_2_lapl3d0(int gpuIdx, T1 a, T2 tmp){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t j=blockIdx.y*blockDim.y+threadIdx.y;
const size_t k=blockIdx.z*blockDim.z+threadIdx.z;
const size_t ind=k*a.Nx*a.Ny+j*a.Nx+i;

const size_t k0=k+gpuIdx*(a.Nz-2); 
if (i>a.Nx/3 && i < 2*a.Nx/3 && j>a.Ny/3 && j < 2*a.Ny/3 && k0>Nz/3 && k0 < 2*Nz/3) 
tmp[ind]=a[ind]=k0; 
else tmp[ind]=a[ind]=0.0; 
}

//		CUP FORALL_3D_IN a a tmp tmp # tmp[ind]=m*(a[ind-1]+a[ind+1]+a[ind-Nx]+a[ind+Nx]+a[ind-Nxy]+a[ind+Nxy]); #
template<typename T1, typename T2> __global__ void CUP_3D_2_lapl3d1(int gpuIdx, T1 a, T2 tmp){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t j=blockIdx.y*blockDim.y+threadIdx.y;
const size_t k=blockIdx.z*blockDim.z+threadIdx.z;
if (i>0 && i < a.Nx-1 && j>0 && j < a.Ny-1 && k>0 && k < a.Nz-1){
const size_t ind=k*a.Nx*a.Ny+j*a.Nx+i;
tmp[ind]=m*(a[ind-1]+a[ind+1]+a[ind-Nx]+a[ind+Nx]+a[ind-Nxy]+a[ind+Nxy]); }
}


int main(){
	try{	
	d_Array3d<T> a(Nx, Ny, Nz), tmp(Nx, Ny, Nz);  // Массивы вещественных с распределенем памяти на нескольких GPU
	
    // начальные и граничные условия
{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(a.Nx/threads.x, a.Ny/threads.y, ((a.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3D_2_lapl3d0<<<grid, threads, 0, s[i]>>>(i, split(i, a), split(i, tmp));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamDestroy(s[i]);
}
}






	int NN=3;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	// итерации
	for(int i=0; i<NN; i++){
{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(a.Nx/threads.x, a.Ny/threads.y, ((a.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamCreate(&s[i]);
CUP_3D_2_lapl3d1<<<grid, threads, 0, s[i]>>>(i, split(i, a), split(i, tmp));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
cudaStreamDestroy(s[i]);
}
}
		sync(tmp);
		swap(a, tmp);	
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	elapsedTime=elapsedTime/(( (long long) Nx )*Ny*Nz*NN)*LBE_ncores*LBE_gpuf*1e-3;

	cout << elapsedTime << " GPU ticks\n";

	h_Array3d<T> c(Nx, Ny, Nz);
	cpy(c,a);
	{
		ofstream ofs("1.dat");
		for (int k=0; k<c.nz(); k++){
			for (int i=0; i<c.ny(); i++)
				ofs << c(c.nx()/2,i,k) << '\t';
			ofs << "\n";
		}
	}
	// array3d2paraview(c, "1.mhd");  // paraview 
	}
	catch(MP_Exeption & exept){
		cout << exept.str;
	}
	
	return 0;
}

