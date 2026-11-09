#ifndef MP__H__
#define MP__H__

#include <vector>
#include <string>

#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

// SUPRA
 #define LBE_ncores 480
 #define LBE_gpuf 1140*1e6

// SKN
// #define LBE_ncores 512
// #define LBE_gpuf 1566*1e6

// nusc
// #define LBE_ncores 512
// #define LBE_gpuf 1301*1e6

template<typename T> T split(size_t , const T & a){ return a; }

// for dll export/import
#ifdef WIN32
	#ifdef MP_LIBRARY
		#define MP_EXPORT __declspec(dllexport)
	#else
		#define MP_EXPORT __declspec(dllimport)
	#endif /* MP_LIBRARY */
#else
	#define MP_EXPORT
#endif /* WIN32 */

class MP_Exeption{
public:
	std::string str;
	MP_Exeption(const std::string & str):str(str){}
};

class MP_Parallel{
public:
	std::vector<size_t> GPU_pool;    // Массив с номерами используемых GPU
	size_t Nx, Ny, Nz, Nz_bound;     // Количество потоков в блоке
	MP_Parallel():GPU_pool(1, 0), Nx(16), Ny(16), Nz(1), Nz_bound(1){
		int deviceCount = 0;
    	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
		GPU_pool.resize(deviceCount);
		for(size_t i=0; i<GPU_pool.size(); i++)
			GPU_pool[i]=i;		
	}	
	MP_Parallel(int NGPU):GPU_pool(NGPU, 0), Nx(16), Ny(16), Nz(1), Nz_bound(1){
		for(size_t i=0; i<GPU_pool.size(); i++)
			GPU_pool[i]=i;
	}
	MP_Parallel(const std::vector<size_t> & GPU_pool):GPU_pool(GPU_pool), Nx(16), Ny(16), Nz(1), Nz_bound(1){}

	MP_Parallel(int *a, int N):GPU_pool(N), Nx(16), Ny(16), Nz(1), Nz_bound(1){
		for(size_t i=0; i<GPU_pool.size(); i++)
			GPU_pool[i]=a[i];
	}
};

extern "C" MP_EXPORT const MP_Parallel MP_parallel; // Параметра распараллеливания


#endif /* MP__H__ */

