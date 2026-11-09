#include <sstream>
#include "cuda_core.h"

//#include <cuda.h>
//#include <cuda_runtime_api.h>
//#include <cutil_inline.h>

using namespace std;
string str_info;

void gpu_info(){
	ostringstream ss;	
	int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    ss << "Found "<< deviceCount <<" CUDA Capable device(s)\n";
	for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
		ss << "Device " << dev << ": \"" << deviceProp.name << "\"\n";
		ss << "  Total amount of global memory:                    "<< (float)deviceProp.totalGlobalMem/1048576.0f << " MBytes (" << deviceProp.totalGlobalMem << " bytes)\n";
		ss << "  (" <<deviceProp.multiProcessorCount <<") Multiprocessors x (" << 32 <<") CUDA Cores/MP: " << 32 * deviceProp.multiProcessorCount << " CUDA Cores\n"; 
		ss << "  GPU Clock Speed:                                              " << deviceProp.clockRate * 1e-6f << " GHz\n"; 
	}

	str_info.resize(0);
	str_info+=ss.str();
}

