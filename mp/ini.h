#ifndef INI_H__
#define INI_H__

#include<map>
#include<string>
#include<sstream>
#include<fstream>

#include <mp/d_array3d.h>

//			CUP FORALL_3D_IJK a array v value nx1 nx1*array.Nx ny1 ny1*array.Ny nz1 nz1*array.Nz nx2 nx2*array.Nx ny2 ny2*array.Ny nz2 nz2*array.Nz  #
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __global__ void CUP_3D_8_0(int gpuIdx, T1 a, T2 v, T3 nx1, T4 ny1, T5 nz1, T6 nx2, T7 ny2, T8 nz2){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t j=blockIdx.y*blockDim.y+threadIdx.y;
const size_t k=blockIdx.z*blockDim.z+threadIdx.z;
const size_t ind=k*a.Nx*a.Ny+j*a.Nx+i;

if( (i>=nx1 && i<nx2) && (j>=ny1 && j<ny2) && (k+(a.Nz-2)*gpuIdx>=nz1 && k+(a.Nz-2)*gpuIdx<nz2) ) 
a[ind]=v; 
}

//			CUP FORALL_3D_IJK array array value value nx0 nx0*array.Nx ny0 ny0*array.Ny nz0 nz0*array.Nz rx rx*array.Nx ry ry*array.Ny rz rz*array.Nz #
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __global__ void CUP_3D_8_1(int gpuIdx, T1 array, T2 value, T3 nx0, T4 ny0, T5 nz0, T6 rx, T7 ry, T8 rz){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t j=blockIdx.y*blockDim.y+threadIdx.y;
const size_t k=blockIdx.z*blockDim.z+threadIdx.z;
const size_t ind=k*array.Nx*array.Ny+j*array.Nx+i;

if( (i>=nx0-rx && i<nx0+rx) && (j>=ny0-ry && j<ny0+ry) && (k+(array.Nz-2)*gpuIdx>=nz0-rz && k+(array.Nz-2)*gpuIdx<nz0+rz) ) 
array[ind]=value; 
}

//			CUP FORALL_3D_IJK array array value value R0 R0*array.Nx nx0 nx0*array.Nx ny0 ny0*array.Ny nz0 nz0*array.Nz  #
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __global__ void CUP_3D_6_2(int gpuIdx, T1 array, T2 value, T3 R0, T4 nx0, T5 ny0, T6 nz0){const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
const size_t j=blockIdx.y*blockDim.y+threadIdx.y;
const size_t k=blockIdx.z*blockDim.z+threadIdx.z;
const size_t ind=k*array.Nx*array.Ny+j*array.Nx+i;

int k1=k+(array.Nz-2)*gpuIdx; 
int R=sqrt( (nx0-i)*(nx0-i)+(ny0-j)*(ny0-j)+(nz0-k1)*(nz0-k1) ); 
if( R < R0 ) array[ind]=value; 
}


template<class T> bool ini_read(const char * fname, T & value, const char * label, const T & default_=T() ){
	std::map<std::string, std::string> m;
	std::ifstream ifs(fname);
	std::string label_, value_;
	while(ifs >> label_ && getline(ifs, value_) ) 
		m.insert(std::make_pair(label_, value_));
			
	std::map<std::string, std::string>::iterator p=m.find(label);
	if ( p!=m.end() ){
		std::istringstream os(m[label]);
		os >> value;
		return true;
	}
	value=default_;
	return false;
}

template<class T> T ini_read(const char * fname, const char * label, const T & default_=T()){
	T tmp;
	ini_read(fname, tmp, label, default_);
	return tmp;
}

bool ini_read(const char * fname, d_Array3d<LBE_T> & array, const char * label){
	std::multimap<std::string, std::string> m;
	std::ifstream ifs(fname);
	std::string label_, value_;
	while(ifs >> label_ && getline(ifs, value_) ){
		m.insert(std::make_pair(label_, value_));
	}
	std::multimap<std::string, std::string>::iterator lower, upper;
	lower=m.lower_bound(label);
	upper=m.upper_bound(label);
	if (lower==m.end()) return false;
	
	for( ; lower!=upper; lower++){	
		std::istringstream is(lower->second);
		is >> label_;
		if (label_ == "rect_corner"){
			double nx1, ny1, nz1, nx2, ny2, nz2;
			LBE_T value;
			is >> nx1 >> ny1 >> nz1 >> nx2 >> ny2 >> nz2 >> value;
{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(array.Nx/threads.x, array.Ny/threads.y, ((array.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
s[i] = 0;
CUP_3D_8_0<<<grid, threads, 0, s[i]>>>(i, split(i, array), split(i, value), split(i, nx1*array.Nx), split(i, ny1*array.Ny), split(i, nz1*array.Nz), split(i, nx2*array.Nx), split(i, ny2*array.Ny), split(i, nz2*array.Nz));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);

}
}



		}
		if (label_ == "rect"){
			double nx0, ny0, nz0, rx, ry, rz;
			LBE_T value;
			is >> nx0 >> ny0 >> nz0 >> rx >> ry >> rz >> value;
{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(array.Nx/threads.x, array.Ny/threads.y, ((array.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
s[i] = 0;
CUP_3D_8_1<<<grid, threads, 0, s[i]>>>(i, split(i, array), split(i, value), split(i, nx0*array.Nx), split(i, ny0*array.Ny), split(i, nz0*array.Nz), split(i, rx*array.Nx), split(i, ry*array.Ny), split(i, rz*array.Nz));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);

}
}



		}
		if (label_ == "sphere"){
			double nx0, ny0, nz0, R0;
			LBE_T value;
			is >> nx0 >> ny0 >> nz0 >> R0 >> value;
{
dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);
dim3 grid(array.Nx/threads.x, array.Ny/threads.y, ((array.Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);
cudaStream_t s[4];
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);
s[i] = 0;
CUP_3D_6_2<<<grid, threads, 0, s[i]>>>(i, split(i, array), split(i, value), split(i, R0*array.Nx), split(i, nx0*array.Nx), split(i, ny0*array.Ny), split(i, nz0*array.Nz));
}
for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){
cudaSetDevice(MP_parallel.GPU_pool[i]);

}
}




		}
	}
	return true;
}

bool ini_read(const char * fname, MP_Parallel & p, const char * label, const MP_Parallel & default_=MP_Parallel() ){
	std::map<std::string, std::string> m;
	std::ifstream ifs(fname);
	std::string label_, value_;
	while(ifs >> label_ && getline(ifs, value_) ) 
		m.insert(std::make_pair(label_, value_));
			
	std::map<std::string, std::string>::iterator pp=m.find(label);
	if ( pp!=m.end() ){
		std::istringstream os(pp->second);
		p.GPU_pool.resize(0);
		int gpu;
		while ( os >> gpu ) p.GPU_pool.push_back(gpu);
		return true;
	}
	p=default_;
	return false;
}

#endif 

