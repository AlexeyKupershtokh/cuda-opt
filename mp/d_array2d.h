#ifndef D_CUDA_ARRAY2D_H_
#define D_CUDA_ARRAY2D_H_

#include <mp/h_array2d.h>

template<typename T> class d_Array2d{
public:
	T*p;
public:
	int Nx, Ny;
	bool clone; 
public:
	d_Array2d():Nx(0), Ny(0){
		clone=false;
		p=0;
	}	
	d_Array2d(size_t Nx, size_t Ny):Nx(Nx), Ny(Ny){
		cudaMalloc((void **)&p, Nx*Ny*sizeof(T));
		clone=false;
	}
	~d_Array2d(){
		//if (!clone) cudaFree(p);
	}
	void resize(int Nx_, int Ny_){
		if ( p!=0 ) delete[] p;
		Nx=Nx_; Ny=Ny_;
		cudaMalloc(&p, Nx*Ny*sizeof(T));
		clone=false;
	}
	inline __host__ __device__ int nx()const{return Nx;}
	inline __host__ __device__ int ny()const{return Ny;}
	inline __device__       T & operator[](int ind)     {return p[ind];}
	inline __device__ const T & operator[](int ind)const{return p[ind];}	 
	inline __device__       T & operator()(int ix, int iy)     {return p[iy*Nx+ix];}
	inline __device__ const T & operator()(int ix, int iy)const{return p[iy*Nx+ix];}	 


	template<typename TT> friend d_Array2d<TT> clone(d_Array2d<TT> & a);
};

template<typename T> void swap(d_Array2d<T> &a, d_Array2d<T> &b){
	T *p=a.p;
	a.p=b.p;
	b.p=p;
}

template<typename T> void cpy(h_Array2d<T> &a, d_Array2d<T> &b){  // a=b
	cudaMemcpy(a.p, b.p, b.nx()*b.ny()*sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T> void cpy(d_Array2d<T> &a, h_Array2d<T> &b){  // a=b
	cudaMemcpy(a.p, b.p, b.nx()*b.ny()*sizeof(T), cudaMemcpyHostToDevice);
}

#endif
