#ifndef CUDA_ARRAT1D_H_

template<typename T> class d_Array1d{
public:
	T*p;
private:
	int Nx;
	bool clone; 
public:
	d_Array1d():Nx(0){
		clone=false;
		p=0;
	}	
	d_Array1d(size_t Nx):Nx(Nx){
		cudaMalloc(&p, Nx*sizeof(T));
		clone=false;
	}
	~d_Array1d(){
		if (!clone) cudaFree(p);
	}
	inline __host__ __device__ int nx()const{return Nx;}
	inline __device__       T & operator[](int ind)     {return p[ind];}
	inline __device__ const T & operator[](int ind)const{return p[ind];}	 

	template<typename TT> friend d_Array1d<TT> clone(d_Array1d<TT> & a);
};

template<typename TT> d_Array1d<TT> clone(d_Array1d<TT> & a){
	d_Array1d<TT> tmp;
	tmp.p=a.p;
	tmp.Nx=a.Nx;
	tmp.clone=true;
	return tmp;	
}

#endif

