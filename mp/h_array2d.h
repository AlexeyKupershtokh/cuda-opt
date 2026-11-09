#ifndef H_CUDA_ARRAY2D_H_
#define H_CUDA_ARRAY2D_H_

template<typename T> class h_Array2d{
public:
	T*p;
private:
	int Nx, Ny;
public:
	h_Array2d():Nx(0), Ny(0){
		p=0;
	}	
	h_Array2d(size_t Nx, size_t Ny):Nx(Nx), Ny(Ny){
		p=new T[Nx*Ny];
	}
	~h_Array2d(){
		if (p!=0) delete[] p;
	}
	void resize(size_t Nx_, size_t Ny_){
		Nx=Nx_; Ny=Ny_;
		if (p!=0) delete[] p; 
		p=new T[Nx*Ny];
	}
	inline int nx()const{return Nx;}
	inline int ny()const{return Ny;}
	inline T & operator[](int ind)     {return p[ind];}
	inline const T & operator[](int ind)const{return p[ind];}

	inline       T & operator()(int j, int i)     {return p[i*Nx+j];}
	inline const T & operator()(int j, int i)const{return p[i*Nx+j];}


	template<typename TT> friend h_Array2d<TT> clone(h_Array2d<TT> & a);
};

#endif

