#ifndef H_CUDA_ARRAY3D_H_
#define H_CUDA_ARRAY3D_H_

#include <string>
#include <stdio.h>

template<typename T> class h_Array3d{
public:
	T*p;
	int Nx, Ny, Nz;
public:
	h_Array3d():Nx(0), Ny(0), Nz(0){
		p=0;
	}	
	h_Array3d(size_t Nx, size_t Ny, size_t Nz):Nx(Nx), Ny(Ny), Nz(Nz){
		p=new T[Nx*Ny*Nz];
	}
	~h_Array3d(){
		if (p) delete[] p;
	}
	void resize(size_t Nx_, size_t Ny_, size_t Nz_){
		Nx=Nx_; Ny=Ny_; Nz=Nz_;
		if (p) delete[] p;
		p=new T[Nx*Ny*Nz];
	}
	inline int nx()const{return Nx;}
	inline int ny()const{return Ny;}
	inline int nz()const{return Nz;}
	inline T & operator[](int ind)     {return p[ind];}
	inline const T & operator[](int ind)const{return p[ind];}	 

	inline       T & operator()(int j, int i, int k)     {return p[k*Nx*Ny+i*Nx+j];}
	inline const T & operator()(int j, int i, int k)const{return p[k*Nx*Ny+i*Nx+j];}	 

	//template<typename TT> friend h_Array2d<TT> clone(h_Array2d<TT> & a);
};

void array3d2paraview(const h_Array3d<double> & a, const std::string & fname){
    FILE * f=fopen(fname.c_str(), "w");
    fprintf(f, "NDims = 3\nDimSize = %d %d %d\n", a.nx(), a.ny(), a.nz());
    fputs("ElementSize = 1.0 1.0 1.0\nElementSpacing = 1.0 1.0 1.0\nElementType = MET_DOUBLE\nElementByteOrderMSB = False\nElementDataFile = local\n", f);
    fwrite(a.p, sizeof(double), a.nx()*a.ny()*a.nz(), f);
    fclose(f);
}

void array3d2paraview(const h_Array3d<float> & a, const std::string & fname){
    FILE * f=fopen(fname.c_str(), "w");
    fprintf(f, "NDims = 3\nDimSize = %d %d %d\n", a.nx(), a.ny(), a.nz());
    fputs("ElementSize = 1.0 1.0 1.0\nElementSpacing = 1.0 1.0 1.0\nElementType = MET_FLOAT\nElementByteOrderMSB = False\nElementDataFile = local\n", f);
    fwrite(a.p, sizeof(float), a.nx()*a.ny()*a.nz(), f);
    fclose(f);
}

#endif

