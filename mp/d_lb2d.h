#ifndef LBE2D_LB_H_
#define LBE2D_LB_H_

#include <mp/d_array2d.h>
#include <mp/d_lbe2d.h>

//	CUP FORALL_2D  A  lb.rho rho rho # A[ind]=rho;
//	CUP FORALL_2D  A  lb.ux  ux ux   # A[ind]=ux;
//	CUP FORALL_2D  A  lb.uy  uy uy   # A[ind]=uy;
//	CUP FORALL_2D  A  lb.uu uu uu    # A[ind]=uu;
//	CUP FORALL_2D lb lb # lb.f0[ind]=LBE_f0(lb.rho[ind], lb.ux[ind], lb.uy[ind], lb.uu[ind]);
//	CUP FORALL_2D lb lb # lb.f1[ind]=LBE_f1(lb.rho[ind], lb.ux[ind], lb.uy[ind], lb.uu[ind]);
//	CUP FORALL_2D f lb.f2 rho lb.rho ux lb.ux uy lb.uy uu lb.uu # f[ind]=LBE_f2(rho[ind], ux[ind], uy[ind], uu[ind]);
//	CUP FORALL_2D f lb.f3 rho lb.rho ux lb.ux uy lb.uy uu lb.uu # f[ind]=LBE_f3(rho[ind], ux[ind], uy[ind], uu[ind]);
//	CUP FORALL_2D f lb.f4 rho lb.rho ux lb.ux uy lb.uy uu lb.uu # f[ind]=LBE_f4(rho[ind], ux[ind], uy[ind], uu[ind]);
//	CUP FORALL_2D f lb.f5 rho lb.rho ux lb.ux uy lb.uy uu lb.uu # f[ind]=LBE_f5(rho[ind], ux[ind], uy[ind], uu[ind]);
//	CUP FORALL_2D f lb.f6 rho lb.rho ux lb.ux uy lb.uy uu lb.uu # f[ind]=LBE_f6(rho[ind], ux[ind], uy[ind], uu[ind]);
//	CUP FORALL_2D f lb.f7 rho lb.rho ux lb.ux uy lb.uy uu lb.uu # f[ind]=LBE_f7(rho[ind], ux[ind], uy[ind], uu[ind]);
//	CUP FORALL_2D f lb.f8 rho lb.rho ux lb.ux uy lb.uy uu lb.uu # f[ind]=LBE_f8(rho[ind], ux[ind], uy[ind], uu[ind]);
//	CUP FORALL_2D_IN lb lb # lb.feq[ind]=lb.f1[ind-1];
//	CUP FORALL_2D_IN lb lb # lb.f1[ind]=lb.feq[ind];
//	CUP FORALL_2D_IN lb lb # lb.feq[ind]=lb.f2[ind+lb.Nx];
//	CUP FORALL_2D_IN lb lb # lb.f2[ind]=lb.feq[ind];
//	CUP FORALL_2D_IN lb lb # lb.feq[ind]=lb.f3[ind+1];
//	CUP FORALL_2D_IN lb lb # lb.f3[ind]=lb.feq[ind];
//	CUP FORALL_2D_IN lb lb # lb.feq[ind]=lb.f4[ind-lb.Nx];
//	CUP FORALL_2D_IN lb lb # lb.f4[ind]=lb.feq[ind];
//	CUP FORALL_2D_IN lb lb # lb.feq[ind]=lb.f5[ind+lb.Nx-1];
//	CUP FORALL_2D_IN lb lb # lb.f5[ind]=lb.feq[ind];
//	CUP FORALL_2D_IN lb lb # lb.feq[ind]=lb.f6[ind+lb.Nx+1];
//	CUP FORALL_2D_IN lb lb # lb.f6[ind]=lb.feq[ind];
//	CUP FORALL_2D_IN lb lb # lb.feq[ind]=lb.f7[ind-lb.Nx+1];
//	CUP FORALL_2D_IN lb lb # lb.f7[ind]=lb.feq[ind];
//	CUP FORALL_2D_IN lb lb # lb.feq[ind]=lb.f8[ind-lb.Nx-1];
//	CUP FORALL_2D_IN lb lb # lb.f8[ind]=lb.feq[ind];
//	CUP FORALL_2D lb lb # lb.rho[ind]=lb.f0[ind]+lb.f1[ind]+lb.f2[ind]+lb.f3[ind]+lb.f4[ind]+lb.f5[ind]+lb.f6[ind]+lb.f7[ind]+lb.f8[ind];
//	CUP FORALL_2D rhoux lb.ux rho lb.rho f1 lb.f1 f3 lb.f3 f5 lb.f5 f6 lb.f6 f7 lb.f7 f8 lb.f8 # rhoux[ind] =(f1[ind]-f3[ind]+f5[ind]-f6[ind]-f7[ind]+f8[ind])/rho[ind];
//CUP FORALL_2D rhouy lb.uy rho lb.rho f2 lb.f2 f4 lb.f4 f5 lb.f5 f6 lb.f6 f7 lb.f7 f8 lb.f8 # rhouy[ind] =(f2[ind]-f4[ind]+f5[ind]+f6[ind]-f7[ind]-f8[ind])/rho[ind];
//	CUP FORALL_2D uu lb.uu ux lb.ux uy lb.uy # uu[ind]=ux[ind]*ux[ind]+uy[ind]*uy[ind];
//	CUP FORALL_2D lb lb taurev taurev # const LBE_T duu=1.0-LBE_d*lb.uu[ind]; lb.f0[ind]+=(LBE_w0*lb.rho[ind]*(duu) - lb.f0[ind])*taurev; lb.f1[ind]+=(LBE_w1*lb.rho[ind]*(duu + LBE_a*lb.ux[ind] + LBE_b*lb.ux[ind]*lb.ux[ind])-lb.f1[ind])*taurev; lb.f2[ind]+=(LBE_w1*lb.rho[ind]*(duu + LBE_a*lb.uy[ind] + LBE_b*lb.uy[ind]*lb.uy[ind])-lb.f2[ind])*taurev; lb.f3[ind]+=(LBE_w1*lb.rho[ind]*(duu - LBE_a*lb.ux[ind] + LBE_b*lb.ux[ind]*lb.ux[ind])-lb.f3[ind])*taurev; lb.f4[ind]+=(LBE_w1*lb.rho[ind]*(duu - LBE_a*lb.uy[ind] + LBE_b*lb.uy[ind]*lb.uy[ind])-lb.f4[ind])*taurev; const LBE_T uxuy=lb.ux[ind]+lb.uy[ind];	const LBE_T ux_uy=lb.ux[ind]-lb.uy[ind]; lb.f5[ind]+=(LBE_w2*lb.rho[ind]*(duu + LBE_a*uxuy  + LBE_b*uxuy *uxuy )-lb.f5[ind])*taurev; lb.f6[ind]+=(LBE_w2*lb.rho[ind]*(duu - LBE_a*ux_uy + LBE_b*ux_uy*ux_uy)-lb.f6[ind])*taurev; lb.f7[ind]+=(LBE_w2*lb.rho[ind]*(duu - LBE_a*uxuy  + LBE_b*uxuy*uxuy)  -lb.f7[ind])*taurev; lb.f8[ind]+=(LBE_w2*lb.rho[ind]*(duu + LBE_a*ux_uy + LBE_b*ux_uy*ux_uy)-lb.f8[ind])*taurev;

class d_LB2d{
public:	
	int Nx, Ny;
	d_Array2d<LBE_T> rho, ux, uy, uu;
	d_Array2d<LBE_T> f0, f1, f2, f3, f4, f5, f6, f7, f8;
	d_Array2d<LBE_T> feq;
	d_LB2d(int Nx=0, int Ny=0): Nx(Nx), Ny(Ny), rho(Nx, Ny), ux(Nx, Ny), uy(Nx, Ny), uu(Nx, Ny), f0(Nx, Ny), f1(Nx, Ny), f2(Nx, Ny), f3(Nx, Ny), f4(Nx, Ny), f5(Nx, Ny), f6(Nx, Ny), f7(Nx, Ny), f8(Nx, Ny), feq(Nx, Ny){}
	inline __host__ __device__ int nx()const{return Nx;}
	inline __host__ __device__ int ny()const{return Ny;}
};

void init(d_LB2d & lb, LBE_T rho=1.0, LBE_T ux=0.0, LBE_T uy=0.0){
	const LBE_T uu=ux*ux+uy*uy;
}

void rhou2f(d_LB2d & lb){
	// CUP FORALL_2D f lb.f[0] rho lb.rho ux lb.ux uy lb.uy uu lb.uu # f[ind]=LBE_f0(rho[ind], ux[ind], uy[ind], uu[ind]);
	
	// CUP FORALL_2D f lb.f[1] rho lb.rho ux lb.ux uy lb.uy uu lb.uu # f[ind]=LBE_f1(rho[ind], ux[ind], uy[ind], uu[ind]);

}

void transfer(d_LB2d &lb){







}

void f2rhou(d_LB2d & lb){

// CUP FORALL_2D lb lb # lb.ux[ind] =(lb.f1[ind]-lb.f3[ind]+lb.f5[ind]-lb.f6[ind]-lb.f7[ind]+lb.f8[ind])/lb.rho[ind];


}


void relax(d_LB2d &lb, LBE_T tau){
	const LBE_T taurev=1.0/tau;

}

#endif

