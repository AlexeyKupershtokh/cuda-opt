// Simple MPPG application
// Need mpp-2.2.0
#include <cmath>
#include <fstream>

#include <mpp/mpp_window.h>
#include <mpp/draw.h>

#include <mp/h_array2d.h>

#include "cuda_core.h"

using namespace mpp;

double mult=100;
h_Array2d<LBE_T> rhoxy1, rhoxz1, rhoyz1, rhoxy2, rhoxz2, rhoyz2;

Scale scalexy(0, 128, 128, 0, 1);
Scale scaleyz(0, 128, 128, 0, 1);
Scale scalexz(0, 128, 128, 0, 1);

int step=0;

void f(){
	while (may_i_repeat()){
		cuda_calc();
		step+=10;
		for(int j=0; j<Ny; j++)
		for(int i=0; i<Nx; i++){
			rhoxy1(i,j)=rho3d1(i, j, Nz/2);
			rhoxy2(i,j)=rho3d2(i, j, Nz/2);
		}

		for(int k=0; k<Nz; k++)
		for(int i=0; i<Nx; i++){
			rhoxz1(i,k)=rho3d1(i, Ny/2, k);
			rhoxz2(i,k)=rho3d2(i, Ny/2, k);
		}

		for(int k=0; k<Nz; k++)
		for(int j=0; j<Ny; j++){
			rhoyz1(j,k)=rho3d1(Nx/2, j, k);
			rhoyz2(j,k)=rho3d2(Nx/2, j, k);
		}
		ask_update();
	}
	array3d2paraview(rho3d1, "rho1.mhd");
	array3d2paraview(rho3d2, "rho2.mhd");
	{
		std::ofstream ofs("ux.dat");
		for (int i=0; i<ux1.ny(); i++)
			ofs << 0.5*(ux1(ux1.nx()/2, i, ux1.nz()/2)+ux2(ux2.nx()/2, i, ux2.nz()/2)) << '\n';
	}
}

void udxy(){
    set_scale(scalexy);
    set_color(FL_BLACK);
    
    for(int j=0; j<Ny; j+=10)
    for(int i=0; i<Nx; i+=10){
		int k=Nz/2;	
		double x=i+(ux1(i,j,k)+ux2(i,j,k))*mult, y=j+(uy1(i,j,k)+uy2(i,j,k))*mult;
		line(i, j, x, y);
		pointb(i, j);
    }
}

void udxz(){
    set_scale(scalexz);
    set_color(FL_BLACK);

    for(int k=0; k<Nz; k+=10)
    for(int i=0; i<Nx; i+=10){
		int j=Ny/2;
		double x=i+(ux1(i,j,k)+ux2(i,j,k))*mult, y=k+(uz1(i,j,k)+uz2(i,j,k))*mult;
		line(i, k, x, y);
		pointb(i, k);
    }
}

void udyz(){
    set_scale(scaleyz);
    set_color(FL_BLACK);

    for(int k=0; k<Nz; k+=10)
    for(int j=0; j<Ny; j+=10){
		int i=Nx/2;
		double x=j+(uy1(i,j,k)+uy2(i,j,k))*mult, y=k+(uz1(i,j,k)+uz2(i,j,k))*mult;
		line(j, k, x, y);
		pointb(j, k);
    }
}


int main(){
	init();
	
	rhoxy1.resize(Nx, Ny);
	rhoxz1.resize(Nx, Nz);
	rhoyz1.resize(Ny, Nz);
	rhoxy2.resize(Nx, Ny);
	rhoxz2.resize(Nx, Nz);
	rhoyz2.resize(Ny, Nz);

	scalexy=Scale(0, Nx, Ny, 0, 1);
	scaleyz=Scale(0, Ny, Nz, 0, 1);
	scalexz=Scale(0, Nx, Nz, 0, 1);

	Canvas canvas_xy("rho xy");
	Canvas canvas_xz("rho xz");
	Canvas canvas_yz("rho yz");
	canvas_xy.add_shape(new BitmapRB<LBE_T>(rhoxy1.p, rhoxy2.p, Nx, Ny, 0.0, 2.0, scalexy)); 
	canvas_xy.add_shape(new User_draw(udxy));
	canvas_xz.add_shape(new BitmapRB<LBE_T>(rhoxz1.p, rhoxz2.p, Nx, Nz, 0.0, 2.0, scalexz));
	canvas_xz.add_shape(new User_draw(udxz));
	canvas_yz.add_shape(new BitmapRB<LBE_T>(rhoyz1.p, rhoyz2.p, Ny, Nz, 0.0, 2.0, scaleyz));
	canvas_yz.add_shape(new User_draw(udyz));
	
	Table table(260, 400, "Multi component");
	table.add_item("Nx = ", Nx, 1);
	table.add_item("Ny = ", Ny, 1);
	table.add_item("Nz = ", Nz, 1);
	table.add_item("step = ", step);
	//table.add_item("G = ", G);
	//table.add_item("T1 = ", T1);
	//table.add_item("T2 = ", T2);
	//table.add_item("gx = ", gx);
	table.add_item("gy = ", gy);
	//table.add_item("ticks = ", LBE_elapsedTime);
	table.add_item("velo_mult = ", mult);
	set_work(f);
	// start();
	Fl::visual(FL_RGB);
	return run();
}

