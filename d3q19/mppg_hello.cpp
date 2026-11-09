// Simple LBE
#include <cmath>
#include <fstream>
#include <iostream>

#include <mpp/mpp_window.h>
#include <mpp/draw.h>

#include <mp/mp.h>
#include <mp/h_array2d.h>
#include <mp/lb3d.h>

#include "cuda_core.h"

using namespace mpp;

double mult=100;

h_Array2d<LBE_T> rhoxy, rhoxz, rhoyz;

Scale scale(0, 128, 128, 0, 1);
Scale scalexy(0, 128, 128, 0, 1);
Scale scaleyz(0, 128, 128, 0, 1);
Scale scalexz(0, 128, 128, 0, 1);

int step=0;

void f(){
	while (may_i_repeat()){
		cuda_calc();
		step+=30;
		ask_update();
		std::ofstream ofs("rho.dat");

		for(int j=0; j<Ny; j++)
		for(int i=0; i<Nx; i++)
			rhoxy(i,j)=rho(i, j, Nz/2);

		for(int k=0; k<Nz; k++)
		for(int i=0; i<Nx; i++)
			rhoxz(i,k)=rho(i, Ny/2, k);

		for(int k=0; k<Nz; k++)
		for(int j=0; j<Ny; j++)
			rhoyz(j,k)=rho(Nx/2, j, k);
	}
	//array3d2paraview(rho, "rho.mhd");
}

void udxy(){
    set_scale(scalexy);
    set_color(FL_RED);

    for(int j=0; j<Ny; j+=10)
    for(int i=0; i<Nx; i+=10){
		int k=Nz/2;	
		double x=i+ux(i,j,k)*mult, y=j+uy(i,j,k)*mult;
		line(i, j, x, y);
		pointb(i, j);
    }
}

void udxz(){
    set_scale(scalexz);
    set_color(FL_RED);

    for(int k=0; k<Nz; k+=10)
    for(int i=0; i<Nx; i+=10){
		int j=Ny/2;
		double x=i+ux(i,j,k)*mult, y=k+uz(i,j,k)*mult;
		line(i, k, x, y);
		pointb(i, k);
    }
}

void udyz(){
    set_scale(scaleyz);
    set_color(FL_RED);

    for(int k=0; k<Nz; k+=10)
    for(int j=0; j<Ny; j+=10){
		int i=Nx/2;
		double x=j+uy(i,j,k)*mult, y=k+uz(i,j,k)*mult;
		line(j, k, x, y);
		pointb(j, k);
    }
}

int main(int argc, char* argv[]){
	try{
		init();		
	}
	catch(MP_Exeption exept){
		std::cout << exept.str;
	}
	
	rhoxy.resize(Nx, Ny);
	rhoxz.resize(Nx, Nz);
	rhoyz.resize(Ny, Nz);

	scalexy=Scale(0, Nx, Ny, 0, 1);
	scaleyz=Scale(0, Ny, Nz, 0, 1);
	scalexz=Scale(0, Nx, Nz, 0, 1);

	Canvas canvas_xy("rho xy");
	Canvas canvas_xz("rho xz");
	Canvas canvas_yz("rho yz");
	canvas_xy.add_shape(new Bitmap(rhoxy.p, Nx, Ny, 0.5, 1.5, scalexy));
	canvas_xy.add_shape(new User_draw(udxy));
	canvas_xz.add_shape(new Bitmap(rhoxz.p, Nx, Nz, 0.5, 1.5, scalexz));
	canvas_xz.add_shape(new User_draw(udxz));
	canvas_yz.add_shape(new Bitmap(rhoyz.p, Ny, Nz, 0.5, 1.5, scaleyz));
	canvas_yz.add_shape(new User_draw(udyz));

	Table table("Simple LBE");

	table.add_item("Nx = ", Nx);
	table.add_item("Ny = ", Ny);
	table.add_item("Nz = ", Nz);
	table.add_item("step = ", step);  
	// table.add_item("ticks = ", LBE_elapsedTime);
	table.add_item("tau = ", tau);
	set_work(f);
	ask_update();
	return run();
}
