// Simple MPPG application
// Need mpp-2.2.0
#include <cmath>
#include <fstream>

#include <mpp/mpp_window.h>
#include <mpp/draw.h>

#include <mp/h_array2d.h>

#include "cuda_core.h"

using namespace mpp;

h_Array2d<double> rho2d(128, 128), ux2d(128, 128), uy2d(128, 128), uz2d(128, 128);
h_Array2d<double> rhoyx(128, 128), rhozx(128, 128), rhozy(128, 128);

h_Array2d<double> rho1(128,128), rho2(128,128);

Scale scale(0, 128, 128, 0, 1);
Scale scaleyx(0, 128, 128, 0, 1);
Scale scalezy(0, 128, 128, 0, 1);
Scale scalezx(0, 128, 128, 0, 1);


int step=0;

void f(){
	while (may_i_repeat()){
		cuda_calc();
		step++;
		ask_update();
		std::ofstream ofs("rho.dat");

		for(int j=0; j<Ny; j++)
		for(int i=0; i<Nx; i++){
			rhoyx(i,j)=rho(i, j, Nz/2);
			rho1(i,j)=rho3d1(i, j, Nz/2);
			rho2(i,j)=rho3d2(i, j, Nz/2);
		}

		for(int k=0; k<Nz; k++)
		for(int i=0; i<Nx; i++)
			rhozx(i,k)=rho(i, Ny/2, k);

		for(int k=0; k<Nz; k++)
		for(int j=0; j<Ny; j++)
			rhozy(j,k)=rho(Nx/2, j, k);

		for(int k=0; k<Nz; k++)
		for(int j=0; j<Ny; j++){
			int i=Nx/2;
			rho2d(j,k)=rho(i,j,k);
			ux2d(j,k)=ux(i,j,k);
			uy2d(j,k)=uy(i,j,k);
			uz2d(j,k)=uz(i,j,k);
		}
		
		/*
		for(int k=0; k<Nz; k++){
		    for(int j=0; j<Ny; j++){
			for(int i=0; i<Nx; i++)
			    ofs<<rho(i,j,k)<<'\t';
			ofs << '\n';
		    }
		    ofs<<'\n';
		}
		*/
	}
	array3d2paraview(rho, "rho.mhd");
}

void ud(){
    set_scale(scale);
    set_color(FL_RED);

    double mult=75;
    for(int i=0; i<Nx; i+=10)
    for(int k=0; k<Nz; k+=10){
	int j=10;	
	double x=i+ux(i,j,k)*mult, z=k+uz(i,j,k)*mult;
	line(i,k, x, z);
	pointb(i,k);
	rho2d(i,k)=rho(i,j,k);
	ux2d(i,k)=ux(i,j,k);
	uy2d(i,k)=uy(i,j,k);
	uz2d(i,k)=uz(i,j,k);
    }

}

int main(){
	init();
	rho1.resize(Nx, Ny);
	rho2.resize(Nx, Ny);
	
	rho2d.resize(Nx, Nz);
	ux2d.resize(Nx, Nz);
	uy2d.resize(Nx, Nz);
	uz2d.resize(Nx, Nz);
	rhoyx.resize(Nx, Ny);
	rhozx.resize(Nx, Nz);
	rhozy.resize(Ny, Nz);

	scale=Scale(0, Nx, Nz, 0, 1);
	scaleyx=Scale(0, Nx, Ny, 0, 1);
	scalezy=Scale(0, Ny, Nz, 0, 1);
	scalezx=Scale(0, Nx, Nz, 0, 1);

	Canvas canvas_yx("yx");
	Canvas canvas_zx("zx");
	Canvas canvas_zy("zy");
	canvas_yx.add_shape(new BitmapRB<LBE_T>(rho1.p, rho2.p, Nx, Ny, 0.0, 1.5, scaleyx));
	canvas_zx.add_shape(new Bitmap(rhozx.p, Nx, Nz, 0.0, 1.0, scalezx));
	canvas_zy.add_shape(new Bitmap(rhozy.p, Ny, Nz, 0.0, 1.0, scalezy));

	Mpp_window win;
	Canvas canvas_ux("ux");
	Canvas canvas_uy("uz");
	
	canvas_ux.add_shape(new Bitmap(ux2d.p, Nx, Nz, -0.2, +0.2, scale));
	canvas_uy.add_shape(new Bitmap(uz2d.p, Nx, Nz, -0.2, +0.2, scale));
	canvas_ux.add_shape(new User_draw(ud));	
	canvas_uy.add_shape(new User_draw(ud));
	win.canvas.add_shape(new Bitmap(rho2d.p, Nx, Nz, 0.8, 1.2, scale));
	
	win.table.add_item("Nx = ", Nx);
	win.table.add_item("Ny = ", Ny);
	win.table.add_item("Nz = ", Nz);
	win.table.add_item("step = ", step);  
	win.table.add_item("ticks = ", LBE_elapsedTime);
	set_work(f);
	// start();
	Fl::visual(FL_RGB);
	return run();
}
