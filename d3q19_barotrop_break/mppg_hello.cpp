// –аспад разрыва
#include <fstream>
#include <cmath>

#include <mpp/mpp_window.h>
#include <mpp/draw.h>

#include <mp/h_array2d.h>
#include <mp/lb3d.h>

#include "cuda_core.h"

using namespace mpp;

Scale scale(0, 1024, -2, 2);

int step=0;

void break_(double T, double rhol, double ul, double rhor, double ur, double xt, double & rho, double & u){
	double u_=0.5*(ul+ur)+0.5*sqrt(T)*log(rhol/rhor);
	//double D=0.5*(u_+sqrt(u_*u_+4.0*T));
	if ((u_>ur)&&(u_>ul)){//Shock wave to the right and Rarefaction wave to the left
		double D=0.5*((u_-ur)+sqrt((u_-ur)*(u_-ur)+4.0*T));
		if (xt>=D+ur){
			u=ur;
			rho=rhor;
			return;
		}
		if (xt>=u_-sqrt(T)){
			u=u_;
			rho=sqrt(rhor*rhol*exp((ul-ur)/sqrt(T)));
			return;
		}
		if (xt>=ul-sqrt(T)){
			u=xt+sqrt(T);
			rho=rhol*exp(-(u-ul)/sqrt(T));
			return;
		}
		rho=rhol;
		u=ul;
	
	}
	if ((u_>ur)&&(u_<=ul)){//Shock wave to the right and Shock wave to the left
		double D=0.5*((u_-ur)+sqrt((u_-ur)*(u_-ur)+4.0*T));
		if (xt>=D+ur){
			u=ur;
			rho=rhor;
			return;
		}
		D=0.5*((-u_+ul)+sqrt((-u_+ul)*(-u_+ul)+4.0*T));
		if (xt>ul-D){
			u=u_;
			rho=sqrt(rhor*rhol*exp((ul-ur)/sqrt(T)));
			return;
		}
		rho=rhol;
		u=ul;
	}
	if ((u_<=ur)&&(u_<=ul)){//Rarefaction wave to the rigth and Shock wave to the left
		if (xt>=ur+sqrt(T)){
			u=ur;
			rho=rhor;
			return;
		}
		if (xt>=u_+sqrt(T)){
			u=xt-sqrt(T);
			rho=rhor*exp((u-ur)/sqrt(T));
			return;
		}
		double D=0.5*((-u_+ul)+sqrt((-u_+ul)*(-u_+ul)+4.0*T));
		if (xt>ul-D){
			u=u_;
			rho=sqrt(rhor*rhol*exp((ul-ur)/sqrt(T)));
			return;
		}
		rho=rhol;
		u=ul;
	}
	if ((u_<=ur)&&(u_>ul)){//2 Rarefaction waves
		if (xt>=ur+sqrt(T)){
			u=ur;
			rho=rhor;
			return;
		}
		if (xt>=u_+sqrt(T)){
			u=xt-sqrt(T);
			rho=rhor*exp((u-ur)/sqrt(T));
			return;
		}
		if (xt>=u_-sqrt(T)){
			u=u_;
			rho=sqrt(rhor*rhol*exp((ul-ur)/sqrt(T)));
			return;
		}
		if (xt>=ul-sqrt(T)){
			u=xt+sqrt(T);
			rho=rhol*exp(-(u-ul)/sqrt(T));
			return;
		}
		rho=rhol;
		u=ul;	
	}
}

void f(){
	resize();
	init();
	scale=Scale(0, Nx, -0.6, 1.2);
	step=0;
	while (may_i_repeat()){
		cuda_calc();
		step+=10;
		ask_update();
		if (step>Nx/3) break;
	}
	std::ofstream ofs("rho_u.dat");
	ofs << "# «начени€ параметров течени€ слева и справа от разрыва: rhol=" << rhol << ", uxl= " << ul << ", rhor" << rhor << ", uxr= " << ul << '\n';
	ofs << "# rho ux численные, rho ux аналитические\n";
	for(int i=0; i<rho.Nx; i++){
		double rho_, u_;
		break_(T, rhol, ul, rhor, ur, double(i-rho.Nx/2)/step, rho_, u_);
		ofs << rho(i, rho.Ny/2, rho.Nz/2) << '\t' << ux(i, ux.Ny/2, ux.Nz/2) << '\t' << rho_ << '\t' << u_ << '\n';
	}
}


void ud(){
    set_scale(scale);
    set_color(FL_RED);
    fl_draw("rho", 10,15);
    set_color(FL_YELLOW);
    fl_draw("u", 10,30);
	for(int i=0; i<rho.Nx; i++){
		double rho_, u_;
		break_(T, rhol, ul, rhor, ur, double(i-rho.Nx/2)/step, rho_, u_);
		set_color(FL_WHITE);
		point(i, rho_);
		point(i, u_);
		set_color(FL_RED);
		point(i, rho(i, rho.Ny/2, rho.Nz/2));
		set_color(FL_YELLOW);
		point(i, ux(i, ux.Ny/2, ux.Nz/2));	
	}
}

int main(){
	resize();
	init();
	Mpp_window win;
	
	win.canvas.add_shape(new User_draw(ud));
	win.canvas.scale(scale); 
	
	win.table.add_item("Nx = ", Nx);
	win.table.add_item("Ny = ", Ny);
	win.table.add_item("Nz = ", Nz);
	win.table.add_item("time = ", step);
	win.table.add_item("tau = ", tau);
	win.table.add_item("T = ", T);
	win.table.add_item("rhol = ", rhol);
	win.table.add_item("ul = ", ul);
	win.table.add_item("rhor = ", rhor);
	win.table.add_item("ur = ", ur); 
	win.table.add_item("ticks = ", LBE_elapsedTime);
	set_work(f);
	// start();
	return run();
}
