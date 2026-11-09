#ifndef CUDA_CORE_H_
#define CUDA_CORE_H_

#include <mp/mp.h>
#include <mp/lb3d.h>
#include <mp/h_array3d.h>

extern "C" MP_EXPORT int Nx;
extern "C" MP_EXPORT int Ny;
extern "C" MP_EXPORT int Nz;

extern "C" MP_EXPORT float LBE_elapsedTime;

extern "C" MP_EXPORT h_Array3d<LBE_T> rho, rho3d1, rho3d2, ux, uy, uz;


extern "C" MP_EXPORT void init();
extern "C" MP_EXPORT void cuda_calc();

#endif
