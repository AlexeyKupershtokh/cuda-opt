#ifndef D_LBE2D__H__
#define D_LBE2D__H__

#include <mp/lbe2d.h>

__device__ __host__ __forceinline__ LBE_T LBE_f0(LBE_T rho, LBE_T ux, LBE_T uy, LBE_T uu){ return LBE_w0*rho*(1 - LBE_d*uu);}
__device__ __host__ __forceinline__ LBE_T LBE_f1(LBE_T rho, LBE_T ux, LBE_T uy, LBE_T uu){ return LBE_w1*rho*(1 - LBE_d*uu + LBE_a*ux + LBE_b*ux*ux ); }
__device__ __host__ __forceinline__ LBE_T LBE_f2(LBE_T rho, LBE_T ux, LBE_T uy, LBE_T uu){ return LBE_w1*rho*(1 - LBE_d*uu + LBE_a*uy + LBE_b*uy*uy ); }
__device__ __host__ __forceinline__ LBE_T LBE_f3(LBE_T rho, LBE_T ux, LBE_T uy, LBE_T uu){ return LBE_w1*rho*(1 - LBE_d*uu - LBE_a*ux + LBE_b*ux*ux ); }
__device__ __host__ __forceinline__ LBE_T LBE_f4(LBE_T rho, LBE_T ux, LBE_T uy, LBE_T uu){ return LBE_w1*rho*(1 - LBE_d*uu - LBE_a*uy + LBE_b*uy*uy ); }
__device__ __host__ __forceinline__ LBE_T LBE_f5(LBE_T rho, LBE_T ux, LBE_T uy, LBE_T uu){ return LBE_w2*rho*(1 - LBE_d*uu + LBE_a*(ux+uy) + LBE_b*(ux+uy)*(ux+uy) ); }
__device__ __host__ __forceinline__ LBE_T LBE_f6(LBE_T rho, LBE_T ux, LBE_T uy, LBE_T uu){ return LBE_w2*rho*(1 - LBE_d*uu + LBE_a*(uy-ux) + LBE_b*(uy-ux)*(uy-ux) ); }
__device__ __host__ __forceinline__ LBE_T LBE_f7(LBE_T rho, LBE_T ux, LBE_T uy, LBE_T uu){ return LBE_w2*rho*(1 - LBE_d*uu - LBE_a*(ux+uy) + LBE_b*(ux+uy)*(ux+uy) ); }
__device__ __host__ __forceinline__ LBE_T LBE_f8(LBE_T rho, LBE_T ux, LBE_T uy, LBE_T uu){ return LBE_w2*rho*(1 - LBE_d*uu + LBE_a*(ux-uy) + LBE_b*(uy-ux)*(uy-ux) ); }

#endif
