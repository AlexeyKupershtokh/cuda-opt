#ifndef CUDA_CORE_H_
#define CUDA_CORE_H_

#include <string>

#include "../mp/mp.h"

extern "C" MP_EXPORT std::string str_info;

extern "C" MP_EXPORT void gpu_info();

#endif
