all: lapl3d

%: %.cu
	nvcc -arch=sm_20 -I../ lapl3d.cu -o $@

