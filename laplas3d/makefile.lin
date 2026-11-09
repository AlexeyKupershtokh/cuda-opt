CXX         =g++

all: lapl3d

%: %.cup 
	./cup lapl3d.cup lapl3d.cu
	nvcc -O3 -arch=sm_20 -I../ lapl3d.cu -o $@

