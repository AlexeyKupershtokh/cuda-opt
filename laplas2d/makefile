CXX         =g++

all: lapl2d lapl2d2 lapl2d3

%.cu: %.cup 
	./cup $< $@

%: %.cu
	nvcc -O3 -arch=sm_20 -I../ $< -o $@

