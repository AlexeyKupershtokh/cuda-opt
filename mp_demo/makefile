ALL=mp
all: $(ALL)
clean:
	rm -f *.o $(ALL) *.tif *.png *.eps
	
%: %.cpp  
	g++ $< -lfltk -lX11 -o $@
