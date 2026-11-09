function printijk(){
	print("const size_t i=blockIdx.x*blockDim.x+threadIdx.x;");		
	print("const size_t j=blockIdx.y*blockDim.y+threadIdx.y;");
	print("const size_t k=blockIdx.z*blockDim.z+threadIdx.z;");
}
function printcode(){
	for(i=argc*2+4; i<=NF; ++i) if ($i == "#"){ return } else printf($i" ");
	while (0==0){
		getline ;
		printf("\n");
		for(i=1; i<=NF; ++i) if ($i == "#"){ return } else printf($i" ");
	}
	
}

BEGIN{
    #system("rm -rf qwert.qwert ; touch qwert.qwert");
    #print "/*Begin cup parser!*/"; 
    pref="";
    counter=0
}

/CUP/ {
	if ( $1 == "//") next
	if ($2 == "INSERT"){
		pref=$3;
		next;
	}

	print("//"$0);
	
	if ( $2 == "FORALL_3D" ){
		ind=3
		while( $ind != "#" ) ind++;
		argc=(ind-3)/2;

		printf("template<typename T1");
		for(i=2; i<=argc; i++) printf(", typename T"i)
		printf("> __global__ void CUP_3D_"argc"_"pref""counter"(int gpuIdx, T1 "$3);
		for(i=2 ; i<=argc; i++)	printf(", T"i" "$(1+i*2));
		printf("){");
			print("const size_t ind=(blockIdx.z*blockDim.z+threadIdx.z)*"$3".Nx*"$3".Ny+(blockIdx.y*blockDim.y+threadIdx.y)*"$3".Nx +   blockIdx.x * blockDim.x + threadIdx.x;");
			printcode(); 
		printf("}\n\n");
	}
	if ( $2 == "FORALL_3D_IN" ){
		ind=3
		while( $ind != "#" ) ind++;
		argc=(ind-3)/2;

		printf("template<typename T1");
		for(i=2; i<=argc; i++) printf(", typename T"i)
		printf("> __global__ void CUP_3D_"argc"_"pref""counter"(int gpuIdx, T1 "$3);
		for(i=2 ; i<=argc; i++)	printf(", T"i" "$(1+i*2));
		printf("){");
			printijk();
			print("if (i>0 && i < "$3".Nx-1 && j>0 && j < "$3".Ny-1 && k>0 && k < "$3".Nz-1){");
				print("const size_t ind=k*"$3".Nx*"$3".Ny+j*"$3".Nx+i;");		
				printcode();
			printf("}\n");
		printf("}\n\n");
	}	

	if ( $2 == "FORALL_3D_IJK" ){
		ind=3
		while( $ind != "#" ) ind++;
		argc=(ind-3)/2;

		printf("template<typename T1");
		for(i=2; i<=argc; i++) printf(", typename T"i)
		printf("> __global__ void CUP_3D_"argc"_"pref""counter"(int gpuIdx, T1 "$3);
		for(i=2 ; i<=argc; i++)	printf(", T"i" "$(1+i*2));
		printf("){");
		printijk();
		print("const size_t ind=k*"$3".Nx*"$3".Ny+j*"$3".Nx+i;");		
		printcode();
		printf("}\n\n");		
	}

	if ( $2 == "FORALL_3DYZ" ){
		ind=3
		while( $ind != "#" ) ind++;
		argc=(ind-3)/2;

		printf("template<typename T1");
		for(i=2; i<=argc; i++) printf(", typename T"i)
		printf("> __global__ void CUP_3DYZ_"argc"_"pref""counter"(T1 "$3);
		for(i=2 ; i<=argc; i++)	printf(", T"i" "$(1+i*2));
		printf("){");
		
		print("const size_t j=blockIdx.x*blockDim.x+threadIdx.x;");
		print("const size_t k=blockIdx.y*blockDim.y+threadIdx.y;");
		print("const size_t ind=k*"$3".Nx*"$3".Ny+j*"$3".Nx;");		
		printcode();
		printf("}\n\n");		
	}	
	
	if ( $2 == "FORALL_3DXZ" ){
		ind=3
		while( $ind != "#" ) ind++;
		argc=(ind-3)/2;

		printf("template<typename T1");
		for(i=2; i<=argc; i++) printf(", typename T"i)
		printf("> __global__ void CUP_3DXZ_"argc"_"pref""counter"(T1 "$3);
		for(i=2 ; i<=argc; i++)	printf(", T"i" "$(1+i*2));
		printf("){");
		
		print("const size_t i=blockIdx.x*blockDim.x+threadIdx.x;");
		print("const size_t k=blockIdx.y*blockDim.y+threadIdx.y;");
		print("const size_t ind=k*"$3".Nx*"$3".Ny+i;");
		printcode();		
		printf("}\n\n");		
	}	

	if ( $2 == "FORALL_3DXY" ){
		ind=3
		while( $ind != "#" ) ind++;
		argc=(ind-3)/2;

		printf("template<typename T1");
		for(i=2; i<=argc; i++) printf(", typename T"i)
		printf("> __global__ void CUP_3DXY_"argc"_"pref""counter"(int gpuIdx, T1 "$3);
		for(i=2 ; i<=argc; i++)	printf(", T"i" "$(1+i*2));
		printf("){");

		print("const size_t i=blockIdx.x*blockDim.x+threadIdx.x;");
		print("const size_t j=blockIdx.y*blockDim.y+threadIdx.y;");
		print("const size_t ind=i+j*"$3".Nx;");		
		printcode();
		printf("}\n\n");		
	}	

	if ( $2 == "FORALL_3DX" ){
		ind=3
		while( $ind != "#" ) ind++;
		argc=(ind-3)/2;

		printf("template<typename T1");
		for(i=2; i<=argc; i++) printf(", typename T"i)
		printf("> __global__ void CUP_3DX_"argc"_"pref""counter"(T1 "$3);
		for(i=2 ; i<=argc; i++)	printf(", T"i" "$(1+i*2));
		printf("){");
		print("const size_t ind=blockIdx.x*blockDim.x+threadIdx.x;");
		printcode();
		printf("}\n\n");		
	}

	if ( $2 == "FORALL_3DY" ){
		ind=3
		while( $ind != "#" ) ind++;
		argc=(ind-3)/2;
		printf("template<typename T1");
		for(i=2; i<=argc; i++) printf(", typename T"i)
		printf("> __global__ void CUP_3DY_"argc"_"pref""counter"(T1 "$3);
		for(i=2 ; i<=argc; i++)	printf(", T"i" "$(1+i*2));
		printf("){");
		print("const size_t ind=(blockIdx.x*blockDim.x+threadIdx.x)*"$3".Nx;");
		printcode();
		printf("}\n\n");		
	}

	if ( $2 == "FORALL_3DZ" ){
		ind=3
		while( $ind != "#" ) ind++;
		argc=(ind-3)/2;
		printf("template<typename T1");
		for(i=2; i<=argc; i++) printf(", typename T"i)
		printf("> __global__ void CUP_3DZ_"argc"_"pref""counter"(T1 "$3);
		for(i=2 ; i<=argc; i++)	printf(", T"i" "$(1+i*2));
		printf("){");
		print("const size_t ind=(blockIdx.x*blockDim.x+threadIdx.x)*"$3".Nx*"$3".Ny;");
		printcode();
		printf("}\n\n");		
	}

	
	counter++;
	next
}

END { 
#print "/*End cup parser*/" 
}

