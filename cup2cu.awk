function removecode(){
	for(i=argc*2+4; i<=NF; ++i) if ($i == "#"){ return } 
	while ( 0==0){
		getline ;
		printf("\n");
		for(i=1; i<=NF; ++i) if ($i == "#"){ return }
	}
}
BEGIN{
    counter=0
}

/CUP/ {
	if ( $1 == "//"){
		print($0); 
		next;
	}
	if ($2 == "INSERT"){
		pref=$3;
		system("cat qwert.qwert; rm -rf qwert.qwert");
		next;
	}
	#print("//"$0); 
	
	if ( $2 == "FORALL_3D" || $2 == "FORALL_3D_IN" || $2 == "FORALL_3D_IJK" ){
		ind=3
		while( $ind != "#" ) ind++;
		argc=(ind-3)/2;
		print("{")
			print("dim3 threads(MP_parallel.Nx, MP_parallel.Ny, MP_parallel.Nz);")
			print("dim3 grid("$4".Nx/threads.x, "$4".Ny/threads.y, (("$4".Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.z);")
			print("cudaStream_t s[4];")
			print("for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){");
				print("cudaSetDevice(MP_parallel.GPU_pool[i]);");
				print("s[i] = 0;")
				printf("CUP_3D_"argc"_"pref""counter"<<<grid, threads, 0, s[i]>>>(i, split(i, "$4")");
				for(i=6 ; i<=(argc*2+2) ; i+=2)	printf(", split(i, "$i")");
				print(");");
			print("}")

			#printf("cudaThreadSynchronize();")
			print("for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){");
				print("cudaSetDevice(MP_parallel.GPU_pool[i]);");
				print("")
			print("}")
		print("}")
		removecode();
	}

	if ( $2 == "FORALL_3DYZ" ){
		ind=3
		while( $ind != "#" ) ind++;
		argc=(ind-3)/2;
		print("{")
			print("dim3 threads(MP_parallel.Ny, MP_parallel.Nz_bound);")
			print("dim3 grid("$4".Ny/threads.x, (("$4".Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.y);")
			
			print("cudaStream_t s[4];")
			print("for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){");
				print("cudaSetDevice(MP_parallel.GPU_pool[i]);");
				print("s[i] = 0;")
				printf("CUP_3DYZ_"argc"_"pref""counter"<<<grid, threads, 0, s[i]>>>(split(i, "$4")");
				for(i=6 ; i<=(argc*2+2) ; i+=2)	printf(", split(i, "$i")");
				print(");");
			print("}")

			print("for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){");
				print("cudaSetDevice(MP_parallel.GPU_pool[i]);");
				print("")
			print("}")		
		print("}")
		removecode();
	}
	
	if ( $2 == "FORALL_3DXZ"  ){
		ind=3
		while( $ind != "#" ) ind++;
		argc=(ind-3)/2;		
		print("{")
			print("dim3 threads(MP_parallel.Nx, MP_parallel.Nz_bound);")
			print("dim3 grid("$4".Nx/threads.x, (("$4".Nz-2)/MP_parallel.GPU_pool.size()+2)/threads.y);")
		
			print("cudaStream_t s[4];")
			print("for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){");
				print("cudaSetDevice(MP_parallel.GPU_pool[i]);");
				print("s[i] = 0;")
				printf("CUP_3DXZ_"argc"_"pref""counter"<<<grid, threads, 0, s[i]>>>(split(i, "$4")");
				for(i=6 ; i<=(argc*2+2) ; i+=2)	printf(", split(i, "$i")");
				print(");");
			print("}")

			print("for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){");
				print("cudaSetDevice(MP_parallel.GPU_pool[i]);");
				print("")
			print("}")		
		print("}")
		removecode();
	}
	
	if ( $2 == "FORALL_3DXY" ){
		ind=3
		while( $ind != "#" ) ind++;
		argc=(ind-3)/2;
		print("{")
			print("dim3 threads(MP_parallel.Nx, MP_parallel.Ny);")
			print("dim3 grid("$4".Nx/threads.x, "$4".Ny/threads.y);")
			
			print("cudaStream_t s[4];")
			print("for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){");
				print("cudaSetDevice(MP_parallel.GPU_pool[i]);");
				print("s[i] = 0;")
				printf("CUP_3DXY_"argc"_"pref""counter"<<<grid, threads, 0, s[i]>>>(i, split(i, "$4")");
				for(i=6 ; i<=(argc*2+2) ; i+=2)	printf(", split(i, "$i")");
				print(");");
			print("}")

			print("for(size_t i=0; i<MP_parallel.GPU_pool.size(); i++){");
				print("cudaSetDevice(MP_parallel.GPU_pool[i]);");
				print("");
			print("}")
		print("}")
		removecode();
	}
	
	counter++;
	next
}
{print $0}

END { 
#print "/*End cup parser*/" 
}

