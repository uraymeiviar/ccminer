# To change the cuda arch, edit Makefile.am and run ./build.sh

extracflags="-march=x86-64 -D_REENTRANT -falign-functions=16 -falign-jumps=16 -falign-labels=16 -static-libstdc++"

CUDA_CFLAGS="-O3 -lineno -Xcompiler -Wall  -D_FORCE_INLINES -Wno-deprecated-declarations --cudart=static" \
	./configure CXXFLAGS="-O3 $extracflags" --with-cuda=/usr/local/cuda --with-nvml=libnvidia-ml.so

