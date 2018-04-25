#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

#define INTENSIVE_GMF
#include "x11/cuda_x11_aes_alexis.cuh"

__device__
static void echo_round(const uint32_t sharedMemory[4][256], uint32_t *W, uint32_t &k0){
	// Big Sub Words
	#pragma unroll 16
	for (int idx = 0; idx < 16; idx++)
		AES_2ROUND(sharedMemory,W[(idx<<2) + 0], W[(idx<<2) + 1], W[(idx<<2) + 2], W[(idx<<2) + 3], k0);

	// Shift Rows
	#pragma unroll 4
	for (int i = 0; i < 4; i++){
		uint32_t t[4];
		/// 1, 5, 9, 13
		t[0] = W[i+ 4];
		t[1] = W[i+ 8];
		t[2] = W[i+24];
		t[3] = W[i+60];
		W[i + 4] = W[i + 20];
		W[i + 8] = W[i + 40];
		W[i +24] = W[i + 56];
		W[i +60] = W[i + 44];

		W[i +20] = W[i +36];
		W[i +40] = t[1];
		W[i +56] = t[2];
		W[i +44] = W[i +28];
				
		W[i +28] = W[i +12];
		W[i +12] = t[3];
		W[i +36] = W[i +52];
		W[i +52] = t[0];
	}
	// Mix Columns
	#pragma unroll 4
	for (int i = 0; i < 4; i++){ // Schleife über je 2*uint32_t
		#pragma unroll 4
		for (int idx = 0; idx < 64; idx += 16){ // Schleife über die elemnte
			uint32_t a[4];
			a[0] = W[idx + i];
			a[1] = W[idx + i + 4];
			a[2] = W[idx + i + 8];
			a[3] = W[idx + i +12];

			uint32_t ab = a[0] ^ a[1];
			uint32_t bc = a[1] ^ a[2];
			uint32_t cd = a[2] ^ a[3];

			uint32_t t, t2, t3;
			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			uint32_t abx = (t  >> 7) * 27U ^ ((ab^t) << 1);
			uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[idx + i] = bc ^ a[3] ^ abx;
			W[idx + i + 4] = a[0] ^ cd ^ bcx;
			W[idx + i + 8] = ab ^ a[3] ^ cdx;
			W[idx + i +12] = ab ^ a[2] ^ (abx ^ bcx ^ cdx);
		}
	}
}

#define TPB_E 256
__global__ __launch_bounds__(TPB_E, 2) /* will force 80 registers */
static void xevan_echo512_gpu_hash_128(uint32_t threads, uint32_t *g_hash)
{
	__shared__ uint32_t sharedMemory[4][256];

	aes_gpu_init256(sharedMemory);
//__syncthreads();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint32_t k0;
	uint32_t h[16];
	uint32_t hash[16];
	if (thread < threads){

		uint32_t *Hash = &g_hash[thread<<4];

		*(uint2x4*)&h[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
		*(uint2x4*)&h[ 8] = __ldg4((uint2x4*)&Hash[ 8]);
		
//		__syncthreads();
		uint32_t W[64];
		
		#define FILL_L 512UL
		#define FILL_H 0UL
		#define FILL_X 0UL

		W[0]=FILL_L;
        W[1]=FILL_H;
        W[2]=FILL_X;
        W[3]=FILL_X;

        W[4]=FILL_L;
        W[5]=FILL_H;
        W[6]=FILL_X;
        W[7]=FILL_X;

        W[8]=FILL_L;
        W[9]=FILL_H;
        W[10]=FILL_X;
        W[11]=FILL_X;

        W[12]=FILL_L;
        W[13]=FILL_H;
        W[14]=FILL_X;
        W[15]=FILL_X;

        W[16]=FILL_L;
        W[17]=FILL_H;
        W[18]=FILL_X;
        W[19]=FILL_X;

        W[20]=FILL_L;
        W[21]=FILL_H;
        W[22]=FILL_X;
        W[23]=FILL_X;

        W[24]=FILL_L;
        W[25]=FILL_H;
        W[26]=FILL_X;
        W[27]=FILL_X;

        W[28]=FILL_L;
        W[29]=FILL_H;
        W[30]=FILL_X;
        W[31]=FILL_X;

		uint32_t k0=1024;
	
		uint32_t Z[32];

//#pragma unroll 32
		for(int i=0;i<32;i++)Z[i]=W[i];
//#pragma unroll 16
		for(int i=32;i<48;i++)W[i]=h[i-32];
//#pragma unroll 16
                for(int i=48;i<64;i++)W[i]=0;

__syncthreads();

//#pragma unroll 5
		for(int i=0;i<10;i++)
			echo_round(sharedMemory,W,k0);

#pragma unroll 16
		for(int i=0;i<16;i++)
			Z[i] ^= h[i] ^ W[i] ^ W[32+i];

#pragma unroll 16
		for(int i=16;i<32;i++) Z[i] ^=W[i] ^ W[32+i]; 

//#pragma unroll 32
		for(int i=0;i<32;i++) W[i] = Z[i];

		W[32]=0x80;

//#pragma unroll 31
		for(int i=33;i<64;i++) W[i] = 0;
		
		W[59]=0x2000000;
		W[60]=0x400;

		k0=0;

		for(int i=0;i<10;i++){
            echo_round(sharedMemory,W,k0);
        }

		Z[0] ^= 0x80 ^ W[0] ^ W[32];

#pragma unroll 16
		for(int i=1;i<16;i++){
             Z[i] ^=  W[i] ^ W[i+32];
        }

		*(uint2x4*)&Hash[ 0] =  *(uint2x4*)&Z[ 0];
		*(uint2x4*)&Hash[ 8] =  *(uint2x4*)&Z[ 8];
	}
}

__host__
void xevan_echo512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash){

	const uint32_t threadsperblock = TPB_E;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	xevan_echo512_gpu_hash_128<<<grid, block>>>(threads, d_hash);
}