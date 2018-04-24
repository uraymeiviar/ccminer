// Auf QuarkCoin spezialisierte Version von Groestl inkl. Bitslice

#include <stdio.h>
#include <memory.h>
#include <sys/types.h> // off_t

#include <cuda_helper.h>
#include <cuda_vectors.h>
#include <cuda_vectors_alexis.h>

#ifdef __INTELLISENSE__
#define __CUDA_ARCH__ 500
#endif

#define TPB 256
#define THF 4U

#if __CUDA_ARCH__ >= 300
#include "groestl_functions_quad.h"
#include "groestl_transf_quad.h"
#endif

#define WANT_GROESTL80
#ifdef WANT_GROESTL80
__constant__ static uint32_t c_Message80[20];
#endif

#include "cuda_quark_groestl512_sm2.cuh"

__global__ __launch_bounds__(TPB, THF)
void quark_groestl512_gpu_hash_64_quad(const uint32_t threads, const uint32_t startNounce, uint32_t * g_hash, uint32_t * __restrict g_nonceVector)
{
#if __CUDA_ARCH__ >= 300

	// BEWARE : 4-WAY CODE (one hash need 4 threads)
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;

	if (thread < threads)
	{
		uint32_t message[8];
		uint32_t state[8];

		uint32_t nounce = g_nonceVector ? g_nonceVector[thread] : (startNounce + thread);
		off_t hashPosition = nounce - startNounce;
		uint32_t *pHash = &g_hash[hashPosition << 4];

		const uint32_t thr = threadIdx.x & 0x3; // % THF

		/*| M0 M1 M2 M3 | M4 M5 M6 M7 | (input)
		--|-------------|-------------|
		T0|  0  4  8 12 | 80          |
		T1|  1  5    13 |             |
		T2|  2  6    14 |             |
		T3|  3  7    15 |          01 |
		--|-------------|-------------| */

		#pragma unroll
		for(int k=0;k<4;k++) message[k] = pHash[thr + (k * THF)];

		#pragma unroll
		for(int k=4;k<8;k++) message[k] = 0;

		if (thr == 0) message[4] = 0x80U; // end of data tag
		if (thr == 3) message[7] = 0x01000000U;

		uint32_t msgBitsliced[8];
		to_bitslice_quad(message, msgBitsliced);

		groestl512_progressMessage_quad(state, msgBitsliced);

		uint32_t hash[16];
		from_bitslice_quad(state, hash);

		// uint4 = 4x4 uint32_t = 16 bytes
		if (thr == 0) {
			uint4 *phash = (uint4*) hash;
			uint4 *outpt = (uint4*) pHash;
			outpt[0] = phash[0];
			outpt[1] = phash[1];
			outpt[2] = phash[2];
			outpt[3] = phash[3];
		}
	}
#endif
}

__host__
void quark_groestl512_cpu_init(int thr_id, uint32_t threads)
{
	int dev_id = device_map[thr_id];
	cuda_get_arch(thr_id);
	if (device_sm[dev_id] < 300 || cuda_arch[dev_id] < 300)
		quark_groestl512_sm20_init(thr_id, threads);
}

__host__
void quark_groestl512_cpu_free(int thr_id)
{
	int dev_id = device_map[thr_id];
	if (device_sm[dev_id] < 300 || cuda_arch[dev_id] < 300)
		quark_groestl512_sm20_free(thr_id);
}

__host__
void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	uint32_t threadsperblock = TPB;

	// Compute 3.0 benutzt die registeroptimierte Quad Variante mit Warp Shuffle
	// mit den Quad Funktionen brauchen wir jetzt 4 threads pro Hash, daher Faktor 4 bei der Blockzahl
	const uint32_t factor = THF;

	dim3 grid(factor*((threads + threadsperblock-1)/threadsperblock));
	dim3 block(threadsperblock);

	int dev_id = device_map[thr_id];

	if (device_sm[dev_id] >= 300 && cuda_arch[dev_id] >= 300)
		quark_groestl512_gpu_hash_64_quad<<<grid, block>>>(threads, startNounce, d_hash, d_nonceVector);
	else
		quark_groestl512_sm20_hash_64(thr_id, threads, startNounce, d_nonceVector, d_hash, order);
}

// --------------------------------------------------------------------------------------------------------------------------------------------

#ifdef WANT_GROESTL80

__host__
void groestl512_setBlock_80(int thr_id, uint32_t *endiandata)
{
	cudaMemcpyToSymbol(c_Message80, endiandata, sizeof(c_Message80), 0, cudaMemcpyHostToDevice);
}

__global__ __launch_bounds__(TPB, THF)
void groestl512_gpu_hash_80_quad(const uint32_t threads, const uint32_t startNounce, uint32_t * g_outhash)
{
#if __CUDA_ARCH__ >= 300
	// BEWARE : 4-WAY CODE (one hash need 4 threads)
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;
	if (thread < threads)
	{
		const uint32_t thr = threadIdx.x & 0x3; // % THF

		/*| M0 M1 M2 M3 M4 | M5 M6 M7 | (input)
		--|----------------|----------|
		T0|  0  4  8 12 16 | 80       |
		T1|  1  5       17 |          |
		T2|  2  6       18 |          |
		T3|  3  7       Nc |       01 |
		--|----------------|----------| TPR */

		uint32_t message[8];

		#pragma unroll 5
		for(int k=0; k<5; k++) message[k] = c_Message80[thr + (k * THF)];

		#pragma unroll 3
		for(int k=5; k<8; k++) message[k] = 0;

		if (thr == 0) message[5] = 0x80U;
		if (thr == 3) {
			message[4] = cuda_swab32(startNounce + thread);
			message[7] = 0x01000000U;
		}

		uint32_t msgBitsliced[8];
		to_bitslice_quad(message, msgBitsliced);

		uint32_t state[8];
		groestl512_progressMessage_quad(state, msgBitsliced);

		uint32_t hash[16];
		from_bitslice_quad(state, hash);

		if (thr == 0) { /* 4 threads were done */
			const off_t hashPosition = thread;
			//if (!thread) hash[15] = 0xFFFFFFFF;
			uint4 *outpt = (uint4*) &g_outhash[hashPosition << 4];
			uint4 *phash = (uint4*) hash;
			outpt[0] = phash[0];
			outpt[1] = phash[1];
			outpt[2] = phash[2];
			outpt[3] = phash[3];
		}
	}
#endif
}

__host__
void groestl512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	int dev_id = device_map[thr_id];

	if (device_sm[dev_id] >= 300 && cuda_arch[dev_id] >= 300) {
		const uint32_t threadsperblock = TPB;
		const uint32_t factor = THF;

		dim3 grid(factor*((threads + threadsperblock-1)/threadsperblock));
		dim3 block(threadsperblock);

		groestl512_gpu_hash_80_quad <<<grid, block>>> (threads, startNounce, d_hash);

	} else {

		const uint32_t threadsperblock = 256;
		dim3 grid((threads + threadsperblock-1)/threadsperblock);
		dim3 block(threadsperblock);

		groestl512_gpu_hash_80_sm2 <<<grid, block>>> (threads, startNounce, d_hash);
	}
}

#endif

#define TPB128 512

__global__ __launch_bounds__(TPB128, 2)
void quark_groestl512_gpu_hash_128_quad(const uint32_t threads,  uint32_t * g_hash)
{
#if __CUDA_ARCH__ >= 300

	// BEWARE : 4-WAY CODE (one hash need 4 threads)
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;

	if (thread < threads)
	{
		uint32_t m[8];
		uint32_t g[8];
		uint32_t h[8];
		uint32_t m2[8];

		uint32_t *pHash = &g_hash[thread << 4];

		const uint32_t thr = threadIdx.x & 0x3; // % THF

		/*| M0 M1 M2 M3 | M4 M5 M6 M7 | (input)
		--|-------------|-------------|
		T0|  0  4  8 12 | 80          |
		T1|  1  5    13 |             |
		T2|  2  6    14 |             |
		T3|  3  7    15 |          01 |
		--|-------------|-------------| */

		#pragma unroll
		for(int k=0;k<4;k++) m[k] = pHash[thr + (k * THF)];

		#pragma unroll
		for(int k=4;k<8;k++) m[k] = 0;

		if (thr == 0) m[4] = 0x0U; // end of data tag
		if (thr == 3) m[7] = 0x0000000U;

		uint32_t mBitsliced[8];
		to_bitslice_quad(m, mBitsliced);

		#pragma unroll
		for(int k=0;k<4;k++) g[k] = pHash[thr + (k * THF)];

		#pragma unroll
		for(int k=4;k<8;k++) g[k] = 0;

		if (thr == 0) g[4] = 0x0U; // end of data tag
		if (thr == 3) g[7] = 0x0020000U;

		uint32_t gBitsliced[8];
		to_bitslice_quad(g, gBitsliced);

		#pragma unroll
		for(int k=0;k<4;k++) h[k] = 0;

		#pragma unroll
		for(int k=4;k<8;k++) h[k] = 0;

		if (thr == 0) h[4] = 0x0U; // end of data tag
		if (thr == 3) h[7] = 0x0020000U;

		uint32_t hBitsliced[8];
		to_bitslice_quad(h, hBitsliced);

		#pragma unroll
		for(int k=0;k<8;k++) m2[k] = 0;

		if (thr == 0) m2[0] = 0x80U; // end of data tag
		if (thr == 3) m2[7] = 0x2000000;

		uint32_t m2Bitsliced[8];
		to_bitslice_quad(m2, m2Bitsliced);

        groestl512_perm_P_quad(gBitsliced);
		groestl512_perm_Q_quad(mBitsliced);

		for (unsigned int u = 0; u < 8; u++){
			hBitsliced[u] ^= gBitsliced[u] ^ mBitsliced[u];
		}

		for (unsigned int u = 0; u < 8; u++){
			gBitsliced[u] = m2Bitsliced[u] ^ hBitsliced[u];
		}
        groestl512_perm_P_quad(gBitsliced);

		if(thr == 0){
			m2Bitsliced[0]=0x89aecd65;
			m2Bitsliced[1]=0x64a6d130;
			m2Bitsliced[2]=0x3f3d9e18;
			m2Bitsliced[3]=0xae0389d4;
			m2Bitsliced[4]=0xbbf2c8a2;
			m2Bitsliced[5]=0x3b1b2f4;
			m2Bitsliced[6]=0xeca737be;
			m2Bitsliced[7]=0xe4d92093;
		}

		if(thr == 1){
			m2Bitsliced[0]=0x813d1bbf;
			m2Bitsliced[1]=0x64aea6;
			m2Bitsliced[2]=0xcac17604;
			m2Bitsliced[3]=0x7edc9d98;
			m2Bitsliced[4]=0xf895469;
			m2Bitsliced[5]=0x3450f60c;
			m2Bitsliced[6]=0xedaae1a4;
			m2Bitsliced[7]=0x363761e9;
		}

		if(thr == 2){
			m2Bitsliced[0]=0xb81a7b17;
			m2Bitsliced[1]=0x322e9ee6;
			m2Bitsliced[2]=0x1ce5c5cd;
			m2Bitsliced[3]=0x79e2d9b0;
			m2Bitsliced[4]=0x7734ec9c;
			m2Bitsliced[5]=0xde433ef;
			m2Bitsliced[6]=0x7459f800;
			m2Bitsliced[7]=0xec98575b;
		}

		if(thr == 3){
			m2Bitsliced[0]=0xe1eb324e;
			m2Bitsliced[1]=0x30530c30;
			m2Bitsliced[2]=0xef2f21d0;
			m2Bitsliced[3]=0x8a0194b3;
			m2Bitsliced[4]=0x9516fd30;
			m2Bitsliced[5]=0xd8f3a4bf;
			m2Bitsliced[6]=0x4d3cbccd;
			m2Bitsliced[7]=0x5ac552f;
		}

        uint32_t hxBitsliced[8];

		for (unsigned int u = 0; u < 8; u++){
			hxBitsliced[u] = hBitsliced[u] ^= gBitsliced[u] ^ m2Bitsliced[u];
		}

        groestl512_perm_P_quad(hxBitsliced);

		for (unsigned int u = 0; u < 8; u++){
			hxBitsliced[u] ^= hBitsliced[u];
		}

		uint32_t hash[16];
		from_bitslice_quad(hxBitsliced, hash);

		if (thr == 0) {
			uint2x4 *phash = (uint2x4*) hash;
			uint2x4 *outpt = (uint2x4*) pHash;
			outpt[0] = phash[0];
			outpt[1] = phash[1];
		}
	}
#endif
}

__host__
void quark_groestl512_cpu_hash_128(int thr_id, uint32_t threads,  uint32_t *d_hash)
{
	uint32_t threadsperblock = TPB128;
	const uint32_t factor = THF;

	dim3 grid(factor*((threads + threadsperblock-1)/threadsperblock));
	dim3 block(threadsperblock);
	quark_groestl512_gpu_hash_128_quad<<<grid, block>>>(threads, d_hash);
}