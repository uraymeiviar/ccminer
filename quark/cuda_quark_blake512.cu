#include <stdio.h>
#include <memory.h>
#include <sys/types.h> // off_t

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_helper_alexis.h"
#include "cuda_vectors.h"
#include "cuda_vector_uint2x4.h"

#define ROTR(x,n) ROTR64(x,n)

// use sp kernel on SM 5+
#define SP_KERNEL

#define USE_SHUFFLE 0

__constant__
static uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)

// ---------------------------- BEGIN CUDA quark_blake512 functions ------------------------------------

__device__ __constant__
static const uint8_t c_sigma_big[16][16] = {
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },

	{12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13 , 0 },

	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 }
};

__device__ __constant__
static const uint64_t c_u512[16] =
{
	0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL,
	0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL,
	0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL,
	0xc0ac29b7c97c50ddULL, 0x3f84d5b5b5470917ULL,
	0x9216d5d98979fb1bULL, 0xd1310ba698dfb5acULL,
	0x2ffd72dbd01adfb7ULL, 0xb8e1afed6a267e96ULL,
	0xba7c9045f12c7f99ULL, 0x24a19947b3916cf7ULL,
	0x0801f2e2858efc16ULL, 0x636920d871574e69ULL
};

#define G(a,b,c,d,x) { \
	uint32_t idx1 = sigma[i][x]; \
	uint32_t idx2 = sigma[i][x+1]; \
	v[a] += (m[idx1] ^ u512[idx2]) + v[b]; \
	v[d] = SWAPDWORDS(v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROTR( v[b] ^ v[c], 25); \
	v[a] += (m[idx2] ^ u512[idx1]) + v[b]; \
	v[d] = ROTR( v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = ROTR( v[b] ^ v[c], 11); \
}

__device__ __forceinline__
void quark_blake512_compress(uint64_t *h, const uint64_t *block, const uint8_t ((*sigma)[16]), const uint64_t *u512, const int T0)
{
	uint64_t v[16];
	uint64_t m[16];

	#pragma unroll
	for(int i=0; i < 16; i++) {
		m[i] = cuda_swab64(block[i]);
	}

	//#pragma unroll 8
	for(int i=0; i < 8; i++)
		v[i] = h[i];

	v[ 8] = u512[0];
	v[ 9] = u512[1];
	v[10] = u512[2];
	v[11] = u512[3];
	v[12] = u512[4] ^ T0;
	v[13] = u512[5] ^ T0;
	v[14] = u512[6];
	v[15] = u512[7];

	//#pragma unroll 16
	for(int i=0; i < 16; i++)
	{
		/* column step */
		G( 0, 4, 8, 12, 0 );
		G( 1, 5, 9, 13, 2 );
		G( 2, 6, 10, 14, 4 );
		G( 3, 7, 11, 15, 6 );
		/* diagonal step */
		G( 0, 5, 10, 15, 8 );
		G( 1, 6, 11, 12, 10 );
		G( 2, 7, 8, 13, 12 );
		G( 3, 4, 9, 14, 14 );
	}

	h[0] ^= v[0] ^ v[8];
	h[1] ^= v[1] ^ v[9];
	h[2] ^= v[2] ^ v[10];
	h[3] ^= v[3] ^ v[11];
	h[4] ^= v[4] ^ v[12];
	h[5] ^= v[5] ^ v[13];
	h[6] ^= v[6] ^ v[14];
	h[7] ^= v[7] ^ v[15];
}

__global__ __launch_bounds__(256, 4)
void quark_blake512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *g_nonceVector, uint64_t *g_hash)
{
#if !defined(SP_KERNEL) || __CUDA_ARCH__ < 500
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

#if USE_SHUFFLE
	const uint32_t warpBlockID = (thread + 15)>>4; // aufrunden auf volle Warp-Blöcke

	if (warpBlockID < ( (threads+15)>>4 ))
#else
	if (thread < threads)
#endif
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		off_t hashPosition = nounce - startNounce;
		uint64_t *inpHash = &g_hash[hashPosition<<3]; // hashPosition * 8

		// 128 Bytes
		uint64_t buf[16];

		// State
		uint64_t h[8] = {
			0x6a09e667f3bcc908ULL,
			0xbb67ae8584caa73bULL,
			0x3c6ef372fe94f82bULL,
			0xa54ff53a5f1d36f1ULL,
			0x510e527fade682d1ULL,
			0x9b05688c2b3e6c1fULL,
			0x1f83d9abfb41bd6bULL,
			0x5be0cd19137e2179ULL
		};

		// Message for first round
		#pragma unroll 8
		for (int i=0; i < 8; ++i)
			buf[i] = inpHash[i];

		// Hash Pad
		buf[8]  = 0x0000000000000080ull;
		buf[9]  = 0;
		buf[10] = 0;
		buf[11] = 0;
		buf[12] = 0;
		buf[13] = 0x0100000000000000ull;
		buf[14] = 0;
		buf[15] = 0x0002000000000000ull;

		// Ending round
		quark_blake512_compress(h, buf, c_sigma_big, c_u512, 512);

#if __CUDA_ARCH__ <= 350
		uint32_t *outHash = (uint32_t*)&g_hash[hashPosition * 8U];
		#pragma unroll 8
		for (int i=0; i < 8; i++) {
			outHash[2*i+0] = cuda_swab32( _HIDWORD(h[i]) );
			outHash[2*i+1] = cuda_swab32( _LODWORD(h[i]) );
		}
#else
		uint64_t *outHash = &g_hash[hashPosition * 8U];
		for (int i=0; i < 8; i++) {
			outHash[i] = cuda_swab64(h[i]);
		}
#endif
	}
#endif /* SP */
}

__global__ __launch_bounds__(256,4)
void quark_blake512_gpu_hash_80(uint32_t threads, uint32_t startNounce, void *outputHash)
{
//#if !defined(SP_KERNEL) || __CUDA_ARCH__ < 500
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t buf[16];
		#pragma unroll
		for (int i=0; i < 16; ++i)
			buf[i] = c_PaddedMessage80[i];

		// The test Nonce
		const uint32_t nounce = startNounce + thread;
		((uint32_t*)buf)[19] = cuda_swab32(nounce);

		uint64_t h[8] = {
			0x6a09e667f3bcc908ULL,
			0xbb67ae8584caa73bULL,
			0x3c6ef372fe94f82bULL,
			0xa54ff53a5f1d36f1ULL,
			0x510e527fade682d1ULL,
			0x9b05688c2b3e6c1fULL,
			0x1f83d9abfb41bd6bULL,
			0x5be0cd19137e2179ULL
		};

		quark_blake512_compress(h, buf, c_sigma_big, c_u512, 640);

#if __CUDA_ARCH__ <= 350
		uint32_t *outHash = (uint32_t*)outputHash + (thread * 16U);
		#pragma unroll 8
		for (uint32_t i=0; i < 8; i++) {
			outHash[2*i]   = cuda_swab32( _HIDWORD(h[i]) );
			outHash[2*i+1] = cuda_swab32( _LODWORD(h[i]) );
		}
#else
		uint64_t *outHash = (uint64_t*)outputHash + (thread * 8U);
		for (uint32_t i=0; i < 8; i++) {
			outHash[i] = cuda_swab64( h[i] );
		}
#endif
	}
//#endif
}

#ifdef SP_KERNEL
#include "cuda_quark_blake512_sp.cuh"
#endif

__host__
void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_outputHash, int order)
{
#ifdef SP_KERNEL
	int dev_id = device_map[thr_id];
	if (device_sm[dev_id] >= 500 && cuda_arch[dev_id] >= 500)
		quark_blake512_cpu_hash_64_sp(threads, startNounce, d_nonceVector, d_outputHash);
	else
#endif
	{
		const uint32_t threadsperblock = 256;
		dim3 grid((threads + threadsperblock-1)/threadsperblock);
		dim3 block(threadsperblock);
		quark_blake512_gpu_hash_64<<<grid, block>>>(threads, startNounce, d_nonceVector, (uint64_t*)d_outputHash);
	}
	//MyStreamSynchronize(NULL, order, thr_id);
}

__host__
void quark_blake512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash)
{
#ifdef SP_KERNEL
	int dev_id = device_map[thr_id];
	if (device_sm[dev_id] >= 500 && cuda_arch[dev_id] >= 500)
		quark_blake512_cpu_hash_80_sp(threads, startNounce, d_outputHash);
	else
#endif
	{
		const uint32_t threadsperblock = 256;
		dim3 grid((threads + threadsperblock-1)/threadsperblock);
		dim3 block(threadsperblock);

		quark_blake512_gpu_hash_80<<<grid, block>>>(threads, startNounce, d_outputHash);
	}
}

// ---------------------------- END CUDA quark_blake512 functions ------------------------------------

__host__
void quark_blake512_cpu_init(int thr_id, uint32_t threads)
{
	cuda_get_arch(thr_id);
}

__host__
void quark_blake512_cpu_free(int thr_id)
{
}

// ----------------------------- Host midstate for 80-bytes input ------------------------------------

#undef SPH_C32
#undef SPH_T32
#undef SPH_C64
#undef SPH_T64

extern "C" {
#include "sph/sph_blake.h"
}

__host__
void quark_blake512_cpu_setBlock_80(int thr_id, uint32_t *endiandata)
{
#ifdef SP_KERNEL
	int dev_id = device_map[thr_id];
	if (device_sm[dev_id] >= 500 && cuda_arch[dev_id] >= 500)
		quark_blake512_cpu_setBlock_80_sp(thr_id, (uint64_t*) endiandata);
	else
#endif
	{
		uint64_t message[16];

		memcpy(message, endiandata, 80);
		message[10] = 0x80;
		message[11] = 0;
		message[12] = 0;
		message[13] = 0x0100000000000000ull;
		message[14] = 0;
		message[15] = 0x8002000000000000ull; // 0x280

		cudaMemcpyToSymbol(c_PaddedMessage80, message, sizeof(message), 0, cudaMemcpyHostToDevice);
	}
	CUDA_LOG_ERROR();
}

//=====================================================================================
__constant__ uint2 _ALIGN(16) c_m[16]; // padded message (80 bytes + padding)
__constant__ uint2 _ALIGN(16) c_v[16]; //state
__constant__ uint2 _ALIGN(16) c_x[128]; //precomputed xors

#define GShost(a,b,c,d,e,f) { \
	v[a] += (m[e] ^ z[f]) + v[b]; \
	v[d] = ROTR64(v[d] ^ v[a],32); \
	v[c] += v[d]; \
	v[b] = ROTR64( v[b] ^ v[c], 25); \
	v[a] += (m[f] ^ z[e]) + v[b]; \
	v[d] = ROTR64( v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = ROTR64( v[b] ^ v[c], 11); \
}

__host__
void xevan_blake512_cpu_setBlock_80(int thr_id, uint32_t *endiandata){
	uint64_t m[16],v[16],xors[128];
	memcpy(m, endiandata, 80);
	m[10] = 0x8000000000000000ull;
	m[11] = 0;
	m[12] = 0;
	m[13] = 0x01;
	m[14] = 0;
	m[15] = 0x280;

	for(int i=0;i<10;i++){
		m[ i] = cuda_swab64(m[ i]);
	}
	
	uint64_t h[8] = {
		0x6a09e667f3bcc908ULL,	0xbb67ae8584caa73bULL,	0x3c6ef372fe94f82bULL,	0xa54ff53a5f1d36f1ULL,
		0x510e527fade682d1ULL,	0x9b05688c2b3e6c1fULL,	0x1f83d9abfb41bd6bULL,	0x5be0cd19137e2179ULL
	};

	const uint64_t z[16] = {
		0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL,	0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL,
		0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL,	0xc0ac29b7c97c50ddULL, 0x3f84d5b5b5470917ULL,
		0x9216d5d98979fb1bULL, 0xd1310ba698dfb5acULL,	0x2ffd72dbd01adfb7ULL, 0xb8e1afed6a267e96ULL,
		0xba7c9045f12c7f99ULL, 0x24a19947b3916cf7ULL,	0x0801f2e2858efc16ULL, 0x636920d871574e69ULL
	};

	for(int i=0;i<8;i++){
		v[ i] = h[ i];
	}
	v[ 8] = z[0];
	v[ 9] = z[1];
	v[10] = z[2];
	v[11] = z[3];
	v[12] = z[4] ^ 640;
	v[13] = z[5] ^ 640;
	v[14] = z[6];
	v[15] = z[7];
	
	/* column step */
	GShost( 0, 4, 8,12, 0, 1);
	GShost( 1, 5, 9,13, 2, 3);
	GShost( 2, 6,10,14, 4, 5);
	GShost( 3, 7,11,15, 6, 7);

	GShost( 1, 6,11,12,10,11);
	GShost( 2, 7, 8,13,12,13);
	GShost( 3, 4, 9,14,14,15);

	v[ 0]+= (m[ 8] ^ z[ 9]) + v[ 5];
	v[15] = ROTR64(v[15]^v[ 0],32);
	v[10]+= v[15];
	v[ 5] = ROTR64(v[ 5] ^ v[10], 25);

	v[ 0]+= v[ 5];
	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_m, m, sizeof(m), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_v, v, sizeof(m), 0, cudaMemcpyHostToDevice));

	int i=0;
	
	xors[i++] = m[ 4]^z[ 8];
	xors[i++] = m[ 8]^z[ 4];	
	xors[i++] = m[ 6]^z[13];
	xors[i++] = m[ 1]^z[12];	
	xors[i++] = m[ 0]^z[ 2];
	xors[i++] = m[ 5]^z[ 3];
	xors[i++] = m[ 2]^z[ 0];	
	xors[i++] = m[ 7]^z[11];	
	xors[i++] = m[ 3]^z[ 5];
//2
	xors[i++] = m[ 5]^z[ 2];
	xors[i++] = m[ 8]^z[11];	
	xors[i++] = m[ 0]^z[12];
	xors[i++] = m[ 2]^z[ 5];
	xors[i++] = m[ 3]^z[ 6];	
	xors[i++] = m[ 7]^z[ 1];	
	xors[i++] = m[ 6]^z[ 3];	
	xors[i++] = m[ 1]^z[ 7];	
	xors[i++] = m[ 4]^z[ 9];
//3
	xors[i++] = m[ 7]^z[ 9];
	xors[i++] = m[ 3]^z[ 1];
	xors[i++] = m[ 1]^z[ 3];
	xors[i++] = m[14]^z[11];
	xors[i++] = m[ 2]^z[ 6];
	xors[i++] = m[ 5]^z[10];
	xors[i++] = m[ 4]^z[ 0];
	xors[i++] = m[ 6]^z[ 2];
	xors[i++] = m[ 0]^z[ 4];
	xors[i++] = m[ 8]^z[15];
//4
	xors[i++] = m[ 5]^z[ 7];
	xors[i++] = m[ 2]^z[ 4];
	xors[i++] = m[ 0]^z[ 9];
	xors[i++] = m[ 7]^z[ 5];
	xors[i++] = m[ 4]^z[ 2];
	xors[i++] = m[ 6]^z[ 8];
	xors[i++] = m[ 3]^z[13];
	xors[i++] = m[ 1]^z[14];
	xors[i++] = m[ 8]^z[ 6];
//5
	xors[i++] = m[ 2]^z[12];
	xors[i++] = m[ 6]^z[10];
	xors[i++] = m[ 0]^z[11];
	xors[i++] = m[ 8]^z[ 3];
	xors[i++] = m[ 3]^z[ 8];
	xors[i++] = m[ 4]^z[13];
	xors[i++] = m[ 7]^z[ 5];	
	xors[i++] = m[ 1]^z[ 9];
	xors[i++] = m[ 5]^z[ 7];
//6
	xors[i++] = m[ 1]^z[15];
	xors[i++] = m[ 4]^z[10];
	xors[i++] = m[ 5]^z[12];
	xors[i++] = m[ 0]^z[ 7];
	xors[i++] = m[ 6]^z[ 3];
	xors[i++] = m[ 8]^z[11];
	xors[i++] = m[ 7]^z[ 0];
	xors[i++] = m[ 3]^z[ 6];
	xors[i++] = m[ 2]^z[ 9];
//7
	xors[i++] = m[ 7]^z[14];
	xors[i++] = m[ 3]^z[ 9];
	xors[i++] = m[ 1]^z[12];
	xors[i++] = m[ 5]^z[ 0];
	xors[i++] = m[ 8]^z[ 6];
	xors[i++] = m[ 2]^z[10];
	xors[i++] = m[ 0]^z[ 5];
	xors[i++] = m[ 4]^z[15];
	xors[i++] = m[ 6]^z[ 8];
//8
	xors[i++] = m[ 6]^z[15];
	xors[i++] = m[ 0]^z[ 8];
	xors[i++] = m[ 3]^z[11];
	xors[i++] = m[ 8]^z[ 0];
	xors[i++] = m[ 1]^z[ 4];
	xors[i++] = m[ 2]^z[12];
	xors[i++] = m[ 7]^z[13];
	xors[i++] = m[ 4]^z[ 1];
	xors[i++] = m[ 5]^z[10];
//9
	xors[i++] = m[ 8]^z[ 4];
	xors[i++] = m[ 7]^z[ 6];
	xors[i++] = m[ 1]^z[ 5];
	xors[i++] = m[ 2]^z[10];
	xors[i++] = m[ 4]^z[ 8];
	xors[i++] = m[ 6]^z[ 7];
	xors[i++] = m[ 5]^z[ 1];
	xors[i++] = m[ 3]^z[12];
	xors[i++] = m[ 0]^z[13];
//10
	xors[i++] = m[ 0]^z[ 1];
	xors[i++] = m[ 2]^z[ 3];
	xors[i++] = m[ 4]^z[ 5];
	xors[i++] = m[ 6]^z[ 7];
	xors[i++] = m[ 1]^z[ 0];
	xors[i++] = m[ 3]^z[ 2];
	xors[i++] = m[ 5]^z[ 4];
	xors[i++] = m[ 7]^z[ 6];
	xors[i++] = m[ 8]^z[ 9];

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_x,xors, i*sizeof(uint2), 0, cudaMemcpyHostToDevice));
}


//=====================================================================================

#define TPB_128_B 128

__global__ __launch_bounds__(TPB_128_B, 3)
void quark_blake512_gpu_hash_128(uint32_t threads,  uint64_t *g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		//uint64_t *inpHash = &g_hash[thread<<3]; // hashPosition * 8

		// 128 Bytes
		uint64_t buf[16];

		// State
		uint64_t h[8] = {
			0x6a09e667f3bcc908ULL,
			0xbb67ae8584caa73bULL,
			0x3c6ef372fe94f82bULL,
			0xa54ff53a5f1d36f1ULL,
			0x510e527fade682d1ULL,
			0x9b05688c2b3e6c1fULL,
			0x1f83d9abfb41bd6bULL,
			0x5be0cd19137e2179ULL
		};

		uint2x4 *phash = (uint2x4*)&g_hash[thread<<3];
		uint2x4 *outpt = (uint2x4*)buf;
		outpt[0] = __ldg4(&phash[0]);
		outpt[1] = __ldg4(&phash[1]);

		// Message for first round
		#pragma unroll 8
		for (int i=0; i < 8; ++i){
			buf[i] = cuda_swab64(buf[i]);
			//buf[i+8]=0;
		}

		// Hash Pad
		buf[8]  = 0;
		buf[9]  = 0;
		buf[10] = 0;
		buf[11] = 0;
		buf[12] = 0;
		buf[13] = 0;
		buf[14] = 0;
		buf[15] = 0;
		// Ending round
		quark_blake512_compress(h, buf, c_sigma_big, c_u512, 1024);

		buf[0] = 0x8000000000000000;
		buf[1] = 0;
		buf[2] = 0;
		buf[3] = 0;
		buf[4] = 0;
		buf[5] = 0;
		buf[6] = 0;
		buf[7] = 0;
		buf[8] = 0;
		buf[9] = 0;
		buf[10] = 0;
		buf[11] = 0;
		buf[12] = 0;
		buf[13] = 1;
		buf[14] = 0;
		buf[15] = 0x400;

		quark_blake512_compress(h, buf, c_sigma_big, c_u512, 0);

		uint64_t *outHash = &g_hash[thread * 8U];
		#pragma unroll 8
		for (int i=0; i < 8; i++) {
			outHash[i] = cuda_swab64(h[i]);
		}
	}
}

__host__
void quark_blake512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_outputHash)
{
		const uint32_t threadsperblock = TPB_128_B;
		dim3 grid((threads + threadsperblock-1)/threadsperblock);
		dim3 block(threadsperblock);
		quark_blake512_gpu_hash_128<<<grid, block>>>(threads,  (uint64_t*)d_outputHash);
}


//=================================================================================
#define TPB80 256

#define TPB52_64 192
#define TPB50_64 192

// Wolf's BMW512, loosely based on SPH's implementation
#define as_uint2(x) (x)
#define FAST_ROTL64_LO ROTL64
#define FAST_ROTL64_HI ROTL64

#define CONST_EXP2  q[i+0] + FAST_ROTL64_LO(as_uint2(q[i+1]), 5)  + q[i+2] + FAST_ROTL64_LO(as_uint2(q[i+3]), 11) + \
                    q[i+4] + FAST_ROTL64_LO(as_uint2(q[i+5]), 27) + q[i+6] + as_ulong(as_uint2(q[i+7]).s10) + \
                    q[i+8] + FAST_ROTL64_HI(as_uint2(q[i+9]), 37) + q[i+10] + FAST_ROTL64_HI(as_uint2(q[i+11]), 43) + \
                    q[i+12] + FAST_ROTL64_HI(as_uint2(q[i+13]), 53) + (SHR(q[i+14],1) ^ q[i+14]) + (SHR(q[i+15],2) ^ q[i+15])

#define SHL(x, n) ((x) << (n))
#define SHR(x, n) ((x) >> (n))

#define s64_0(x)  (SHR((x), 1) ^ SHL((x), 3) ^ FAST_ROTL64_LO(as_uint2((x)),  4) ^ FAST_ROTL64_HI(as_uint2((x)), 37))
#define s64_1(x)  (SHR((x), 1) ^ SHL((x), 2) ^ FAST_ROTL64_LO(as_uint2((x)), 13) ^ FAST_ROTL64_HI(as_uint2((x)), 43))
#define s64_2(x)  (SHR((x), 2) ^ SHL((x), 1) ^ FAST_ROTL64_LO(as_uint2((x)), 19) ^ FAST_ROTL64_HI(as_uint2((x)), 53))
#define s64_3(x)  (SHR((x), 2) ^ SHL((x), 2) ^ FAST_ROTL64_LO(as_uint2((x)), 28) ^ FAST_ROTL64_HI(as_uint2((x)), 59))
#define s64_4(x)  (SHR((x), 1) ^ (x))
#define s64_5(x)  (SHR((x), 2) ^ (x))

#define r64_01(x) FAST_ROTL64_LO(as_uint2((x)),  5)
#define r64_02(x) FAST_ROTL64_LO(as_uint2((x)), 11)
#define r64_03(x) FAST_ROTL64_LO(as_uint2((x)), 27)
#define r64_04(x) devectorize(SWAPDWORDS2(vectorize((x))))
#define r64_05(x) FAST_ROTL64_HI(as_uint2((x)), 37)
#define r64_06(x) FAST_ROTL64_HI(as_uint2((x)), 43)
#define r64_07(x) FAST_ROTL64_HI(as_uint2((x)), 53)

#define Q0	s64_0( (BMW_H[ 5] ^ msg[ 5])-(BMW_H[ 7] ^ msg[ 7])+(BMW_H[10] ^ msg[10])+(BMW_H[13] ^ msg[13])+(BMW_H[14] ^ msg[14])) + BMW_H[1]
#define Q1	s64_1( (BMW_H[ 6] ^ msg[ 6])-(BMW_H[ 8] ^ msg[ 8])+(BMW_H[11] ^ msg[11])+(BMW_H[14] ^ msg[14])-(BMW_H[15] ^ msg[15])) + BMW_H[2]
#define Q2 	s64_2( (BMW_H[ 0] ^ msg[ 0])+(BMW_H[ 7] ^ msg[ 7])+(BMW_H[ 9] ^ msg[ 9])-(BMW_H[12] ^ msg[12])+(BMW_H[15] ^ msg[15])) + BMW_H[3]
#define Q3	s64_3( (BMW_H[ 0] ^ msg[ 0])-(BMW_H[ 1] ^ msg[ 1])+(BMW_H[ 8] ^ msg[ 8])-(BMW_H[10] ^ msg[10])+(BMW_H[13] ^ msg[13])) + BMW_H[4]
#define Q4	s64_4( (BMW_H[ 1] ^ msg[ 1])+(BMW_H[ 2] ^ msg[ 2])+(BMW_H[ 9] ^ msg[ 9])-(BMW_H[11] ^ msg[11])-(BMW_H[14] ^ msg[14])) + BMW_H[5]
#define Q5	s64_0( (BMW_H[ 3] ^ msg[ 3])-(BMW_H[ 2] ^ msg[ 2])+(BMW_H[10] ^ msg[10])-(BMW_H[12] ^ msg[12])+(BMW_H[15] ^ msg[15])) + BMW_H[6]
#define Q6	s64_1( (BMW_H[ 4] ^ msg[ 4])-(BMW_H[ 0] ^ msg[ 0])-(BMW_H[ 3] ^ msg[ 3])-(BMW_H[11] ^ msg[11])+(BMW_H[13] ^ msg[13])) + BMW_H[7]
#define Q7	s64_2( (BMW_H[ 1] ^ msg[ 1])-(BMW_H[ 4] ^ msg[ 4])-(BMW_H[ 5] ^ msg[ 5])-(BMW_H[12] ^ msg[12])-(BMW_H[14] ^ msg[14])) + BMW_H[8]
#define Q8	s64_3( (BMW_H[ 2] ^ msg[ 2])-(BMW_H[ 5] ^ msg[ 5])-(BMW_H[ 6] ^ msg[ 6])+(BMW_H[13] ^ msg[13])-(BMW_H[15] ^ msg[15])) + BMW_H[9]
#define Q9	s64_4( (BMW_H[ 0] ^ msg[ 0])-(BMW_H[ 3] ^ msg[ 3])+(BMW_H[ 6] ^ msg[ 6])-(BMW_H[ 7] ^ msg[ 7])+(BMW_H[14] ^ msg[14])) + BMW_H[10]
#define Q10	s64_0( (BMW_H[ 8] ^ msg[ 8])-(BMW_H[ 1] ^ msg[ 1])-(BMW_H[ 4] ^ msg[ 4])-(BMW_H[ 7] ^ msg[ 7])+(BMW_H[15] ^ msg[15])) + BMW_H[11]
#define Q11	s64_1( (BMW_H[ 8] ^ msg[ 8])-(BMW_H[ 0] ^ msg[ 0])-(BMW_H[ 2] ^ msg[ 2])-(BMW_H[ 5] ^ msg[ 5])+(BMW_H[ 9] ^ msg[ 9])) + BMW_H[12]
#define Q12	s64_2( (BMW_H[ 1] ^ msg[ 1])+(BMW_H[ 3] ^ msg[ 3])-(BMW_H[ 6] ^ msg[ 6])-(BMW_H[ 9] ^ msg[ 9])+(BMW_H[10] ^ msg[10])) + BMW_H[13]
#define Q13	s64_3( (BMW_H[ 2] ^ msg[ 2])+(BMW_H[ 4] ^ msg[ 4])+(BMW_H[ 7] ^ msg[ 7])+(BMW_H[10] ^ msg[10])+(BMW_H[11] ^ msg[11])) + BMW_H[14]
#define Q14	s64_4( (BMW_H[ 3] ^ msg[ 3])-(BMW_H[ 5] ^ msg[ 5])+(BMW_H[ 8] ^ msg[ 8])-(BMW_H[11] ^ msg[11])-(BMW_H[12] ^ msg[12])) + BMW_H[15]
#define Q15	s64_0( (BMW_H[12] ^ msg[12])-(BMW_H[ 4] ^ msg[ 4])-(BMW_H[ 6] ^ msg[ 6])-(BMW_H[ 9] ^ msg[ 9])+(BMW_H[13] ^ msg[13])) + BMW_H[0]

__device__ __forceinline__ uint64_t BMW_Expand1(uint32_t i, const  uint64_t * msg, const  uint64_t * q, const  uint64_t * h)
{
	return ( s64_1(q[i - 16])          + s64_2(q[i - 15])   + s64_3(q[i - 14]  ) + s64_0(q[i - 13] ) \
           + s64_1(q[i - 12])          + s64_2(q[i - 11])   + s64_3(q[i - 10]  ) + s64_0(q[i -  9] ) \
		   + s64_1(q[i -  8])          + s64_2(q[i -  7])   + s64_3(q[i -  6]  ) + s64_0(q[i -  5] ) \
		   + s64_1(q[i -  4])          + s64_2(q[i -  3])   + s64_3(q[i -  2]  ) + s64_0(q[i -  1] ) \
		   + ((i*(0x0555555555555555ull) + FAST_ROTL64_LO(as_uint2(msg[i - 16]), ((i - 16) + 1)) + FAST_ROTL64_LO(as_uint2(msg[(i-13)]), ((i - 13) + 1)) - FAST_ROTL64_LO(as_uint2(msg[i - 6]), ((i - 6) + 1))) ^ h[((i - 16) + 7)]));
}

__device__ __forceinline__ uint64_t BMW_Expand2(uint32_t i, const uint64_t * msg, const uint64_t * q, const uint64_t * h)
{
	return ( q[i - 16] + r64_01(q[i - 15])  + q[i - 14] + r64_02(q[i - 13]) + \
                    q[i - 12] + r64_03(q[i - 11]) + q[i - 10] + r64_04(q[i - 9]) + \
                    q[i - 8] + r64_05(q[i - 7]) + q[i - 6] + r64_06(q[i - 5]) + \
                    q[i - 4] + r64_07(q[i - 3]) + s64_4(q[i - 2]) + s64_5(q[i - 1]) + \
		   ((i*(0x0555555555555555ull) + FAST_ROTL64_LO(as_uint2(msg[i - 16]), (i - 16) + 1) + FAST_ROTL64_LO(as_uint2(msg[(i - 13) & 15]), ((i - 13) & 15) + 1) - FAST_ROTL64_LO(as_uint2(msg[(i - 6) & 15]), ((i - 6) & 15) + 1)) ^ h[((i - 16) + 7) & 15]));
}

__device__ __forceinline__  void BMW_Compression(uint64_t * msg, const uint64_t *__restrict__ BMW_H, uint64_t *q)
{
	q[ 0] = s64_0( (BMW_H[ 5] ^ msg[ 5])-(BMW_H[ 7] ^ msg[ 7])+(BMW_H[10] ^ msg[10])+(BMW_H[13] ^ msg[13])+(BMW_H[14] ^ msg[14])) + BMW_H[1];
	q[ 1] = s64_1( (BMW_H[ 6] ^ msg[ 6])-(BMW_H[ 8] ^ msg[ 8])+(BMW_H[11] ^ msg[11])+(BMW_H[14] ^ msg[14])-(BMW_H[15] ^ msg[15])) + BMW_H[2];
	q[ 2] = s64_2( (BMW_H[ 0] ^ msg[ 0])+(BMW_H[ 7] ^ msg[ 7])+(BMW_H[ 9] ^ msg[ 9])-(BMW_H[12] ^ msg[12])+(BMW_H[15] ^ msg[15])) + BMW_H[3];
	q[ 3] = s64_3( (BMW_H[ 0] ^ msg[ 0])-(BMW_H[ 1] ^ msg[ 1])+(BMW_H[ 8] ^ msg[ 8])-(BMW_H[10] ^ msg[10])+(BMW_H[13] ^ msg[13])) + BMW_H[4];
	q[ 4] = s64_4( (BMW_H[ 1] ^ msg[ 1])+(BMW_H[ 2] ^ msg[ 2])+(BMW_H[ 9] ^ msg[ 9])-(BMW_H[11] ^ msg[11])-(BMW_H[14] ^ msg[14])) + BMW_H[5];
	q[ 5] = s64_0( (BMW_H[ 3] ^ msg[ 3])-(BMW_H[ 2] ^ msg[ 2])+(BMW_H[10] ^ msg[10])-(BMW_H[12] ^ msg[12])+(BMW_H[15] ^ msg[15])) + BMW_H[6];
	q[ 6] = s64_1( (BMW_H[ 4] ^ msg[ 4])-(BMW_H[ 0] ^ msg[ 0])-(BMW_H[ 3] ^ msg[ 3])-(BMW_H[11] ^ msg[11])+(BMW_H[13] ^ msg[13])) + BMW_H[7];
	q[ 7] = s64_2( (BMW_H[ 1] ^ msg[ 1])-(BMW_H[ 4] ^ msg[ 4])-(BMW_H[ 5] ^ msg[ 5])-(BMW_H[12] ^ msg[12])-(BMW_H[14] ^ msg[14])) + BMW_H[8];
	q[ 8] = s64_3( (BMW_H[ 2] ^ msg[ 2])-(BMW_H[ 5] ^ msg[ 5])-(BMW_H[ 6] ^ msg[ 6])+(BMW_H[13] ^ msg[13])-(BMW_H[15] ^ msg[15])) + BMW_H[9];
	q[ 9] = s64_4( (BMW_H[ 0] ^ msg[ 0])-(BMW_H[ 3] ^ msg[ 3])+(BMW_H[ 6] ^ msg[ 6])-(BMW_H[ 7] ^ msg[ 7])+(BMW_H[14] ^ msg[14])) + BMW_H[10];
	q[10] = s64_0( (BMW_H[ 8] ^ msg[ 8])-(BMW_H[ 1] ^ msg[ 1])-(BMW_H[ 4] ^ msg[ 4])-(BMW_H[ 7] ^ msg[ 7])+(BMW_H[15] ^ msg[15])) + BMW_H[11];
	q[11] = s64_1( (BMW_H[ 8] ^ msg[ 8])-(BMW_H[ 0] ^ msg[ 0])-(BMW_H[ 2] ^ msg[ 2])-(BMW_H[ 5] ^ msg[ 5])+(BMW_H[ 9] ^ msg[ 9])) + BMW_H[12];
	q[12] = s64_2( (BMW_H[ 1] ^ msg[ 1])+(BMW_H[ 3] ^ msg[ 3])-(BMW_H[ 6] ^ msg[ 6])-(BMW_H[ 9] ^ msg[ 9])+(BMW_H[10] ^ msg[10])) + BMW_H[13];
	q[13] = s64_3( (BMW_H[ 2] ^ msg[ 2])+(BMW_H[ 4] ^ msg[ 4])+(BMW_H[ 7] ^ msg[ 7])+(BMW_H[10] ^ msg[10])+(BMW_H[11] ^ msg[11])) + BMW_H[14];
	q[14] = s64_4( (BMW_H[ 3] ^ msg[ 3])-(BMW_H[ 5] ^ msg[ 5])+(BMW_H[ 8] ^ msg[ 8])-(BMW_H[11] ^ msg[11])-(BMW_H[12] ^ msg[12])) + BMW_H[15];
	q[15] = s64_0( (BMW_H[12] ^ msg[12])-(BMW_H[ 4] ^ msg[ 4])-(BMW_H[ 6] ^ msg[ 6])-(BMW_H[ 9] ^ msg[ 9])+(BMW_H[13] ^ msg[13])) + BMW_H[0];
	
	#pragma unroll 16
	for(int i = 0; i < 16; ++i) q[i + 16] = (i < 2) ? BMW_Expand1(i + 16, msg, q, BMW_H) : BMW_Expand2(i + 16, msg, q, BMW_H);
			
	const ulong XL64 = q[16]^q[17]^q[18]^q[19]^q[20]^q[21]^q[22]^q[23];
	const ulong XH64 = XL64^q[24]^q[25]^q[26]^q[27]^q[28]^q[29]^q[30]^q[31];
		
	msg[0] = (SHL(XH64, 5) ^ SHR(q[16],5) ^ msg[0]) + ( XL64 ^ q[24] ^ q[0]);
	msg[1] = (SHR(XH64, 7) ^ SHL(q[17],8) ^ msg[1]) + ( XL64 ^ q[25] ^ q[1]);
	msg[2] = (SHR(XH64, 5) ^ SHL(q[18],5) ^ msg[2]) + ( XL64 ^ q[26] ^ q[2]);
	msg[3] = (SHR(XH64, 1) ^ SHL(q[19],5) ^ msg[3]) + ( XL64 ^ q[27] ^ q[3]);
	msg[4] = (SHR(XH64, 3) ^ q[20] ^ msg[4]) + ( XL64 ^ q[28] ^ q[4]);
	msg[5] = (SHL(XH64, 6) ^ SHR(q[21],6) ^ msg[5]) + ( XL64 ^ q[29] ^ q[5]);
	msg[6] = (SHR(XH64, 4) ^ SHL(q[22],6) ^ msg[6]) + ( XL64 ^ q[30] ^ q[6]);
	msg[7] = (SHR(XH64,11) ^ SHL(q[23],2) ^ msg[7]) + ( XL64 ^ q[31] ^ q[7]);

	msg[8] = FAST_ROTL64_LO(as_uint2(msg[4]), 9) + ( XH64 ^ q[24] ^ msg[8]) + (SHL(XL64,8) ^ q[23] ^ q[8]);
	msg[9] = FAST_ROTL64_LO(as_uint2(msg[5]),10) + ( XH64 ^ q[25] ^ msg[9]) + (SHR(XL64,6) ^ q[16] ^ q[9]);
	msg[10] = FAST_ROTL64_LO(as_uint2(msg[6]),11) + ( XH64 ^ q[26] ^ msg[10]) + (SHL(XL64,6) ^ q[17] ^ q[10]);
	msg[11] = FAST_ROTL64_LO(as_uint2(msg[7]),12) + ( XH64 ^ q[27] ^ msg[11]) + (SHL(XL64,4) ^ q[18] ^ q[11]);
	msg[12] = FAST_ROTL64_LO(as_uint2(msg[0]),13) + ( XH64 ^ q[28] ^ msg[12]) + (SHR(XL64,3) ^ q[19] ^ q[12]);
	msg[13] = FAST_ROTL64_LO(as_uint2(msg[1]),14) + ( XH64 ^ q[29] ^ msg[13]) + (SHR(XL64,4) ^ q[20] ^ q[13]);
	msg[14] = FAST_ROTL64_LO(as_uint2(msg[2]),15) + ( XH64 ^ q[30] ^ msg[14]) + (SHR(XL64,7) ^ q[21] ^ q[14]);
	msg[15] = FAST_ROTL64_LO(as_uint2(msg[3]),16) + ( XH64 ^ q[31] ^ msg[15]) + (SHR(XL64,2) ^ q[22] ^ q[15]);
}

__constant__ const  uint64_t BMW512_IV[16] =
{
	0x8081828384858687UL, 0x88898A8B8C8D8E8FUL, 0x9091929394959697UL, 0x98999A9B9C9D9E9FUL,
	0xA0A1A2A3A4A5A6A7UL, 0xA8A9AAABACADAEAFUL, 0xB0B1B2B3B4B5B6B7UL, 0xB8B9BABBBCBDBEBFUL,
	0xC0C1C2C3C4C5C6C7UL, 0xC8C9CACBCCCDCECFUL, 0xD0D1D2D3D4D5D6D7UL, 0xD8D9DADBDCDDDEDFUL,
	0xE0E1E2E3E4E5E6E7UL, 0xE8E9EAEBECEDEEEFUL, 0xF0F1F2F3F4F5F6F7UL, 0xF8F9FAFBFCFDFEFFUL
};

__constant__ const  uint64_t BMW512_FINAL[16] =
{
	0xAAAAAAAAAAAAAAA0UL, 0xAAAAAAAAAAAAAAA1UL, 0xAAAAAAAAAAAAAAA2UL, 0xAAAAAAAAAAAAAAA3UL,
	0xAAAAAAAAAAAAAAA4UL, 0xAAAAAAAAAAAAAAA5UL, 0xAAAAAAAAAAAAAAA6UL, 0xAAAAAAAAAAAAAAA7UL,
	0xAAAAAAAAAAAAAAA8UL, 0xAAAAAAAAAAAAAAA9UL, 0xAAAAAAAAAAAAAAAAUL, 0xAAAAAAAAAAAAAAABUL,
	0xAAAAAAAAAAAAAAACUL, 0xAAAAAAAAAAAAAAADUL, 0xAAAAAAAAAAAAAAAEUL, 0xAAAAAAAAAAAAAAAFUL
};

#define GSn4(a,b,c,d,e,f,a1,b1,c1,d1,e1,f1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3){\
	v[ a] = v[ a] + e + v[ b];		v[a1] = v[a1] + e1 + v[b1];		v[a2] = v[a2] + e2 + v[b2];		v[a3] = v[a3] + e3 + v[b3];\
	v[ d] = SWAPDWORDS2(v[ d] ^ v[ a]);	v[d1] = SWAPDWORDS2(v[d1] ^ v[a1]);	v[d2] = SWAPDWORDS2(v[d2] ^ v[a2]);	v[d3] = SWAPDWORDS2(v[d3] ^ v[a3]);\
	v[ c] = v[ c] + v[ d];			v[c1] = v[c1] + v[d1];			v[c2] = v[c2] + v[d2];			v[c3] = v[c3] + v[d3];\
	v[ b] = ROR2(v[b] ^ v[c],25);		v[b1] = ROR2(v[b1] ^ v[c1],25);		v[b2] = ROR2(v[b2] ^ v[c2],25);		v[b3] = ROR2(v[b3] ^ v[c3],25); \
	v[ a] = v[ a] + f + v[ b];		v[a1] = v[a1] + f1 + v[b1];		v[a2] = v[a2] + f2 + v[b2];		v[a3] = v[a3] + f3 + v[b3];\
	v[ d] = ROR16(v[d] ^ v[a]);		v[d1] = ROR16(v[d1] ^ v[a1]);		v[d2] = ROR16(v[d2] ^ v[a2]);		v[d3] = ROR16(v[d3] ^ v[a3]);\
	v[ c] = v[ c] + v[ d];			v[c1] = v[c1] + v[d1];			v[c2] = v[c2] + v[d2];			v[c3] = v[c3] + v[d3];\
	v[ b] = ROR2(v[b] ^ v[c],11);		v[b1] = ROR2(v[b1] ^ v[c1],11);		v[b2] = ROR2(v[b2] ^ v[c2],11);		v[b3] = ROR2(v[b3] ^ v[c3],11);\
}

__device__ __forceinline__ uint2 cuda_swab64_U2(uint2 a)
{
	// Input:       77665544 33221100
	// Output:      00112233 44556677
	uint2 result;
	result.y = __byte_perm(a.x, 0, 0x0123);
	result.x = __byte_perm(a.y, 0, 0x0123);
	return result;
}

__global__ __launch_bounds__(TPB80,2)
void quark_blake512_gpu_hash_80_bmw_128(const uint32_t threads,const uint32_t startNounce, uint2x4 *const __restrict__ g_hash){
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint2 v[16];
	uint2 m[10];
	uint2 xors[16];

	const uint2 h[8] = {
		{0xf3bcc908,0x6a09e667}, {0x84caa73b,0xbb67ae85},{0xfe94f82b,0x3c6ef372}, {0x5f1d36f1,0xa54ff53a},
		{0xade682d1,0x510e527f}, {0x2b3e6c1f,0x9b05688c},{0xfb41bd6b,0x1f83d9ab}, {0x137e2179,0x5be0cd19}
	};
	const uint2 z[16] = {
		{0x85a308d3,0x243f6a88},{0x03707344,0x13198a2e},{0x299f31d0,0xa4093822},{0xec4e6c89,0x082efa98},
		{0x38d01377,0x452821e6},{0x34e90c6c,0xbe5466cf},{0xc97c50dd,0xc0ac29b7},{0xb5470917,0x3f84d5b5},
		{0x8979fb1b,0x9216d5d9},{0x98dfb5ac,0xd1310ba6},{0xd01adfb7,0x2ffd72db},{0x6a267e96,0xb8e1afed},
		{0xf12c7f99,0xba7c9045},{0xb3916cf7,0x24a19947},{0x858efc16,0x0801f2e2},{0x71574e69,0x636920d8}
	};
	const uint32_t m150 = 0x280 ^ z[ 9].x;//make_uint2(0x280,0) ^ z[ 9];//2
	const uint32_t m151 = 0x280 ^ z[13].x;//2
	const uint32_t m152 = 0x280 ^ z[ 8].x;//2
	const uint32_t m153 = 0x280 ^ z[10].x;//2
	const uint32_t m154 = 0x280 ^ z[14].x;//3
	const uint32_t m155 = 0x280 ^ z[ 1].x;//1
	const uint32_t m156 = 0x280 ^ z[ 4].x;//1
	const uint32_t m157 = 0x280 ^ z[ 6].x;//1
	const uint32_t m158 = 0x280 ^ z[11].x;//1

	const uint32_t m130 = 0x01 ^ z[ 6].x;//2
	const uint32_t m131 = 0x01 ^ z[15].x;//2
	const uint32_t m132 = 0x01 ^ z[12].x;//3
	const uint32_t m133 = 0x01 ^ z[ 3].x;//2
	const uint32_t m134 = 0x01 ^ z[ 4].x;//2
	const uint32_t m135 = 0x01 ^ z[14].x;//1
	const uint32_t m136 = 0x01 ^ z[11].x;//1
	const uint32_t m137 = 0x01 ^ z[ 7].x;//1
	const uint32_t m138 = 0x01 ^ z[ 0].x;//1

	const uint32_t m100 = 0x80000000 ^ z[14].y;//4
	const uint32_t m101 = 0x80000000 ^ z[ 5].y;//3
	const uint32_t m102 = 0x80000000 ^ z[15].y;//2
	const uint32_t m103 = 0x80000000 ^ z[ 6].y;//2
	const uint32_t m104 = 0x80000000 ^ z[ 4].y;//1
	const uint32_t m105 = 0x80000000 ^ z[ 2].y;//2
	const uint32_t m106 = 0x80000000 ^ z[11].y;//2

	if (thread < threads){
		int i=0;
		
		#pragma unroll 10
		for (int i=0; i < 10; ++i){
			m[i] = c_m[i];
		}

		m[ 9].x = startNounce + thread;

		#pragma unroll 16
		for(int i=0; i < 16; i++){
			v[i] = c_v[i];
		}

		v[ 0]+= (m[ 9] ^ z[ 8]);
		v[15] = ROR16(v[15] ^ v[ 0]);
		v[10]+= v[15];
		v[ 5] = ROR2(v[ 5] ^ v[10], 11);

		xors[ 0] = z[10];			xors[ 1] = c_x[i++];			xors[ 2] = m[ 9]^z[15];			xors[ 3] = make_uint2(m130,z[ 6].y);
		xors[ 4] = make_uint2(z[14].x,m100);	xors[ 5] = c_x[i++];			xors[ 6] = make_uint2(m150,z[9].y);	xors[ 7] = c_x[i++];
		
		xors[ 8] = c_x[i++];			xors[ 9] = c_x[i++];			xors[10] = z[ 7];			xors[11] = c_x[i++];
		xors[12] = z[ 1];			xors[13] = c_x[i++];			xors[14] = c_x[i++];			xors[15] = c_x[i++];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);

		//2:{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 }
		xors[ 0] = z[ 8];			xors[ 1] = z[ 0];			xors[ 2] = c_x[i++];			xors[ 3] = make_uint2(m151,z[13].y);
		xors[ 4] = c_x[i++];			xors[ 5] = c_x[i++];			xors[ 6] = c_x[i++];			xors[ 7] = make_uint2(m131,z[15].y);
		
		xors[ 8] = make_uint2(z[14].x,m100);	xors[ 9] = c_x[i++];			xors[10] = c_x[i++];			xors[11] = m[ 9]^z[ 4];
		xors[12] = z[10];			xors[13] = c_x[i++];			xors[14] = c_x[i++];			xors[15] = c_x[i++];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);

		//3:{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 }
		xors[ 0] = c_x[i++];			xors[ 1] = c_x[i++];			xors[ 2] = make_uint2(m132,z[12].y);	xors[ 3] = z[14];
		xors[ 4] = m[ 9]^z[ 7];			xors[ 5] = c_x[i++];			xors[ 6] = z[13];			xors[ 7] = c_x[i++];
		
		xors[ 8] = c_x[i++];			xors[ 9] = c_x[i++];			xors[10] = c_x[i++];			xors[11] = make_uint2(m152,z[ 8].y);
		xors[12] = c_x[i++];			xors[13] = make_uint2(z[ 5].x,m101);	xors[14] = c_x[i++];			xors[15] = c_x[i++];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);
		
		//4:{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 }
		xors[ 0] = m[ 9]^z[ 0];			xors[ 1] = c_x[i++];			xors[ 2] = c_x[i++];			xors[ 3] = make_uint2(z[15].x,m102);
		xors[ 4] = c_x[i++];			xors[ 5] = c_x[i++];			xors[ 6] = c_x[i++];			xors[ 7] = make_uint2(m153,z[10].y);
		
		xors[ 8] = z[ 1];			xors[ 9] = z[12];			xors[10] = c_x[i++];			xors[11] = c_x[i++];
		xors[12] = c_x[i++];			xors[13] = z[11];			xors[14] = c_x[i++];			xors[15] = make_uint2(m133,z[ 3].y);
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);
		
		//5:{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 }
		xors[ 0] = c_x[i++];			xors[ 1] = c_x[i++];			xors[ 2] = c_x[i++];			xors[ 3] = c_x[i++];
		xors[ 4] = z[ 2];			xors[ 5] = make_uint2(z[ 6].x,m103);	xors[ 6] = z[ 0];			xors[ 7] = c_x[i++];
		
		xors[ 8] = c_x[i++];			xors[ 9] = c_x[i++];			xors[10] = make_uint2(m154,z[14].y);	xors[11] = c_x[i++];
		xors[12] = make_uint2(m134,z[ 4].y);	xors[13] = c_x[i++];			xors[14] = z[15];			xors[15] = m[ 9]^z[ 1];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);
		
		//6:{12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 }
		xors[ 0] = z[ 5];			xors[ 1] = c_x[i++];			xors[ 2] = z[13];			xors[ 3] = c_x[i++];
		xors[ 4] = c_x[i++];			xors[ 5] = make_uint2(m155,z[ 1].y);	xors[ 6] = make_uint2(m135,z[14].y);	xors[ 7] = make_uint2(z[ 4].x,m104);
		
		xors[ 8] = c_x[i++];			xors[ 9] = c_x[i++];			xors[10] = m[ 9]^z[ 2];			xors[11] = c_x[i++];
		xors[12] = c_x[i++];			xors[13] = c_x[i++];			xors[14] = c_x[i++];			xors[15] = z[ 8];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);

		//7:{13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 }
		xors[ 0] = make_uint2(m136,z[11].y);	xors[ 1] = c_x[i++];			xors[ 2] = z[ 1];			xors[ 3] = c_x[i++];
		xors[ 4] = z[13];			xors[ 5] = z[ 7];			xors[ 6] = c_x[i++];			xors[ 7] = m[ 9]^z[ 3];
		
		xors[ 8] = c_x[i++];			xors[ 9] = make_uint2(m156,z[ 4].y);	xors[10] = c_x[i++];			xors[11] = c_x[i++];
		xors[12] = c_x[i++];			xors[13] = c_x[i++];			xors[14] = c_x[i++];			xors[15] = make_uint2(z[ 2].x,m105);
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);
		
		//8:{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 }
		xors[ 0] = c_x[i++];			xors[ 1] = z[ 9];			xors[ 2] = z[ 3];			xors[ 3] = c_x[i++];
		xors[ 4] = make_uint2(m157,z[ 6].y);	xors[ 5] = m[ 9] ^ z[14];		xors[ 6] = c_x[i++];			xors[ 7] = c_x[i++];

		xors[ 8] = z[ 2];			xors[ 9] = make_uint2(m137,z[ 7].y);	xors[10] = c_x[i++];			xors[11] = make_uint2(z[ 5].x,m101);
		xors[12] = c_x[i++];			xors[13] = c_x[i++];			xors[14] = c_x[i++];			xors[15] = c_x[i++];

		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);

		//9:{10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13 , 0 }
		xors[ 0] = make_uint2(z[ 2].x,m105);	xors[ 1] = c_x[i++];			xors[ 2] = c_x[i++];			xors[ 3] = c_x[i++];
		xors[ 4] = c_x[i++];			xors[ 5] = c_x[i++];			xors[ 6] = c_x[i++];			xors[ 7] = c_x[i++];
		
		xors[ 8] = make_uint2(m158,z[11].y);	xors[ 9] = m[ 9]^z[14];			xors[10] = c_x[i++];			xors[11] = make_uint2(m138,z[ 0].y);
		xors[12] = z[15];			xors[13] = z[ 9];			xors[14] = z[ 3];			xors[15] = c_x[i++];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);
		//10:{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }
		xors[ 0] = c_x[i++];		xors[ 1] = c_x[i++];				xors[ 2] = c_x[i++];			xors[ 3] = c_x[i++];
		xors[ 4] = c_x[i++];		xors[ 5] = c_x[i++];				xors[ 6] = c_x[i++];			xors[ 7] = c_x[i++];
		
		xors[ 8] = c_x[i++];		xors[ 9] = make_uint2(z[11].x,m106);		xors[10] = z[13];			xors[11] = z[15];
		xors[12] = m[ 9]^z[ 8];		xors[13] = z[10];				xors[14] = make_uint2(m132,z[12].y);	xors[15] = make_uint2(m154,z[14].y);
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);
//------------------
		i=0;
		//11:{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
		xors[ 0] = z[10];			xors[ 1] = c_x[i++];			xors[ 2] = m[ 9]^z[15];			xors[ 3] = make_uint2(m130,z[ 6].y);
		xors[ 4] = make_uint2(z[14].x,m100);	xors[ 5] = c_x[i++];			xors[ 6] = make_uint2(m150,z[9].y);	xors[ 7] = c_x[i++];
		
		xors[ 8] = c_x[i++];			xors[ 9] = c_x[i++];			xors[10] = z[ 7];			xors[11] = c_x[i++];
		xors[12] = z[ 1];			xors[13] = c_x[i++];			xors[14] = c_x[i++];			xors[15] = c_x[i++];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);

		//12:{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 }
		xors[ 0] = z[ 8];			xors[ 1] = z[ 0];			xors[ 2] = c_x[i++];			xors[ 3] = make_uint2(m151,z[13].y);
		xors[ 4] = c_x[i++];			xors[ 5] = c_x[i++];			xors[ 6] = c_x[i++];			xors[ 7] = make_uint2(m131,z[15].y);
		
		xors[ 8] = make_uint2(z[14].x,m100);	xors[ 9] = c_x[i++];			xors[10] = c_x[i++];			xors[11] = m[ 9]^z[ 4];
		xors[12] = z[10];			xors[13] = c_x[i++];			xors[14] = c_x[i++];			xors[15] = c_x[i++];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);

		//13:{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 }
		xors[ 0] = c_x[i++];			xors[ 1] = c_x[i++];			xors[ 2] = make_uint2(m132,z[12].y);	xors[ 3] = z[14];
		xors[ 4] = m[ 9]^z[ 7];			xors[ 5] = c_x[i++];			xors[ 6] = z[13];			xors[ 7] = c_x[i++];
		
		xors[ 8] = c_x[i++];			xors[ 9] = c_x[i++];			xors[10] = c_x[i++];			xors[11] = make_uint2(m152,z[ 8].y);
		xors[12] = c_x[i++];			xors[13] = make_uint2(z[ 5].x,m101);	xors[14] = c_x[i++];			xors[15] = c_x[i++];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);

		//14:{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 }
		xors[ 0] = m[ 9]^z[ 0];			xors[ 1] = c_x[i++];			xors[ 2] = c_x[i++];			xors[ 3] = make_uint2(z[15].x,m102);
		xors[ 4] = c_x[i++];			xors[ 5] = c_x[i++];			xors[ 6] = c_x[i++];			xors[ 7] = make_uint2(m153,z[10].y);
		
		xors[ 8] = z[ 1];			xors[ 9] = z[12];			xors[10] = c_x[i++];			xors[11] = c_x[i++];
		xors[12] = c_x[i++];			xors[13] = z[11];			xors[14] = c_x[i++];			xors[15] = make_uint2(m133,z[ 3].y);
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);
		//15:{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 }
		xors[ 0] = c_x[i++];			xors[ 1] = c_x[i++];			xors[ 2] = c_x[i++];			xors[ 3] = c_x[i++];
		xors[ 4] = z[ 2];			xors[ 5] = make_uint2(z[ 6].x,m103);	xors[ 6] = z[ 0];			xors[ 7] = c_x[i++];
		
		xors[ 8] = c_x[i++];			xors[ 9] = c_x[i++];			xors[10] = make_uint2(m154,z[14].y);	xors[11] = c_x[i++];
		xors[12] = make_uint2(m134,z[ 4].y);	xors[13] = c_x[i++];			xors[14] = z[15];			xors[15] = m[ 9]^z[ 1];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);

		v[0] = cuda_swab64_U2(xor3x(v[0],h[0],v[ 8]));
		v[1] = cuda_swab64_U2(xor3x(v[1],h[1],v[ 9]));
		v[2] = cuda_swab64_U2(xor3x(v[2],h[2],v[10]));
		v[3] = cuda_swab64_U2(xor3x(v[3],h[3],v[11]));
		v[4] = cuda_swab64_U2(xor3x(v[4],h[4],v[12]));
		v[5] = cuda_swab64_U2(xor3x(v[5],h[5],v[13]));
		v[6] = cuda_swab64_U2(xor3x(v[6],h[6],v[14]));
		v[7] = cuda_swab64_U2(xor3x(v[7],h[7],v[15]));

		uint64_t msg[8];
		uint64_t msg0[16] = { 0 }, msg1[16] = { 0 };
		uint64_t q[32];
		
#pragma unroll 8
		for(int i=0;i<8;i++)msg[i]=devectorize(v[i]);

#pragma unroll 8
		for(int i = 0; i < 8; ++i) msg0[i] = (msg[i]);

		msg1[0] = 0x80UL;
		msg1[15] = 1024UL;
		BMW_Compression(msg0, BMW512_IV, q);
		BMW_Compression(msg1, msg0,q);
		BMW_Compression(msg1, BMW512_FINAL,q);

		uint2x4* outpt = &g_hash[thread<<1];
		outpt[0] = *(uint2x4*)&msg1[8];
		outpt[1] = *(uint2x4*)&msg1[12];
	}
}

__host__
void quark_blake512_cpu_hash_80_bmw_128(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash){
	dim3 grid((threads + TPB80-1)/TPB80);
	dim3 block(TPB80);

	quark_blake512_gpu_hash_80_bmw_128<<<grid, block>>>(threads, startNounce, (uint2x4*)d_outputHash);
}