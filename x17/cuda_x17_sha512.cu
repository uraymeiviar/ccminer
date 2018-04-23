/*
 * sha-512 cuda kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014 djm34
 *               2016 tpruvot
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 */
#include "cuda_helper_alexis.h"
#include "miner.h"
#include "cuda_vectors_alexis.h"

#define SWAP64(u64) cuda_swab64(u64)

static __constant__ uint64_t c_WB[80] = {
	0x428A2F98D728AE22, 0x7137449123EF65CD, 0xB5C0FBCFEC4D3B2F, 0xE9B5DBA58189DBBC,
	0x3956C25BF348B538, 0x59F111F1B605D019, 0x923F82A4AF194F9B, 0xAB1C5ED5DA6D8118,
	0xD807AA98A3030242, 0x12835B0145706FBE, 0x243185BE4EE4B28C, 0x550C7DC3D5FFB4E2,
	0x72BE5D74F27B896F, 0x80DEB1FE3B1696B1, 0x9BDC06A725C71235, 0xC19BF174CF692694,
	0xE49B69C19EF14AD2, 0xEFBE4786384F25E3, 0x0FC19DC68B8CD5B5, 0x240CA1CC77AC9C65,
	0x2DE92C6F592B0275, 0x4A7484AA6EA6E483, 0x5CB0A9DCBD41FBD4, 0x76F988DA831153B5,
	0x983E5152EE66DFAB, 0xA831C66D2DB43210, 0xB00327C898FB213F, 0xBF597FC7BEEF0EE4,
	0xC6E00BF33DA88FC2, 0xD5A79147930AA725, 0x06CA6351E003826F, 0x142929670A0E6E70,
	0x27B70A8546D22FFC, 0x2E1B21385C26C926, 0x4D2C6DFC5AC42AED, 0x53380D139D95B3DF,
	0x650A73548BAF63DE, 0x766A0ABB3C77B2A8, 0x81C2C92E47EDAEE6, 0x92722C851482353B,
	0xA2BFE8A14CF10364, 0xA81A664BBC423001, 0xC24B8B70D0F89791, 0xC76C51A30654BE30,
	0xD192E819D6EF5218, 0xD69906245565A910, 0xF40E35855771202A, 0x106AA07032BBD1B8,
	0x19A4C116B8D2D0C8, 0x1E376C085141AB53, 0x2748774CDF8EEB99, 0x34B0BCB5E19B48A8,
	0x391C0CB3C5C95A63, 0x4ED8AA4AE3418ACB, 0x5B9CCA4F7763E373, 0x682E6FF3D6B2B8A3,
	0x748F82EE5DEFB2FC, 0x78A5636F43172F60, 0x84C87814A1F0AB72, 0x8CC702081A6439EC,
	0x90BEFFFA23631E28, 0xA4506CEBDE82BDE9, 0xBEF9A3F7B2C67915, 0xC67178F2E372532B,
	0xCA273ECEEA26619C, 0xD186B8C721C0C207, 0xEADA7DD6CDE0EB1E, 0xF57D4F7FEE6ED178,
	0x06F067AA72176FBA, 0x0A637DC5A2C898A6, 0x113F9804BEF90DAE, 0x1B710B35131C471B,
	0x28DB77F523047D84, 0x32CAAB7B40C72493, 0x3C9EBE0A15C9BEBC, 0x431D67C49C100D4C,
	0x4CC5D4BECB3E42B6, 0x597F299CFC657E2A, 0x5FCB6FAB3AD6FAEC, 0x6C44198C4A475817
};

#define bsg5_0(x) xor3(ROTR64(x,28),ROTR64(x,34),ROTR64(x,39))
#define bsg5_1(x) xor3(ROTR64(x,14),ROTR64(x,18),ROTR64(x,41))
#define ssg5_0(x) xor3(ROTR64(x,1),ROTR64(x,8),shr_u64(x,7))
#define ssg5_1(x) xor3(ROTR64(x,19),ROTR64(x,61),shr_u64(x,6))

#define BSG5_1 bsg5_1
#define BSG5_0 bsg5_0
#define SSG5_1 ssg5_1
#define SSG5_0 ssg5_0

#define MAJ(X, Y, Z)   (((X) & (Y)) | (((X) | (Y)) & (Z)))
//#define MAJ(x, y, z)   andor(x,y,z)

__device__ __forceinline__
uint64_t Tone(const uint64_t* K, uint64_t* r, uint64_t* W, const uint8_t a, const uint8_t i)
{
	//asm("// TONE \n");
	const uint64_t e = r[(a+4) & 7];
	const uint64_t BSG51 = xor3(ROTR64(e, 14), ROTR64(e, 18), ROTR64(e, 41));
	const uint64_t f = r[(a+5) & 7];
	const uint64_t g = r[(a+6) & 7];
	const uint64_t CHl = ((f ^ g) & e) ^ g; // xandx(e, f, g);
	return (r[(a+7) & 7] + W[i] + BSG51 + CHl + K[i]);
}

#define SHA3_STEP(K, r, W, ord, i) { \
	const int a = (8 - ord) & 7; \
	uint64_t T1 = Tone(K, r, W, a, i); \
	r[(a+3) & 7]+= T1; \
	r[(a+7) & 7] = T1 + (BSG5_0(r[a]) + MAJ(r[a], r[(a+1) & 7], r[(a+2) & 7])); \
}

__global__
__launch_bounds__(256, 4)
void x17_sha512_gpu_hash_64(const uint32_t threads, uint64_t *g_hash){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint64_t IV512[8] = {
		0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
		0x510E527FADE682D1, 0x9B05688C2B3E6C1F, 0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
	};
	uint64_t r[8];
	uint64_t W[80];
	if (thread < threads){

		uint64_t *pHash = &g_hash[thread<<3];

		*(uint2x4*)&W[ 0] = *(uint2x4*)&pHash[ 0];
		*(uint2x4*)&W[ 4] = *(uint2x4*)&pHash[ 4];

		#pragma unroll
		for (int i = 0; i < 8; i ++) {
			W[i] = cuda_swab64(W[i]);
		}
		W[8] = 0x8000000000000000;

		#pragma unroll
		for (int i = 9; i<15; i++) {
			W[i] = 0U;
		}
		W[15] = 0x0000000000000200;

		#pragma unroll 64
		for (int i = 16; i < 80; i++) {
			W[i] = W[i-7] + W[i-16] + SSG5_0(W[i-15]) + SSG5_1(W[i-2]);
		}

		*(uint2x4*)&r[ 0] = *(uint2x4*)&IV512[0];
		*(uint2x4*)&r[ 4] = *(uint2x4*)&IV512[4];

		#pragma unroll 80
		for (int i = 0; i < 80; i ++){
			SHA3_STEP(c_WB, r, W, i&7, i);
		}
		
		#pragma unroll
		for (int u = 0; u < 8; u ++) {
			r[u] = cuda_swab64(r[u] + IV512[u]);
		}

		*(uint2x4*)&pHash[ 0] = *(uint2x4*)&r[ 0];
		*(uint2x4*)&pHash[ 4] = *(uint2x4*)&r[ 4];

	}
}

__host__
void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x17_sha512_gpu_hash_64 <<<grid, block>>> (threads, (uint64_t*)d_hash);

}


__constant__
static uint64_t c_PaddedMessage80[10];

__global__
/*__launch_bounds__(256, 4)*/
void x16_sha512_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint64_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t W[80];
		#pragma unroll
		for (int i = 0; i < 9; i ++) {
			W[i] = SWAP64(c_PaddedMessage80[i]);
		}
		const uint32_t nonce = startNonce + thread;
		//((uint32_t*)W)[19] = cuda_swab32(nonce);
		W[9] = REPLACE_HIDWORD(c_PaddedMessage80[9], cuda_swab32(nonce));
		W[9] = cuda_swab64(W[9]);
		W[10] = 0x8000000000000000;

		#pragma unroll
		for (int i = 11; i<15; i++) {
			W[i] = 0U;
		}
		W[15] = 0x0000000000000280;

		#pragma unroll 64
		for (int i = 16; i < 80; i ++) {
			W[i] = SSG5_1(W[i-2]) + W[i-7];
			W[i] += SSG5_0(W[i-15]) + W[i-16];
		}

		const uint64_t IV512[8] = {
			0x6A09E667F3BCC908, 0xBB67AE8584CAA73B,
			0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
			0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
			0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
		};

		uint64_t r[8];
		#pragma unroll
		for (int i = 0; i < 8; i++) {
			r[i] = IV512[i];
		}

		#pragma unroll
		for (int i = 0; i < 80; i++) {
			SHA3_STEP(c_WB, r, W, i&7, i);
		}

		const uint64_t hashPosition = thread;
		uint64_t *pHash = &g_hash[hashPosition << 3];
		#pragma unroll
		for (int u = 0; u < 8; u ++) {
			pHash[u] = SWAP64(r[u] + IV512[u]);
		}
	}
}

__host__
void x16_sha512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x16_sha512_gpu_hash_80 <<<grid, block >>> (threads, startNounce, (uint64_t*)d_hash);
}

__host__
void x16_sha512_setBlock_80(void *pdata)
{
	cudaMemcpyToSymbol(c_PaddedMessage80, pdata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
}

//=========================================================

static const __constant__ __align__(16) ulong K512[80] =
{
	0x428A2F98D728AE22UL, 0x7137449123EF65CDUL,
	0xB5C0FBCFEC4D3B2FUL, 0xE9B5DBA58189DBBCUL,
	0x3956C25BF348B538UL, 0x59F111F1B605D019UL,
	0x923F82A4AF194F9BUL, 0xAB1C5ED5DA6D8118UL,
	0xD807AA98A3030242UL, 0x12835B0145706FBEUL,
	0x243185BE4EE4B28CUL, 0x550C7DC3D5FFB4E2UL,
	0x72BE5D74F27B896FUL, 0x80DEB1FE3B1696B1UL,
	0x9BDC06A725C71235UL, 0xC19BF174CF692694UL,
	0xE49B69C19EF14AD2UL, 0xEFBE4786384F25E3UL,
	0x0FC19DC68B8CD5B5UL, 0x240CA1CC77AC9C65UL,
	0x2DE92C6F592B0275UL, 0x4A7484AA6EA6E483UL,
	0x5CB0A9DCBD41FBD4UL, 0x76F988DA831153B5UL,
	0x983E5152EE66DFABUL, 0xA831C66D2DB43210UL,
	0xB00327C898FB213FUL, 0xBF597FC7BEEF0EE4UL,
	0xC6E00BF33DA88FC2UL, 0xD5A79147930AA725UL,
	0x06CA6351E003826FUL, 0x142929670A0E6E70UL,
	0x27B70A8546D22FFCUL, 0x2E1B21385C26C926UL,
	0x4D2C6DFC5AC42AEDUL, 0x53380D139D95B3DFUL,
	0x650A73548BAF63DEUL, 0x766A0ABB3C77B2A8UL,
	0x81C2C92E47EDAEE6UL, 0x92722C851482353BUL,
	0xA2BFE8A14CF10364UL, 0xA81A664BBC423001UL,
	0xC24B8B70D0F89791UL, 0xC76C51A30654BE30UL,
	0xD192E819D6EF5218UL, 0xD69906245565A910UL,
	0xF40E35855771202AUL, 0x106AA07032BBD1B8UL,
	0x19A4C116B8D2D0C8UL, 0x1E376C085141AB53UL,
	0x2748774CDF8EEB99UL, 0x34B0BCB5E19B48A8UL,
	0x391C0CB3C5C95A63UL, 0x4ED8AA4AE3418ACBUL,
	0x5B9CCA4F7763E373UL, 0x682E6FF3D6B2B8A3UL,
	0x748F82EE5DEFB2FCUL, 0x78A5636F43172F60UL,
	0x84C87814A1F0AB72UL, 0x8CC702081A6439ECUL,
	0x90BEFFFA23631E28UL, 0xA4506CEBDE82BDE9UL,
	0xBEF9A3F7B2C67915UL, 0xC67178F2E372532BUL,
	0xCA273ECEEA26619CUL, 0xD186B8C721C0C207UL,
	0xEADA7DD6CDE0EB1EUL, 0xF57D4F7FEE6ED178UL,
	0x06F067AA72176FBAUL, 0x0A637DC5A2C898A6UL,
	0x113F9804BEF90DAEUL, 0x1B710B35131C471BUL,
	0x28DB77F523047D84UL, 0x32CAAB7B40C72493UL,
	0x3C9EBE0A15C9BEBCUL, 0x431D67C49C100D4CUL,
	0x4CC5D4BECB3E42B6UL, 0x597F299CFC657E2AUL,
	0x5FCB6FAB3AD6FAECUL, 0x6C44198C4A475817UL
};

#define andor64(a,b,c) ((a & (b | c)) | (b & c))
#define xandx64(e,f,g) (g ^ (e & (g ^ f)))
#define CH  xandx64
#define sph_u64 uint64_t

__device__ __forceinline__ void SHA2_512_STEP2(const ulong *W, uint ord, ulong *r, int i, const ulong *k)
{
	ulong T1;
	int x = 8 - ord;
	
	ulong a = r[x & 7], b = r[(x + 1) & 7], c = r[(x + 2) & 7], d = r[(x + 3) & 7];
	ulong e = r[(x + 4) & 7], f = r[(x + 5) & 7], g = r[(x + 6) & 7], h = r[(x + 7) & 7];
	
	T1 = h + BSG5_1(e) + CH(e, f, g) + W[i] + (K512[i]);
	r[(3 + x) & 7] = d + T1;
	r[(7 + x) & 7] = T1 + BSG5_0(a) + MAJ(a, b, c);
}

__device__ __forceinline__ void SHA512Block(ulong *W, ulong *buf, const ulong *k)
{
	ulong  r[8];

	for(int i = 0; i < 8; ++i){
		r[i] = buf[i];
	}
	
	#pragma unroll 16
	for(int i = 16; i < 80; ++i){
		W[i] = SSG5_1(W[i - 2]) + W[i - 7] + SSG5_0(W[i - 15]) + W[i - 16];
	}
	
	for(int i = 0; i < 80; i += 8){
		#pragma unroll 8
		for(int j = 0; j < 8; ++j){
			SHA2_512_STEP2(W, j, r, i + j, k);
		}
	}

	for(int i = 0; i < 8; ++i) buf[i] += r[i];
}

#define TPB_SHA 176
#define TH_SH 76
__global__ __launch_bounds__(TPB_SHA,1)
__global__ void sha512_gpu_hash_128(int threads,  uint64_t *g_hash)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint64_t *inpHash = &g_hash[thread<<3];

		__shared__ sph_u64 Ws[TH_SH*80];
		sph_u64 Wm[80];
		sph_u64 *W;

		if(threadIdx.x < TH_SH)
			W=&Ws[80*threadIdx.x];
		else
			W=&Wm[0];

		sph_u64 state[8];

		#pragma unroll 8
		for (int i = 0; i < 8; i++){
			W[i] = cuda_swab64(__ldg(&inpHash[i]));
		}

		#pragma unroll 8
		for (int i = 8; i < 16; i++){
			W[i] = 0;
		}

		state[0] = SPH_C64(0x6A09E667F3BCC908);
		state[1] = SPH_C64(0xBB67AE8584CAA73B);
		state[2] = SPH_C64(0x3C6EF372FE94F82B);
		state[3] = SPH_C64(0xA54FF53A5F1D36F1);
		state[4] = SPH_C64(0x510E527FADE682D1);
		state[5] = SPH_C64(0x9B05688C2B3E6C1F);
		state[6] = SPH_C64(0x1F83D9ABFB41BD6B);
		state[7] = SPH_C64(0x5BE0CD19137E2179);

		SHA512Block(W, state,K512);

		W[0] = 0x8000000000000000UL;
		W[1] = 0x0000000000000000UL;
		W[2] = 0x0000000000000000UL;
		W[3] = 0x0000000000000000UL;
		W[4] = 0x0000000000000000UL;
		W[5] = 0x0000000000000000UL;
		W[6] = 0x0000000000000000UL;
		W[7] = 0x0000000000000000UL;
		W[8] = 0x0000000000000000UL;
		W[9] = 0x0000000000000000UL;
		W[10] = 0x0000000000000000UL;
		W[11] = 0x0000000000000000UL;
		W[12] = 0x0000000000000000UL;
		W[13] = 0x0000000000000000UL;
		W[14] = 0x0000000000000000UL;
		W[15] = 0x0000000000000400UL;

		SHA512Block(W, state,K512);

		#pragma unroll 8
		for (int i = 0; i < 8; i++){
			inpHash[i] = cuda_swab64(state[i]);
		}
	}
}

__host__ void x17_sha512_cpu_hash_128(int thr_id, int threads, uint32_t *d_hash)
{
	const int threadsperblock = TPB_SHA;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	sha512_gpu_hash_128<<<grid, block>>>(threads,  (uint64_t*)d_hash);
}
