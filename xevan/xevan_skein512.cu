#include "cuda_helper.h"
#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

#define TPB52 512
#define TPB50 512

#define M9_0_0    0
#define M9_0_1    1
#define M9_0_2    2
#define M9_0_3    3
#define M9_0_4    4
#define M9_0_5    5
#define M9_0_6    6
#define M9_0_7    7

#define M9_1_0    1
#define M9_1_1    2
#define M9_1_2    3
#define M9_1_3    4
#define M9_1_4    5
#define M9_1_5    6
#define M9_1_6    7
#define M9_1_7    8

#define M9_2_0    2
#define M9_2_1    3
#define M9_2_2    4
#define M9_2_3    5
#define M9_2_4    6
#define M9_2_5    7
#define M9_2_6    8
#define M9_2_7    0

#define M9_3_0    3
#define M9_3_1    4
#define M9_3_2    5
#define M9_3_3    6
#define M9_3_4    7
#define M9_3_5    8
#define M9_3_6    0
#define M9_3_7    1

#define M9_4_0    4
#define M9_4_1    5
#define M9_4_2    6
#define M9_4_3    7
#define M9_4_4    8
#define M9_4_5    0
#define M9_4_6    1
#define M9_4_7    2

#define M9_5_0    5
#define M9_5_1    6
#define M9_5_2    7
#define M9_5_3    8
#define M9_5_4    0
#define M9_5_5    1
#define M9_5_6    2
#define M9_5_7    3

#define M9_6_0    6
#define M9_6_1    7
#define M9_6_2    8
#define M9_6_3    0
#define M9_6_4    1
#define M9_6_5    2
#define M9_6_6    3
#define M9_6_7    4

#define M9_7_0    7
#define M9_7_1    8
#define M9_7_2    0
#define M9_7_3    1
#define M9_7_4    2
#define M9_7_5    3
#define M9_7_6    4
#define M9_7_7    5

#define M9_8_0    8
#define M9_8_1    0
#define M9_8_2    1
#define M9_8_3    2
#define M9_8_4    3
#define M9_8_5    4
#define M9_8_6    5
#define M9_8_7    6

#define M9_9_0    0
#define M9_9_1    1
#define M9_9_2    2
#define M9_9_3    3
#define M9_9_4    4
#define M9_9_5    5
#define M9_9_6    6
#define M9_9_7    7

#define M9_10_0   1
#define M9_10_1   2
#define M9_10_2   3
#define M9_10_3   4
#define M9_10_4   5
#define M9_10_5   6
#define M9_10_6   7
#define M9_10_7   8

#define M9_11_0   2
#define M9_11_1   3
#define M9_11_2   4
#define M9_11_3   5
#define M9_11_4   6
#define M9_11_5   7
#define M9_11_6   8
#define M9_11_7   0

#define M9_12_0   3
#define M9_12_1   4
#define M9_12_2   5
#define M9_12_3   6
#define M9_12_4   7
#define M9_12_5   8
#define M9_12_6   0
#define M9_12_7   1

#define M9_13_0   4
#define M9_13_1   5
#define M9_13_2   6
#define M9_13_3   7
#define M9_13_4   8
#define M9_13_5   0
#define M9_13_6   1
#define M9_13_7   2

#define M9_14_0   5
#define M9_14_1   6
#define M9_14_2   7
#define M9_14_3   8
#define M9_14_4   0
#define M9_14_5   1
#define M9_14_6   2
#define M9_14_7   3

#define M9_15_0   6
#define M9_15_1   7
#define M9_15_2   8
#define M9_15_3   0
#define M9_15_4   1
#define M9_15_5   2
#define M9_15_6   3
#define M9_15_7   4

#define M9_16_0   7
#define M9_16_1   8
#define M9_16_2   0
#define M9_16_3   1
#define M9_16_4   2
#define M9_16_5   3
#define M9_16_6   4
#define M9_16_7   5

#define M9_17_0   8
#define M9_17_1   0
#define M9_17_2   1
#define M9_17_3   2
#define M9_17_4   3
#define M9_17_5   4
#define M9_17_6   5
#define M9_17_7   6

#define M9_18_0   0
#define M9_18_1   1
#define M9_18_2   2
#define M9_18_3   3
#define M9_18_4   4
#define M9_18_5   5
#define M9_18_6   6
#define M9_18_7   7

/*
 * M3_ ## s ## _ ## i  evaluates to s+i mod 3 (0 <= s <= 18, 0 <= i <= 1).
 */

#define M3_0_0    0
#define M3_0_1    1
#define M3_1_0    1
#define M3_1_1    2
#define M3_2_0    2
#define M3_2_1    0
#define M3_3_0    0
#define M3_3_1    1
#define M3_4_0    1
#define M3_4_1    2
#define M3_5_0    2
#define M3_5_1    0
#define M3_6_0    0
#define M3_6_1    1
#define M3_7_0    1
#define M3_7_1    2
#define M3_8_0    2
#define M3_8_1    0
#define M3_9_0    0
#define M3_9_1    1
#define M3_10_0   1
#define M3_10_1   2
#define M3_11_0   2
#define M3_11_1   0
#define M3_12_0   0
#define M3_12_1   1
#define M3_13_0   1
#define M3_13_1   2
#define M3_14_0   2
#define M3_14_1   0
#define M3_15_0   0
#define M3_15_1   1
#define M3_16_0   1
#define M3_16_1   2
#define M3_17_0   2
#define M3_17_1   0
#define M3_18_0   0
#define M3_18_1   1

#define XCAT(x, y)     XCAT_(x, y)
#define XCAT_(x, y)    x ## y

#define SKBI(k, s, i)   XCAT(k, XCAT(XCAT(XCAT(M9_, s), _), i))
#define SKBT(t, s, v)   XCAT(t, XCAT(XCAT(XCAT(M3_, s), _), v))

#define TFBIG_ADDKEY_UI2(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
    w0 = (w0 + SKBI(k, s, 0)); \
    w1 = (w1 + SKBI(k, s, 1)); \
    w2 = (w2 + SKBI(k, s, 2)); \
    w3 = (w3 + SKBI(k, s, 3)); \
    w4 = (w4 + SKBI(k, s, 4)); \
    w5 = (w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
    w6 = (w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
    w7 = (w7 + SKBI(k, s, 7) + vectorize(s)); \
}

#define TFBIG_MIX_UI2(x0, x1, rc) { \
    x0 = x0 + x1; \
    x1 = ROL2(x1, rc) ^ x0; \
}

#define TFBIG_MIX8_UI2(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
    TFBIG_MIX_UI2(w0, w1, rc0); \
    TFBIG_MIX_UI2(w2, w3, rc1); \
    TFBIG_MIX_UI2(w4, w5, rc2); \
    TFBIG_MIX_UI2(w6, w7, rc3); \
}

#define TFBIG_4o_UI2(s)  { \
    TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
    TFBIG_MIX8_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
    TFBIG_MIX8_UI2(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
    TFBIG_MIX8_UI2(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
    TFBIG_MIX8_UI2(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
}

#define TFBIG_4e_UI2(s)  { \
    TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
    TFBIG_MIX8_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
    TFBIG_MIX8_UI2(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
    TFBIG_MIX8_UI2(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
    TFBIG_MIX8_UI2(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPB52, 3)
#else
__launch_bounds__(TPB50, 3)
#endif
void xevan_skein512_gpu_hash_128(const uint32_t threads,uint64_t* g_hash, const uint32_t* g_nonceVector){
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads){
// Skein
		uint2 p[8], h[9], m[8];

		const uint32_t hashPosition = (g_nonceVector == NULL) ? thread : g_nonceVector[thread];

		uint64_t *Hash = &g_hash[hashPosition << 3];

		uint2x4 *phash = (uint2x4*)Hash;
		*(uint2x4*)&m[0] = __ldg4(&phash[0]);
		*(uint2x4*)&m[4] = __ldg4(&phash[1]);
		
		#pragma unroll 8
		for(int i = 0; i < 8; i++){
			p[i] = m[i];
		}
		
		h[0] = vectorize((uint64_t)0x4903ADFF749C51CE);
		h[1] = vectorize((uint64_t)0x0D95DE399746DF03);
		h[2] = vectorize((uint64_t)0x8FD1934127C79BCE);
		h[3] = vectorize((uint64_t)0x9A255629FF352CB1);
		h[4] = vectorize((uint64_t)0x5DB62599DF6CA7B0);
		h[5] = vectorize((uint64_t)0xEABE394CA9D5C3F4);
		h[6] = vectorize((uint64_t)0x991112C71A75B523);
		h[7] = vectorize((uint64_t)0xAE18A40B660FCC33);

		h[8] = h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7] ^ vectorize(0x1BD11BDAA9FC1A22);

		uint2 t0,t1,t2;
		t0 = vectorize(((uint64_t)1 << 6) + (uint64_t)0);
		t1 = vectorize(((uint64_t)1 >> 58) + ((uint64_t)224 << 55));
		t2 = t1 ^ t0;

		#define h0  h[0]
		#define h1  h[1]
		#define h2  h[2]
		#define h3  h[3]
		#define h4  h[4]
		#define h5  h[5]
		#define h6  h[6]
		#define h7  h[7]
		#define h8  h[8]

		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);


		h[0] = m[0] ^ p[0];
		h[1] = m[1] ^ p[1];
		h[2] = m[2] ^ p[2];
		h[3] = m[3] ^ p[3];
		h[4] = m[4] ^ p[4];
		h[5] = m[5] ^ p[5];
		h[6] = m[6] ^ p[6];
		h[7] = m[7] ^ p[7];

		#pragma unroll 8
		for(int i = 0; i < 8; i++){
			p[i] = vectorize((uint64_t)0);
		}

//352,64
		h[8] = h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7] ^ vectorize(0x1BD11BDAA9FC1A22);

		t0 = vectorize(((uint64_t)1 << 6) + (uint64_t)64);
		t1 = vectorize(((uint64_t)1 >> 58) + ((uint64_t)352 << 55));
		t2 = t1 ^ t0;

		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		#pragma unroll 8
		for(int i = 0; i < 8; i++){
			h[i] = p[i];
		}

		///510,8
		#pragma unroll 8
		for(int i = 0; i < 8; i++){
			p[i] = vectorize((uint64_t)0);
		}

		h[8] = h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7] ^ vectorize(0x1BD11BDAA9FC1A22);

		t0 = vectorize(((uint64_t)0 << 6) + (uint64_t)8);
		t1 = vectorize(((uint64_t)0 >> 58) + ((uint64_t)510 << 55));
		t2 = t1 ^ t0;

		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		phash = (uint2x4*)p;
		uint2x4 *outpt = (uint2x4*)Hash;
		outpt[0] = phash[0];
		outpt[1] = phash[1];
	}
}

__host__
void xevan_skein512_cpu_hash_128(int thr_id,uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash)
{
	uint32_t tpb = TPB52;
	int dev_id = device_map[thr_id];
	
	if (device_sm[dev_id] <= 500) 
		tpb = TPB50;
	const dim3 grid((threads + tpb - 1) / tpb);
	const dim3 block(tpb);
	xevan_skein512_gpu_hash_128 <<<grid, block>>>(threads, (uint64_t*)d_hash, d_nonceVector);
}