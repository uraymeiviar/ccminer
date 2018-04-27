#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

#define sph_u32 uint32_t
#define sph_s32 int32_t

#define TPB50_1 128
#define TPB50_2 128
#define TPB52_1 128
#define TPB52_2 128

static uint4 *d_temp4[MAX_GPUS];

typedef sph_u32 u32;
typedef sph_s32 s32;
#define C32     SPH_C32
#define T32     SPH_T32
#define ROL32   ROTL32
#define XCAT_(x, y)   x ## y
#define XCAT(x, y)    XCAT_(x, y)
  
__constant__ static const s32 SIMD_Q_128[] = {
    6, 110, -59, -30, 45, 48, -17, -95, 28, -90, -95, 26, 100, -121, -82, 13, 105, 29, 76, -101, 65, -56, 12, -56, 15, 98, 1, 79, -128, -1, -8, 61, -52, 87, 89, -69, -39, -23, -35, 16, 8, 18, -118, -96, -69, -105, 117, -102, 128, -69, -2, 28, 91, -13, 83, -57, 53, 68, -82, 17, -97, 80, -46, -41, 64, -115, 32, 39, -7, -27,-72, -17, 2, -122, 4, 52, 93, -86, 62, 66, 122, -88, -95, 57, 34, 120, 35, -16, 17, -86, 7, 54, -103, -119, 45, -123, -69, 88, 126, 118, -99, -117, 4, -75, -112, -25, 35, -85, -3, -61, 31, 3, -12, 43, 90, 31, 48, 46, 68, 79, -43, 32, 35, 98, -102, -95, 14, 33, -6, -110, 59, 30, -45, -48, 17, 95, -28, 90, 95, -26, -100, 121, 82, -13, -105, -29, -76, 101, -65, 56, -12, 56, -15, -98, -1, -79, 128, 1, 8, -61, 52, -87, -89, 69, 39, 23, 35, -16, -8, -18, 118, 96, 69, 105, -117, 102, -128, 69, 2, -28, -91, 13, -83, 57, -53, -68, 82, -17, 97, -80, 46, 41, -64, 115, -32, -39, 7, 27, 72, 17, -2, 122, -4, -52, -93, 86, -62, -66, -122, 88, 95,-57, -34, -120, -35, 16, -17, 86, -7, -54, 103, 119, -45, 123, 69, -88, -126, -118, 99, 117, -4, 75, 112, 25, -35, 85, 3, 61, -31, -3, 12, -43, -90, -31, -48, -46, -68, -79, 43, -32, -35, -98, 102, 95, -14, -33
  };

static __constant__ const uint8_t c_perm[8][8] = {
	{ 2, 3, 6, 7, 0, 1, 4, 5 },{ 6, 7, 2, 3, 4, 5, 0, 1 },{ 7, 6, 5, 4, 3, 2, 1, 0 },{ 1, 0, 3, 2, 5, 4, 7, 6 },
	{ 0, 1, 4, 5, 6, 7, 2, 3 },{ 6, 7, 2, 3, 0, 1, 4, 5 },{ 6, 7, 0, 1, 4, 5, 2, 3 },{ 4, 5, 2, 3, 6, 7, 0, 1 }
};

static __constant__ const uint32_t c_IV_512[32] = {
	0x0ba16b95, 0x72f999ad, 0x9fecc2ae, 0xba3264fc, 0x5e894929, 0x8e9f30e5, 0x2f1daa37, 0xf0f2c558,
	0xac506643, 0xa90635a5, 0xe25b878b, 0xaab7878f, 0x88817f7a, 0x0a02892b, 0x559a7550, 0x598f657e,
	0x7eef60a1, 0x6b70e3e8, 0x9c1714d1, 0xb958e2a8, 0xab02675e, 0xed1c014f, 0xcd8d65bb, 0xfdb7a257,
	0x09254899, 0xd699c7bc, 0x9019b6dc, 0x2b9022e4, 0x8fa14956, 0x21bf9bd3, 0xb94d0943, 0x6ffddc22
};

static __constant__ const int16_t c_FFT128_8_16_Twiddle[128] = {
	1,   1,   1,   1,   1,    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,  60,   2, 120,   4,  -17,   8, -34,  16, -68,  32, 121,  64, -15, 128, -30,
	1,  46,  60, -67,   2,   92, 120, 123,   4, -73, -17, -11,   8, 111, -34, -22, 1, -67, 120, -73,   8,  -22, -68, -70,  64,  81, -30, -46,  -2,-123,  17,-111,
	1,-118,  46, -31,  60,  116, -67, -61,   2,  21,  92, -62, 120, -25, 123,-122, 1, 116,  92,-122, -17,   84, -22,  18,  32, 114, 117, -49, -30, 118,  67,  62,
	1, -31, -67,  21, 120, -122, -73, -50,   8,   9, -22, -89, -68,  52, -70, 114, 1, -61, 123, -50, -34,   18, -70, -99, 128, -98,  67,  25,  17,  -9,  35, -79
};

static __constant__ const int16_t c_FFT256_2_128_Twiddle[128] = {
	  1,  41,-118,  45,  46,  87, -31,  14, 60,-110, 116,-127, -67,  80, -61,  69,  2,  82,  21,  90,  92, -83, -62,  28,120,  37, -25,   3, 123, -97,-122,-119,
	  4, -93,  42, -77, -73,  91,-124,  56,-17,  74, -50,   6, -11,  63,  13,  19,  8,  71,  84, 103, 111, -75,   9, 112,-34,-109,-100,  12, -22, 126,  26,  38,
	 16,-115, -89, -51, -35, 107,  18, -33,-68,  39,  57,  24, -44,  -5,  52,  76, 32,  27,  79,-102, -70, -43,  36, -66,121,  78, 114,  48, -88, -10, 104,-105,
	 64,  54, -99,  53, 117, -86,  72, 125,-15,-101, -29,  96,  81, -20, -49,  47,128, 108,  59, 106, -23,  85,-113,  -7,-30,  55, -58, -65, -95, -40, -98,  94
};

#define INNER(l, h, mm)   (((u32)((l) * (mm)) & 0xFFFFU) \
              + ((u32)((h) * (mm)) << 16))

#define W_BIG(sb, o1, o2, mm) \
  (INNER(q[16 * (sb) + 2 * 0 + o1], q[16 * (sb) + 2 * 0 + o2], mm), \
   INNER(q[16 * (sb) + 2 * 1 + o1], q[16 * (sb) + 2 * 1 + o2], mm), \
   INNER(q[16 * (sb) + 2 * 2 + o1], q[16 * (sb) + 2 * 2 + o2], mm), \
   INNER(q[16 * (sb) + 2 * 3 + o1], q[16 * (sb) + 2 * 3 + o2], mm), \
   INNER(q[16 * (sb) + 2 * 4 + o1], q[16 * (sb) + 2 * 4 + o2], mm), \
   INNER(q[16 * (sb) + 2 * 5 + o1], q[16 * (sb) + 2 * 5 + o2], mm), \
   INNER(q[16 * (sb) + 2 * 6 + o1], q[16 * (sb) + 2 * 6 + o2], mm), \
   INNER(q[16 * (sb) + 2 * 7 + o1], q[16 * (sb) + 2 * 7 + o2], mm)

#define WB_0_0   W_BIG( 4,    0,    1, 185)
#define WB_0_1   W_BIG( 6,    0,    1, 185)
#define WB_0_2   W_BIG( 0,    0,    1, 185)
#define WB_0_3   W_BIG( 2,    0,    1, 185)
#define WB_0_4   W_BIG( 7,    0,    1, 185)
#define WB_0_5   W_BIG( 5,    0,    1, 185)
#define WB_0_6   W_BIG( 3,    0,    1, 185)
#define WB_0_7   W_BIG( 1,    0,    1, 185)
#define WB_1_0   W_BIG(15,    0,    1, 185)
#define WB_1_1   W_BIG(11,    0,    1, 185)
#define WB_1_2   W_BIG(12,    0,    1, 185)
#define WB_1_3   W_BIG( 8,    0,    1, 185)
#define WB_1_4   W_BIG( 9,    0,    1, 185)
#define WB_1_5   W_BIG(13,    0,    1, 185)
#define WB_1_6   W_BIG(10,    0,    1, 185)
#define WB_1_7   W_BIG(14,    0,    1, 185)
#define WB_2_0   W_BIG(17, -256, -128, 233)
#define WB_2_1   W_BIG(18, -256, -128, 233)
#define WB_2_2   W_BIG(23, -256, -128, 233)
#define WB_2_3   W_BIG(20, -256, -128, 233)
#define WB_2_4   W_BIG(22, -256, -128, 233)
#define WB_2_5   W_BIG(21, -256, -128, 233)
#define WB_2_6   W_BIG(16, -256, -128, 233)
#define WB_2_7   W_BIG(19, -256, -128, 233)
#define WB_3_0   W_BIG(30, -383, -255, 233)
#define WB_3_1   W_BIG(24, -383, -255, 233)
#define WB_3_2   W_BIG(25, -383, -255, 233)
#define WB_3_3   W_BIG(31, -383, -255, 233)
#define WB_3_4   W_BIG(27, -383, -255, 233)
#define WB_3_5   W_BIG(29, -383, -255, 233)
#define WB_3_6   W_BIG(28, -383, -255, 233)
#define WB_3_7   W_BIG(26, -383, -255, 233)

#define IF(x, y, z)    ((((y) ^ (z)) & (x)) ^ (z))
#define MAJ(x, y, z)   (((x) & (y)) | (((x) | (y)) & (z)))

#define PP4_0_0   1
#define PP4_0_1   0
#define PP4_0_2   3
#define PP4_0_3   2
#define PP4_1_0   2
#define PP4_1_1   3
#define PP4_1_2   0
#define PP4_1_3   1
#define PP4_2_0   3
#define PP4_2_1   2
#define PP4_2_2   1
#define PP4_2_3   0

#define PP8_0_0   1
#define PP8_0_1   0
#define PP8_0_2   3
#define PP8_0_3   2
#define PP8_0_4   5
#define PP8_0_5   4
#define PP8_0_6   7
#define PP8_0_7   6

#define PP8_1_0   6
#define PP8_1_1   7
#define PP8_1_2   4
#define PP8_1_3   5
#define PP8_1_4   2
#define PP8_1_5   3
#define PP8_1_6   0
#define PP8_1_7   1

#define PP8_2_0   2
#define PP8_2_1   3
#define PP8_2_2   0
#define PP8_2_3   1
#define PP8_2_4   6
#define PP8_2_5   7
#define PP8_2_6   4
#define PP8_2_7   5

#define PP8_3_0   3
#define PP8_3_1   2
#define PP8_3_2   1
#define PP8_3_3   0
#define PP8_3_4   7
#define PP8_3_5   6
#define PP8_3_6   5
#define PP8_3_7   4

#define PP8_4_0   5
#define PP8_4_1   4
#define PP8_4_2   7
#define PP8_4_3   6
#define PP8_4_4   1
#define PP8_4_5   0
#define PP8_4_6   3
#define PP8_4_7   2

#define PP8_5_0   7
#define PP8_5_1   6
#define PP8_5_2   5
#define PP8_5_3   4
#define PP8_5_4   3
#define PP8_5_5   2
#define PP8_5_6   1
#define PP8_5_7   0

#define PP8_6_0   4
#define PP8_6_1   5
#define PP8_6_2   6
#define PP8_6_3   7
#define PP8_6_4   0
#define PP8_6_5   1
#define PP8_6_6   2
#define PP8_6_7   3

#define M7_0_0   0_
#define M7_1_0   1_
#define M7_2_0   2_
#define M7_3_0   3_
#define M7_4_0   4_
#define M7_5_0   5_
#define M7_6_0   6_
#define M7_7_0   0_

#define M7_0_1   1_
#define M7_1_1   2_
#define M7_2_1   3_
#define M7_3_1   4_
#define M7_4_1   5_
#define M7_5_1   6_
#define M7_6_1   0_
#define M7_7_1   1_

#define M7_0_2   2_
#define M7_1_2   3_
#define M7_2_2   4_
#define M7_3_2   5_
#define M7_4_2   6_
#define M7_5_2   0_
#define M7_6_2   1_
#define M7_7_2   2_

#define M7_0_3   3_
#define M7_1_3   4_
#define M7_2_3   5_
#define M7_3_3   6_
#define M7_4_3   0_
#define M7_5_3   1_
#define M7_6_3   2_
#define M7_7_3   3_

#define STEP_ELT(n, w, fun, s, ppb)   do { \
    u32 tt = T32(D ## n + (w) + fun(A ## n, B ## n, C ## n)); \
    A ## n = T32(ROL32(tt, s) + XCAT(tA, XCAT(ppb, n))); \
    D ## n = C ## n; \
    C ## n = B ## n; \
    B ## n = tA ## n; \
  } while (0)

#define STEP_BIG(w0, w1, w2, w3, w4, w5, w6, w7, fun, r, s, pp8b)   do { \
    u32 tA0 = ROL32(A0, r); \
    u32 tA1 = ROL32(A1, r); \
    u32 tA2 = ROL32(A2, r); \
    u32 tA3 = ROL32(A3, r); \
    u32 tA4 = ROL32(A4, r); \
    u32 tA5 = ROL32(A5, r); \
    u32 tA6 = ROL32(A6, r); \
    u32 tA7 = ROL32(A7, r); \
    STEP_ELT(0, w0, fun, s, pp8b); \
    STEP_ELT(1, w1, fun, s, pp8b); \
    STEP_ELT(2, w2, fun, s, pp8b); \
    STEP_ELT(3, w3, fun, s, pp8b); \
    STEP_ELT(4, w4, fun, s, pp8b); \
    STEP_ELT(5, w5, fun, s, pp8b); \
    STEP_ELT(6, w6, fun, s, pp8b); \
    STEP_ELT(7, w7, fun, s, pp8b); \
  } while (0)

#define STEP_BIG_(w, fun, r, s, pp8b)   STEP_BIG w, fun, r, s, pp8b)

#define ONE_ROUND_BIG(ri, isp, p0, p1, p2, p3)   do { \
    STEP_BIG_(WB_ ## ri ## 0, \
      IF,  p0, p1, XCAT(PP8_, M7_0_ ## isp)); \
    STEP_BIG_(WB_ ## ri ## 1, \
      IF,  p1, p2, XCAT(PP8_, M7_1_ ## isp)); \
    STEP_BIG_(WB_ ## ri ## 2, \
      IF,  p2, p3, XCAT(PP8_, M7_2_ ## isp)); \
    STEP_BIG_(WB_ ## ri ## 3, \
      IF,  p3, p0, XCAT(PP8_, M7_3_ ## isp)); \
    STEP_BIG_(WB_ ## ri ## 4, \
      MAJ, p0, p1, XCAT(PP8_, M7_4_ ## isp)); \
    STEP_BIG_(WB_ ## ri ## 5, \
      MAJ, p1, p2, XCAT(PP8_, M7_5_ ## isp)); \
    STEP_BIG_(WB_ ## ri ## 6, \
      MAJ, p2, p3, XCAT(PP8_, M7_6_ ## isp)); \
    STEP_BIG_(WB_ ## ri ## 7, \
      MAJ, p3, p0, XCAT(PP8_, M7_7_ ## isp)); \
  } while (0)

#define p8_xor(x) ( ((x)%7) == 0 ? 1 : \
	((x)%7) == 1 ? 6 : \
	((x)%7) == 2 ? 2 : \
	((x)%7) == 3 ? 3 : \
	((x)%7) == 4 ? 5 : \
	((x)%7) == 5 ? 7 : 4 )

__device__ __forceinline__
static void STEP8_IF(const uint32_t *w, const uint32_t i, const uint32_t r, const uint32_t s, uint32_t *A, const uint32_t *B, const uint32_t *C, uint32_t *D)
{
	uint32_t R[8];

	#pragma unroll 8
	for(int j=0; j<8; j++)
		R[j] = ROTL32(A[j], r);

	uint32_t W[8];
	*(uint2x4*)&W[0] = *(uint2x4*)&w[0];
	#pragma unroll 8
	for(int j=0; j<8; j++)
		D[j]+= W[j] + IF(A[j], B[j], C[j]);
	#pragma unroll 8
	for(int j=0; j<8; j++)
		D[j] = R[j^p8_xor(i)] + ROTL32(D[j], s);
	#pragma unroll 8
	for(int j=0; j<8; j++)
		A[j] = R[j];
}

__device__ __forceinline__
static void STEP8_MAJ(const uint32_t *w, const uint32_t i, const uint32_t r, const uint32_t s, uint32_t *A, const uint32_t *B, const uint32_t *C, uint32_t *D)
{
	uint32_t R[8];

	uint32_t W[8];
	*(uint2x4*)&W[0] = *(uint2x4*)&w[0];
	
	#pragma unroll 8
	for(int j=0; j<8; j++)
		R[j] = ROTL32(A[j], r);

	#pragma unroll 8
	for(int j=0; j<8; j++)
		D[j]+= W[j] + MAJ(A[j], B[j], C[j]);
	#pragma unroll 8
	for(int j=0; j<8; j++)
		D[j] = R[j^p8_xor(i)] + ROTL32(D[j], s);
	#pragma unroll 8
	for(int j=0; j<8; j++)
		A[j] = R[j];
}


//#define expanded_vector(x) __ldg(&g_fft4[x])
static __device__ __forceinline__ void expanded_vector(uint32_t* w,const uint4* ptr){
	asm volatile ("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(w[0]), "=r"(w[1]),"=r"(w[2]), "=r"(w[3]) : __LDG_PTR(ptr));
}

__device__ __forceinline__
static void Round8(uint32_t* A, const uint32_t thr_offset, const uint4 *const __restrict__ g_fft4) {

	uint32_t w[8];
	uint32_t tmp = thr_offset;

	uint32_t r = 3, s = 23, t = 17, u = 27;
	
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,0, r, s, A, &A[8], &A[16], &A[24]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,1, s, t, &A[24], A, &A[8], &A[16]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,2, t, u, &A[16], &A[24], A, &A[8]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,3, u, r, &A[8], &A[16], &A[24], A);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,4, r, s, A, &A[8], &A[16], &A[24]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,5, s, t, &A[24], A, &A[8], &A[16]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,6, t, u, &A[16], &A[24], A, &A[8]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,7, u, r, &A[8], &A[16], &A[24], A);

	r = 28; s = 19; t = 22; u = 7;
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,8, r, s, A, &A[8], &A[16], &A[24]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,9, s, t, &A[24], A, &A[8], &A[16]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,10, t, u, &A[16], &A[24], A, &A[8]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,11, u, r, &A[8], &A[16], &A[24], A);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,12, r, s, A, &A[8], &A[16], &A[24]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,13, s, t, &A[24], A, &A[8], &A[16]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,14, t, u, &A[16], &A[24], A, &A[8]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,15, u, r, &A[8], &A[16], &A[24], A);

	r = 29; s = 9; t = 15; u = 5;
	
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,16, r, s, A, &A[8], &A[16], &A[24]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,17, s, t, &A[24], A, &A[8], &A[16]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,18, t, u, &A[16], &A[24], A, &A[8]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,19, u, r, &A[8], &A[16], &A[24], A);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,20, r, s, A, &A[8], &A[16], &A[24]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,21, s, t, &A[24], A, &A[8], &A[16]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,22, t, u, &A[16], &A[24], A, &A[8]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,23, u, r, &A[8], &A[16], &A[24], A);

	r =  4; s = 13; t = 10; u = 25;

 	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,24, r, s, A, &A[8], &A[16], &A[24]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,25, s, t, &A[24], A, &A[8], &A[16]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,26, t, u, &A[16], &A[24], A, &A[8]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,27, u, r, &A[8], &A[16], &A[24], A);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,28, r, s, A, &A[8], &A[16], &A[24]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,29, s, t, &A[24], A, &A[8], &A[16]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,30, t, u, &A[16], &A[24], A, &A[8]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,31, u, r, &A[8], &A[16], &A[24], A);

}

/********************* Message expansion ************************/

/*
 * Reduce modulo 257; result is in [-127; 383]
 * REDUCE(x) := (x&255) - (x>>8)
 */
 #define REDUCE(x) \
 (((x)&255) - ((x)>>8))

/*
* Reduce from [-127; 383] to [-128; 128]
* EXTRA_REDUCE_S(x) := x<=128 ? x : x-257
*/
#define EXTRA_REDUCE_S(x) \
 ((x)<=128 ? (x) : (x)-257)

/*
* Reduce modulo 257; result is in [-128; 128]
*/
#define REDUCE_FULL_S(x) \
 EXTRA_REDUCE_S(REDUCE(x))

// Parallelization:
//
// FFT_8  wird 2 times 8-fach parallel ausgeführt (in FFT_64)
//        and  1 time 16-fach parallel (in FFT_128_full)
//
// STEP8_IF and STEP8_MAJ beinhalten je 2x 8-fach parallel Operations

/**
* FFT_8 using w=4 as 8th root of unity
* Unrolled decimation in frequency (DIF) radix-2 NTT.
* Output data is in revbin_permuted order.
*/
__device__ __forceinline__
static void FFT_8(int *y,const uint8_t stripe){

#define BUTTERFLY(i,j,n) \
do { \
 int u= y[stripe*i]; \
 int v= y[stripe*j]; \
 y[stripe*i] = u+v; \
 y[stripe*j] = (u-v) << (n<<1); \
} while(0)

 BUTTERFLY(0, 4, 0);
 BUTTERFLY(1, 5, 1);
 BUTTERFLY(2, 6, 2);
 BUTTERFLY(3, 7, 3);

 y[stripe*6] = REDUCE(y[stripe*6]);
 y[stripe*7] = REDUCE(y[stripe*7]);

 BUTTERFLY(0, 2, 0);
 BUTTERFLY(4, 6, 0);
 BUTTERFLY(1, 3, 2);
 BUTTERFLY(5, 7, 2);

 y[stripe*7] = REDUCE(y[stripe*7]);

 BUTTERFLY(0, 1, 0);
 BUTTERFLY(2, 3, 0);
 BUTTERFLY(4, 5, 0);
 BUTTERFLY(6, 7, 0);

 y[ 0] = REDUCE(y[ 0]);
 y[stripe] = REDUCE(y[stripe]);
 y[stripe<<1] = REDUCE(y[stripe<<1]);
 y[stripe*3] = REDUCE(y[stripe*3]);
 y[stripe<<2] = REDUCE(y[stripe<<2]);
 y[stripe*5] = REDUCE(y[stripe*5]);
 y[stripe*6] = REDUCE(y[stripe*6]);
 y[stripe*7] = REDUCE(y[stripe*7]);
 
 y[ 0] = EXTRA_REDUCE_S(y[ 0]);
 y[stripe] = EXTRA_REDUCE_S(y[stripe]);
 y[stripe<<1] = EXTRA_REDUCE_S(y[stripe<<1]);
 y[stripe*3] = EXTRA_REDUCE_S(y[stripe*3]);
 y[stripe<<2] = EXTRA_REDUCE_S(y[stripe<<2]);
 y[stripe*5] = EXTRA_REDUCE_S(y[stripe*5]);
 y[stripe*6] = EXTRA_REDUCE_S(y[stripe*6]);
 y[stripe*7] = EXTRA_REDUCE_S(y[stripe*7]);

#undef BUTTERFLY
}

/**
* FFT_16 using w=2 as 16th root of unity
* Unrolled decimation in frequency (DIF) radix-2 NTT.
* Output data is in revbin_permuted order.
*/
__device__ __forceinline__
static void FFT_16(int *y){

#define DO_REDUCE_FULL_S(i) \
 do { \
     y[i] = REDUCE(y[i]); \
     y[i] = EXTRA_REDUCE_S(y[i]); \
 } while(0)

 int u,v;

 const uint8_t thr = threadIdx.x&7;

 u = y[0]; // 0..7
 v = y[1]; // 8..15
 y[0] = u+v;
 y[1] = (u-v) << (thr);

 if ((thr) >=3) y[1] = REDUCE(y[1]);  // 11...15

 u = __shfl(y[0],  (threadIdx.x&3),8); // 0,1,2,3  0,1,2,3
 v = __shfl(y[0],4+(threadIdx.x&3),8); // 4,5,6,7  4,5,6,7
 y[0] = ((thr) < 4) ? (u+v) : ((u-v) << ((threadIdx.x&3)<<1));

 u = __shfl(y[1],  (threadIdx.x&3),8); // 8,9,10,11    8,9,10,11
 v = __shfl(y[1],4+(threadIdx.x&3),8); // 12,13,14,15  12,13,14,15
 y[1] = ((thr) < 4) ? (u+v) : ((u-v) << ((threadIdx.x&3)<<1));

 if ((threadIdx.x&1) && (thr >= 4)) {
     y[0] = REDUCE(y[0]);  // 5, 7
     y[1] = REDUCE(y[1]);  // 13, 15
 }

 u = __shfl(y[0],  (threadIdx.x&5),8); // 0,1,0,1  4,5,4,5
 v = __shfl(y[0],2+(threadIdx.x&5),8); // 2,3,2,3  6,7,6,7
 y[0] = ((threadIdx.x&3) < 2) ? (u+v) : ((u-v) << ((threadIdx.x&1)<<2));

 u = __shfl(y[1],  (threadIdx.x&5),8); // 8,9,8,9      12,13,12,13
 v = __shfl(y[1],2+(threadIdx.x&5),8); // 10,11,10,11  14,15,14,15
 y[1] = ((threadIdx.x&3) < 2) ? (u+v) : ((u-v) << ((threadIdx.x&1)<<2));

 u = __shfl(y[0],  (threadIdx.x&6),8); // 0,0,2,2      4,4,6,6
 v = __shfl(y[0],1+(threadIdx.x&6),8); // 1,1,3,3      5,5,7,7
 y[0] = ((threadIdx.x&1) < 1) ? (u+v) : (u-v);

 u = __shfl(y[1],  (threadIdx.x&6),8); // 8,8,10,10    12,12,14,14
 v = __shfl(y[1],1+(threadIdx.x&6),8); // 9,9,11,11    13,13,15,15
 y[1] = ((threadIdx.x&1) < 1) ? (u+v) : (u-v);

 DO_REDUCE_FULL_S( 0); // 0...7
 DO_REDUCE_FULL_S( 1); // 8...15

#undef DO_REDUCE_FULL_S
}

//=========================================================

#if __CUDA_ARCH__ > 500
__global__ __launch_bounds__(TPB52_1,2)
#else
__global__ __launch_bounds__(TPB50_1,2)
#endif
static void xevan_simd512_gpu_expand_128(uint32_t threads,const uint32_t* __restrict__ g_hash, uint4 *g_temp4)
{
	const uint32_t threadBloc = (blockDim.x * blockIdx.x + threadIdx.x)>>3;
	const uint8_t thr        = (threadIdx.x & 7);
	/* Message Expansion using Number Theoretical Transform similar to FFT */
	int expanded[32];

	uint4 vec0;
	int P, Q, P1, Q1, P2, Q2;

	const bool even = (threadIdx.x & 1) == 0;
	const bool hi = (thr)>=4;
	const bool lo = (thr)<4;
	const bool sel = ((threadIdx.x+2)&7) >= 4;  // 2,3,4,5
	
	if (threadBloc < threads){
		
		const uint32_t hashPosition = threadBloc<<4;

		const uint32_t *inpHash = &g_hash[hashPosition];

		const uint32_t data0 = __ldg(&inpHash[thr]);
		const uint32_t data1 = __ldg(&inpHash[thr + 8]);

		// Puffer für expandierte Nachricht
		uint4 *temp4 = &g_temp4[hashPosition<<2];

		#pragma unroll 4
		for (uint32_t i=0; i < 4; i++) {
			expanded[  i] = bfe(__byte_perm(__shfl(data0, i<<1, 8), __shfl(data0, (i<<1)+1, 8), thr),0,8);
		}
		#pragma unroll 4
		for (uint32_t i=0; i < 4; i++) {			
			expanded[4+i] = bfe(__byte_perm(__shfl(data1, i<<1, 8), __shfl(data1, (i<<1)+1, 8), thr),0,8);
		}
		#pragma unroll 8
		for (uint32_t i=8; i < 16; i++) {			
			expanded[ i] = 0;
		}
		/*
		 * FFT_256 using w=41 as 256th root of unity. Decimation in frequency (DIF) NTT. Output data is in revbin_permuted order. In place.
		 */
		#pragma unroll 8
		for (uint32_t i=0; i<8; i++)
			expanded[16+i] = REDUCE(expanded[i] * c_FFT256_2_128_Twiddle[8*i+(thr)]);

		#pragma unroll 8
		for (uint32_t i=24; i < 32; i++) {			
			expanded[ i] = 0;
		}		
		/* handle X^255 with an additional butterfly */
		if (thr==7){
			expanded[15] = 1;
			expanded[31] = REDUCE((-1) * c_FFT256_2_128_Twiddle[127]);
		}

//		FFT_128_full(expanded);
		FFT_8(expanded,2); // eight parallel FFT8's
		FFT_8(&expanded[16],2); // eight parallel FFT8's
		FFT_8(&expanded[ 1],2); // eight parallel FFT8's
		FFT_8(&expanded[17],2); // eight parallel FFT8's
		
		#pragma unroll 16
		for (uint32_t i=0; i<16; i++){
			expanded[i] = REDUCE(expanded[i]*c_FFT128_8_16_Twiddle[i*8+(thr)]);
			expanded[i+16] = REDUCE(expanded[i+16]*c_FFT128_8_16_Twiddle[i*8+(thr)]);			
		}

		#pragma unroll 8
		for (uint32_t i=0; i<8; i++){
			FFT_16(expanded+(i<<1));  // eight sequential FFT16's, each one executed in parallel by 8 threads
			FFT_16(expanded+16+(i<<1));  // eight sequential FFT16's, each one executed in parallel by 8 threads			
		}

		// store w matrices in global memory
		P1 = expanded[ 0]; P2 = __shfl(expanded[ 2], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
		Q1 = expanded[16]; Q2 = __shfl(expanded[18], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
		vec0.x = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[0][thr], 8);
		P1 = expanded[ 8]; P2 = __shfl(expanded[10], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
		Q1 = expanded[24]; Q2 = __shfl(expanded[26], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
		vec0.y = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[0][thr], 8);
		P1 = expanded[ 4]; P2 = __shfl(expanded[ 6], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
		Q1 = expanded[20]; Q2 = __shfl(expanded[22], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
		vec0.z = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[0][thr], 8);
		P1 = expanded[12]; P2 = __shfl(expanded[14], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
		Q1 = expanded[28]; Q2 = __shfl(expanded[30], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
		vec0.w = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[0][thr], 8);
		temp4[thr] = vec0;

		P1 = expanded[ 1]; P2 = __shfl(expanded[ 3], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
		Q1 = expanded[17]; Q2 = __shfl(expanded[19], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
		vec0.x = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[1][thr], 8);
		P1 = expanded[ 9]; P2 = __shfl(expanded[11], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
		Q1 = expanded[25]; Q2 = __shfl(expanded[27], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
		vec0.y = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[1][thr], 8);
		P1 = expanded[ 5]; P2 = __shfl(expanded[ 7], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
		Q1 = expanded[21]; Q2 = __shfl(expanded[23], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
		vec0.z = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[1][thr], 8);
		P1 = expanded[13]; P2 = __shfl(expanded[15], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
		Q1 = expanded[29]; Q2 = __shfl(expanded[31], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
		vec0.w = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[1][thr], 8);
		temp4[8+(thr)] = vec0;

		P1 = hi?expanded[ 1]:expanded[ 0]; P2 = __shfl(hi?expanded[ 3]:expanded[ 2], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
		Q1 = hi?expanded[17]:expanded[16]; Q2 = __shfl(hi?expanded[19]:expanded[18], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
		vec0.x = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[2][thr], 8);
		P1 = hi?expanded[ 9]:expanded[ 8]; P2 = __shfl(hi?expanded[11]:expanded[10], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
		Q1 = hi?expanded[25]:expanded[24]; Q2 = __shfl(hi?expanded[27]:expanded[26], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
		vec0.y = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[2][thr], 8);
		P1 = hi?expanded[ 5]:expanded[ 4]; P2 = __shfl(hi?expanded[ 7]:expanded[ 6], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
		Q1 = hi?expanded[21]:expanded[20]; Q2 = __shfl(hi?expanded[23]:expanded[22], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
		vec0.z = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[2][thr], 8);
		P1 = hi?expanded[13]:expanded[12]; P2 = __shfl(hi?expanded[15]:expanded[14], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
		Q1 = hi?expanded[29]:expanded[28]; Q2 = __shfl(hi?expanded[31]:expanded[30], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
		vec0.w = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[2][thr], 8);
		temp4[16+(thr)] = vec0;

		P1 = lo?expanded[ 1]:expanded[ 0]; P2 = __shfl(lo?expanded[ 3]:expanded[ 2], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
		Q1 = lo?expanded[17]:expanded[16]; Q2 = __shfl(lo?expanded[19]:expanded[18], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
		vec0.x = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[3][thr], 8);
		P1 = lo?expanded[ 9]:expanded[ 8]; P2 = __shfl(lo?expanded[11]:expanded[10], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
		Q1 = lo?expanded[25]:expanded[24]; Q2 = __shfl(lo?expanded[27]:expanded[26], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
		vec0.y = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[3][thr], 8);
		P1 = lo?expanded[ 5]:expanded[ 4]; P2 = __shfl(lo?expanded[ 7]:expanded[ 6], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
		Q1 = lo?expanded[21]:expanded[20]; Q2 = __shfl(lo?expanded[23]:expanded[22], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
		vec0.z = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[3][thr], 8);
		P1 = lo?expanded[13]:expanded[12]; P2 = __shfl(lo?expanded[15]:expanded[14], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
		Q1 = lo?expanded[29]:expanded[28]; Q2 = __shfl(lo?expanded[31]:expanded[30], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
		vec0.w = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[3][thr], 8);
		temp4[24+(thr)] = vec0;

		P1 = sel?expanded[0]:expanded[1]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		Q2 = sel?expanded[2]:expanded[3]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.x = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[4][thr], 8);
		P1 = sel?expanded[8]:expanded[9]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		Q2 = sel?expanded[10]:expanded[11]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.y = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[4][thr], 8);
		P1 = sel?expanded[4]:expanded[5]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		Q2 = sel?expanded[6]:expanded[7]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.z = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[4][thr], 8);
		P1 = sel?expanded[12]:expanded[13]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		Q2 = sel?expanded[14]:expanded[15]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.w = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[4][thr], 8);

		temp4[32+thr] = vec0;

		P1 = sel?expanded[1]:expanded[0]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		Q2 = sel?expanded[3]:expanded[2]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.x = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[5][thr], 8);
		P1 = sel?expanded[9]:expanded[8]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		Q2 = sel?expanded[11]:expanded[10]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.y = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[5][thr], 8);
		P1 = sel?expanded[5]:expanded[4]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		Q2 = sel?expanded[7]:expanded[6]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.z = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[5][thr], 8);
		P1 = sel?expanded[13]:expanded[12]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		Q2 = sel?expanded[15]:expanded[14]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.w = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[5][thr], 8);

		temp4[40+thr] = vec0;

		uint32_t t;
		t = __shfl(expanded[17],(threadIdx.x+4)&7,8); P1 = sel?t:expanded[16]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		t = __shfl(expanded[19],(threadIdx.x+4)&7,8); Q2 = sel?t:expanded[18]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.x = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[6][thr], 8);
		t = __shfl(expanded[25],(threadIdx.x+4)&7,8); P1 = sel?t:expanded[24]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		t = __shfl(expanded[27],(threadIdx.x+4)&7,8); Q2 = sel?t:expanded[26]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.y = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[6][thr], 8);
		t = __shfl(expanded[21],(threadIdx.x+4)&7,8); P1 = sel?t:expanded[20]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		t = __shfl(expanded[23],(threadIdx.x+4)&7,8); Q2 = sel?t:expanded[22]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.z = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[6][thr], 8);
		t = __shfl(expanded[29],(threadIdx.x+4)&7,8); P1 = sel?t:expanded[28]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		t = __shfl(expanded[31],(threadIdx.x+4)&7,8); Q2 = sel?t:expanded[30]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.w = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[6][thr], 8);

		temp4[48+thr] = vec0;

		t = __shfl(expanded[16],(threadIdx.x+4)&7,8); P1 = sel?expanded[17]:t; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		t = __shfl(expanded[18],(threadIdx.x+4)&7,8); Q2 = sel?expanded[19]:t; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.x = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[7][thr], 8);
		t = __shfl(expanded[24],(threadIdx.x+4)&7,8); P1 = sel?expanded[25]:t; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		t = __shfl(expanded[26],(threadIdx.x+4)&7,8); Q2 = sel?expanded[27]:t; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.y = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[7][thr], 8);
		t = __shfl(expanded[20],(threadIdx.x+4)&7,8); P1 = sel?expanded[21]:t; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		t = __shfl(expanded[22],(threadIdx.x+4)&7,8); Q2 = sel?expanded[23]:t; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.z = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[7][thr], 8);
		t = __shfl(expanded[28],(threadIdx.x+4)&7,8); P1 = sel?expanded[29]:t; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		t = __shfl(expanded[30],(threadIdx.x+4)&7,8); Q2 = sel?expanded[31]:t; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.w = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[7][thr], 8);

		temp4[56+thr] = vec0;
	}
}

__global__ 
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPB52_2,2)
#else
__launch_bounds__(TPB50_2,4)
#endif
static void xevan_simd512_gpu_compress_128_maxwell(uint32_t threads, uint32_t *g_hash,const uint4 *const __restrict__ g_fft4)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t thr_offset = thread << 6; // thr_id * 128 (je zwei elemente)
	uint32_t IV[32];
	if (thread < threads){
		uint32_t *Hash = &g_hash[thread<<4];

		uint32_t A[32];

		*(uint2x4*)&IV[ 0] = *(uint2x4*)&c_IV_512[ 0];
		*(uint2x4*)&IV[ 8] = *(uint2x4*)&c_IV_512[ 8];
		*(uint2x4*)&IV[16] = *(uint2x4*)&c_IV_512[16];
		*(uint2x4*)&IV[24] = *(uint2x4*)&c_IV_512[24];

		*(uint2x4*)&A[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
		*(uint2x4*)&A[ 8] = __ldg4((uint2x4*)&Hash[ 8]);

		#pragma unroll 16
		for(uint32_t i=0;i<16;i++){
            A[ i] = A[ i] ^ IV[ i];
        }

		for(uint32_t i=16;i<32;i++){
			A[ i] = IV[ i];
        }

		Round8(A, thr_offset, g_fft4);
		
		STEP8_IF(&IV[ 0],32, 4,13,&A[ 0],&A[ 8],&A[16],&A[24]);
		STEP8_IF(&IV[ 8],33,13,10,&A[24],&A[ 0],&A[ 8],&A[16]);
		STEP8_IF(&IV[16],34,10,25,&A[16],&A[24],&A[ 0],&A[ 8]);
		STEP8_IF(&IV[24],35,25, 4,&A[ 8],&A[16],&A[24],&A[ 0]);

        u32  A0 = A[0];
        u32  A1 = A[1];
        u32  A2 = A[2];
        u32  A3 = A[3];
        u32  A4 = A[4];
        u32  A5 = A[5];
        u32  A6 = A[6];
        u32  A7 = A[7];
        u32  B0 = A[8];
        u32  B1 = A[9];
        u32  B2 = A[10];
        u32  B3 = A[11];
        u32  B4 = A[12];
        u32  B5 = A[13];
        u32  B6 = A[14];
        u32  B7 = A[15];

        u32  C0 = A[16];
        u32  C1 = A[17];
        u32  C2 = A[18];
        u32  C3 = A[19];
        u32  C4 = A[20];
        u32  C5 = A[21];
        u32  C6 = A[22];
        u32  C7 = A[23];
        u32  D0 = A[24];
        u32  D1 = A[25];
        u32  D2 = A[26];
        u32  D3 = A[27];
        u32  D4 = A[28];
        u32  D5 = A[29];
        u32  D6 = A[30];
        u32  D7 = A[31];

        u32 COPY_A0 = A0, COPY_A1 = A1, COPY_A2 = A2, COPY_A3 = A3, COPY_A4 = A4, COPY_A5 = A5, COPY_A6 = A6, COPY_A7 = A7;
        u32 COPY_B0 = B0, COPY_B1 = B1, COPY_B2 = B2, COPY_B3 = B3, COPY_B4 = B4, COPY_B5 = B5, COPY_B6 = B6, COPY_B7 = B7;
        u32 COPY_C0 = C0, COPY_C1 = C1, COPY_C2 = C2, COPY_C3 = C3, COPY_C4 = C4, COPY_C5 = C5, COPY_C6 = C6, COPY_C7 = C7;
        u32 COPY_D0 = D0, COPY_D1 = D1, COPY_D2 = D2, COPY_D3 = D3, COPY_D4 = D4, COPY_D5 = D5, COPY_D6 = D6, COPY_D7 = D7;


        #define q SIMD_Q_128

        A0 ^= 0x400;

        ONE_ROUND_BIG(0_, 0,  3, 23, 17, 27);
        ONE_ROUND_BIG(1_, 1, 28, 19, 22,  7);
        ONE_ROUND_BIG(2_, 2, 29,  9, 15,  5);
        ONE_ROUND_BIG(3_, 3,  4, 13, 10, 25);

        STEP_BIG(
            COPY_A0, COPY_A1, COPY_A2, COPY_A3,
            COPY_A4, COPY_A5, COPY_A6, COPY_A7,
            IF,  4, 13, PP8_4_);

        STEP_BIG(
            COPY_B0, COPY_B1, COPY_B2, COPY_B3,
            COPY_B4, COPY_B5, COPY_B6, COPY_B7,
            IF, 13, 10, PP8_5_);

        STEP_BIG(
            COPY_C0, COPY_C1, COPY_C2, COPY_C3,
            COPY_C4, COPY_C5, COPY_C6, COPY_C7,
            IF, 10, 25, PP8_6_);

        STEP_BIG(
            COPY_D0, COPY_D1, COPY_D2, COPY_D3,
            COPY_D4, COPY_D5, COPY_D6, COPY_D7,
            IF, 25,  4, PP8_0_);

        #undef q

        A[0] = A0;
        A[1] = A1;
        A[2] = A2;
        A[3] = A3;
        A[4] = A4;
        A[5] = A5;
        A[6] = A6;
        A[7] = A7;
        A[8] = B0;
        A[9] = B1;
        A[10] = B2;
        A[11] = B3;
        A[12] = B4;
        A[13] = B5;
        A[14] = B6;
        A[15] = B7;

        *(uint2x4*)&Hash[ 0] = *(uint2x4*)&A[ 0];
        *(uint2x4*)&Hash[ 8] = *(uint2x4*)&A[ 8];
	}
}

__host__
void xevan_simd512_cpu_init(int thr_id, uint32_t threads){
	cudaMalloc(&d_temp4[thr_id], 64*sizeof(uint4)*threads);
}

__host__
void xevan_simd512_cpu_free(int thr_id){
	cudaFree(d_temp4[thr_id]);
}

__host__
void xevan_simd512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash){

	int dev_id = device_map[thr_id];

	uint32_t tpb = TPB52_1;
	if (device_sm[dev_id] <= 500) tpb = TPB50_1;
	const dim3 grid1((8*threads + tpb - 1) / tpb);
	const dim3 block1(tpb);

	tpb = TPB52_2;
	if (device_sm[dev_id] <= 500) tpb = TPB50_2;
	const dim3 grid2((threads + tpb - 1) / tpb);
	const dim3 block2(tpb);
	
	xevan_simd512_gpu_expand_128 <<<grid1, block1>>> (threads, d_hash, d_temp4[thr_id]);
	xevan_simd512_gpu_compress_128_maxwell <<< grid2, block2 >>> (threads, d_hash, d_temp4[thr_id]);
}
