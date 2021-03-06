#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

__constant__ 
static uint2 keccak_round_constants[24] = {
		{ 0x00000001, 0x00000000 }, { 0x00008082, 0x00000000 },	{ 0x0000808a, 0x80000000 }, { 0x80008000, 0x80000000 },
		{ 0x0000808b, 0x00000000 }, { 0x80000001, 0x00000000 },	{ 0x80008081, 0x80000000 }, { 0x00008009, 0x80000000 },
		{ 0x0000008a, 0x00000000 }, { 0x00000088, 0x00000000 },	{ 0x80008009, 0x00000000 }, { 0x8000000a, 0x00000000 },
		{ 0x8000808b, 0x00000000 }, { 0x0000008b, 0x80000000 },	{ 0x00008089, 0x80000000 }, { 0x00008003, 0x80000000 },
		{ 0x00008002, 0x80000000 }, { 0x00000080, 0x80000000 },	{ 0x0000800a, 0x00000000 }, { 0x8000000a, 0x80000000 },
		{ 0x80008081, 0x80000000 }, { 0x00008080, 0x80000000 },	{ 0x80000001, 0x00000000 }, { 0x80008008, 0x80000000 }
};

/* 1344 bytes */
__constant__ static uint32_t c_E8[42][8] = {
	// Round 0 (Function0)
	0xa2ded572, 0x90d6ab81, 0x67f815df, 0xf6875a4d, 0x0a15847b, 0xc54f9f4e, 0x571523b7, 0x402bd1c3, 0xe03a98ea, 0xb4960266, 0x9cfa455c, 0x8a53bbf2, 0x99d2c503, 0x1a1456b5, 0x9a99b266, 0x31a2db88, // 1
	0x5c5aa303, 0x8019051c, 0xdb0e199a, 0x1d959e84, 0x0ab23f40, 0xadeb336f, 0x1044c187, 0xdccde75e, 0x9213ba10, 0x39812c0a, 0x416bbf02, 0x5078aa37, 0x156578dc, 0xd2bf1a3f, 0xd027bbf7, 0xd3910041, // 3
	0x0d5a2d42, 0x0ba75c18, 0x907eccf6, 0xac442bc7, 0x9c9f62dd, 0xd665dfd1, 0xce97c092, 0x23fcc663,	0x036c6e97, 0xbb03f1ee, 0x1ab8e09e, 0xfa618e5d, 0x7e450521, 0xb29796fd, 0xa8ec6c44, 0x97818394, // 5
	0x37858e4a, 0x8173fe8a, 0x2f3003db, 0x6c69b8f8, 0x2d8d672a, 0x4672c78a, 0x956a9ffb, 0x14427fc0,
	// Round 7 (Function0)
	0x8f15f4c5, 0xb775de52, 0xc45ec7bd, 0xbc88e4ae, 0xa76f4475, 0x1e00b882, 0x80bb118f, 0xf4a3a698, 0x338ff48e, 0x20edf1b6, 0x1563a3a9, 0xfde05a7c, 0x24565faa, 0x5ae9ca36, 0x89f9b7d5, 0x362c4206,
	0x433529ce, 0x591ff5d0, 0x3d98fe4e, 0x86814e6f, 0x74f93a53, 0x81ad9d0e, 0xa74b9a73, 0x9f5ad8af, 0x670605a7, 0x26077447, 0x6a6234ee, 0x3f1080c6, 0xbe280b8b, 0x6f7ea0e0, 0x2717b96e, 0x7b487ec6,
	0xa50a550d, 0x81727686, 0xc0a4f84a, 0xd48d6050, 0x9fe7e391, 0x415a9e7e, 0x9ef18e97, 0x62b0e5f3, 0xec1f9ffc, 0xf594d74f, 0x7a205440, 0xd895fa9d, 0x001ae4e3, 0x117e2e55, 0x84c9f4ce, 0xa554c324,
	0x2872df5b, 0xef7c8905, 0x286efebd, 0x2ed349ee, 0xe27ff578, 0x85937e44, 0xb2c4a50f, 0x7f5928eb,
	// Round 14 (Function0)
	0x37695f70, 0x04771bc7, 0x4a3124b3, 0xe720b951, 0xf128865e, 0xe843fe74, 0x65e4d61d, 0x8a87d423,	0xa3e8297d, 0xfb301b1d, 0xf2947692, 0xe01bdc5b, 0x097acbdd, 0x4f4924da, 0xc1d9309b, 0xbf829cf2,
	0x31bae7a4, 0x32fcae3b, 0xffbf70b4, 0x39d3bb53, 0x0544320d, 0xc1c39f45, 0x48bcf8de, 0xa08b29e0,	0xfd05c9e5, 0x01b771a2, 0x0f09aef7, 0x95ed44e3, 0x12347094, 0x368e3be9, 0x34f19042, 0x4a982f4f,
	0x631d4088, 0xf14abb7e, 0x15f66ca0, 0x30c60ae2, 0x4b44c147, 0xc5b67046, 0xffaf5287, 0xe68c6ecc,	0x56a4d5a4, 0x45ce5773, 0x00ca4fbd, 0xadd16430, 0x4b849dda, 0x68cea6e8, 0xae183ec8, 0x67255c14,
	0xf28cdaa3, 0x20b2601f, 0x16e10ecb, 0x7b846fc2, 0x5806e933, 0x7facced1, 0x9a99949a, 0x1885d1a0,
	// Round 21 (Function0)
	0xa15b5932, 0x67633d9f, 0xd319dd8d, 0xba6b04e4, 0xc01c9a50, 0xab19caf6, 0x46b4a5aa, 0x7eee560b,	0xea79b11f, 0x5aac571d, 0x742128a9, 0x76d35075, 0x35f7bde9, 0xfec2463a, 0xee51363b, 0x01707da3,
	0xafc135f7, 0x15638341, 0x42d8a498, 0xa8db3aea, 0x20eced78, 0x4d3bc3fa, 0x79676b9e, 0x832c8332,	0x1f3b40a7, 0x6c4e3ee7, 0xf347271c, 0xfd4f21d2, 0x34f04059, 0x398dfdb8, 0x9a762db7, 0xef5957dc,
	0x490c9b8d, 0xd0ae3b7d, 0xdaeb492b, 0x84558d7a, 0x49d7a25b, 0xf0e9a5f5, 0x0d70f368, 0x658ef8e4,	0xf4a2b8a0, 0x92946891, 0x533b1036, 0x4f88e856, 0x9e07a80c, 0x555cb05b, 0x5aec3e75, 0x4cbcbaf8,
	0x993bbbe3, 0x28acae64, 0x7b9487f3, 0x6db334dc, 0xd6f4da75, 0x50a5346c, 0x5d1c6b72, 0x71db28b8,
	// Round 28 (Function0)
	0xf2e261f8, 0xf1bcac1c, 0x2a518d10, 0xa23fce43, 0x3364dbe3, 0x3cd1bb67, 0xfc75dd59, 0xb043e802,	0xca5b0a33, 0xc3943b92, 0x75a12988, 0x1e4d790e, 0x4d19347f, 0xd7757479, 0x5c5316b4, 0x3fafeeb6,
	0xf7d4a8ea, 0x5324a326, 0x21391abe, 0xd23c32ba, 0x097ef45c, 0x4a17a344, 0x5127234c, 0xadd5a66d,	0xa63e1db5, 0xa17cf84c, 0x08c9f2af, 0x4d608672, 0x983d5983, 0xcc3ee246, 0x563c6b91, 0xf6c76e08,
	0xb333982f, 0xe8b6f406, 0x5e76bcb1, 0x36d4c1be, 0xa566d62b, 0x1582ee74, 0x2ae6c4ef, 0x6321efbc,	0x0d4ec1fd, 0x1614c17e, 0x69c953f4, 0x16fae006, 0xc45a7da7, 0x3daf907e, 0x26585806, 0x3f9d6328,
	0xe3f2c9d2, 0x16512a74, 0x0cd29b00, 0x9832e0f2, 0x30ceaa5f, 0xd830eb0d, 0x300cd4b7, 0x9af8cee3,
	// Round 35 (Function0)
	0x7b9ec54b, 0x574d239b, 0x9279f1b5, 0x316796e6, 0x6ee651ff, 0xf3a6e6cc, 0xd3688604, 0x05750a17,	0xd98176b1, 0xb3cb2bf4, 0xce6c3213, 0x47154778, 0x8452173c, 0x825446ff, 0x62a205f8, 0x486a9323,
	0x0758df38, 0x442e7031, 0x65655e4e, 0x86ca0bd0, 0x897cfcf2, 0xa20940f0, 0x8e5086fc, 0x4e477830,	0x39eea065, 0x26b29721, 0x8338f7d1, 0x6ff81301, 0x37e95ef7, 0xd1ed44a3, 0xbd3a2ce4, 0xe7de9fef,
	0x15dfa08b, 0x7ceca7d8, 0xd9922576, 0x7eb027ab, 0xf6f7853c, 0xda7d8d53, 0xbe42dc12, 0xdea83eaa,	0x93ce25aa, 0xdaef5fc0, 0xd86902bd, 0xa5194a17, 0xfd43f65a, 0x33664d97, 0xf908731a, 0x6a21fd4c,
	0x3198b435, 0xa163d09a, 0x701541db, 0x72409751, 0xbb0f1eea, 0xbf9d75f6, 0x9b54cded, 0xe26f4791
};

__device__ __forceinline__ void SWAP1(uint32_t &x){
	const uint32_t con = 0x55555555 , z = ((x & con) << 1);

	x>>=1;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm ("lop3.b32 %0, %0, %1, %2, 0xEA;" : "+r"(x)	: "r"(con),"r"(z));	// 0xEA = (F0 & CC) | AA
	#else
		x = (x & con) | z;
	#endif
}
__device__ __forceinline__ void SWAP2(uint32_t &x){
	const uint32_t con = 0x33333333, z = ((x & con) << 2);
	x>>=2;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm ("lop3.b32 %0, %0, %1, %2, 0xEA;" : "+r"(x)	: "r"(con),"r"(z));	// 0xEA = (F0 & CC) | AA
	#else
		x = (x & con) | z;
	#endif
}
__device__ __forceinline__ void SWAP4(uint32_t &x){
	const uint32_t con = 0x0F0F0F0F, z = ((x & con) << 4);
	x>>=4;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm ("lop3.b32 %0, %0, %1, %2, 0xEA;" : "+r"(x)	: "r"(con),"r"(z));	// 0xEA = (F0 & CC) | AA
	#else
		x = (x & con) | z;
	#endif
}

//swapping bits 32i||32i+1||......||32i+15 with bits 32i+16||32i+17||......||32i+31 of 32-bit x
#define SWAP16(x) (x) = __byte_perm(x, x, 0x1032);

//swapping bits 16i||16i+1||......||16i+7  with bits 16i+8||16i+9||......||16i+15 of 32-bit x
#define SWAP8(x) (x) = __byte_perm(x, x, 0x2301);

__device__ __forceinline__
void Sbox(uint32_t &m0,uint32_t &m1,uint32_t &m2,uint32_t &m3,const uint32_t cc){
	uint32_t temp;
	
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm ("lop3.b32 %0, %0, %1, %2, 0xD2;" : "+r"(m0)	: "r"(m2),"r"(cc));		// 0xD2 = F0 ^ ((~CC)&AA)
		asm ("lop3.b32 %0, %1, %2, %3, 0x78;" : "=r"(temp)	: "r"(cc),"r"(m0),"r"(m1));	// 0x78 = F0 ^ (CC&AA)
		asm ("lop3.b32 %0, %0, %1, %2, 0xB4;" : "+r"(m0)	: "r"(m2),"r"(m3));		// 0xB4 = F0 ^ (CC&(~AA))
		asm ("lop3.b32 %0, %0, %1, %2, 0x2D;" : "+r"(m3)	: "r"(m1),"r"(m2));		// 0x2D = (~F0) ^ ((~CC)&AA)
		asm ("lop3.b32 %0, %0, %1, %2, 0x78;" : "+r"(m1)	: "r"(m0),"r"(m2));		// 0x78 = F0 ^ (CC&AA)
		asm ("lop3.b32 %0, %0, %1, %2, 0xB4;" : "+r"(m2)	: "r"(m0),"r"(m3));		// 0xB4 = F0 ^ (CC&(~AA))
		asm ("lop3.b32 %0, %0, %1, %2, 0x1E;" : "+r"(m0)	: "r"(m1),"r"(m3));		// 0x1E = F0 ^ (CC|AA)
		asm ("lop3.b32 %0, %0, %1, %2, 0x78;" : "+r"(m3)	: "r"(m1),"r"(m2));		// 0x78 = F0 ^ (CC&AA)
		asm ("lop3.b32 %0, %0, %1, %2, 0x78;" : "+r"(m1)	: "r"(temp),"r"(m0));		// 0x78 = F0 ^ (CC&AA)
	#else
		m0 = m0 ^ (~(m2)) & cc;
		temp = cc ^ (m0 & m1);
		m0 = m0 ^ m2 & (~m3);
		m3 = (~m3) ^ (~(m1)) & m2;
		m1 = m1 ^ (m0 & m2);
		m2 = m2 ^ (m0 & (~(m3)));
		m0 = m0 ^ (m1 | m3);
		m3 = m3 ^ (m1 & m2);
		m1 = m1 ^ (temp & m0);
	#endif
	m2^= temp;
}

__device__ __forceinline__
static void L(uint32_t &m0,uint32_t &m1,uint32_t &m2,uint32_t &m3,uint32_t &m4,uint32_t &m5,uint32_t &m6,uint32_t &m7){
	m4 = m4 ^ m1;
	m5 = m5 ^ m2;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm ("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(m6)	: "r"(m0),"r"(m3));
	#else
		m6 ^= m0 ^ m3;
	#endif
	m7 = m7 ^ m0;
	m0 = m0 ^ m5;
	m1 = m1 ^ m6;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm ("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(m2)	: "r"(m4),"r"(m7));
	#else
		m2 ^= m4 ^ m7;
	#endif
	m3 = m3 ^ m4;
}

__device__ __forceinline__
static void RoundFunction0(uint32_t x[8][4]){
	#pragma unroll 4
	for (int j = 1; j < 8; j += 2) { // 1, 3, 5, 7 (Even)
		//SWAP1x4(x[j]);
		SWAP1(x[j][0]); SWAP1(x[j][1]); SWAP1(x[j][2]); SWAP1(x[j][3]);
	}
}

__device__ __forceinline__
static void RoundFunction1(uint32_t x[8][4]){
	#pragma unroll 4
	for (int j = 1; j < 8; j += 2) {
		//SWAP2x4(x[j]);
		SWAP2(x[j][0]); SWAP2(x[j][1]); SWAP2(x[j][2]); SWAP2(x[j][3]);
	}
}

__device__ __forceinline__
static void RoundFunction2(uint32_t x[8][4]){
	#pragma unroll 4
	for (int j = 1; j < 8; j += 2) {
		//SWAP4x4(x[j]);
		SWAP4(x[j][0]); SWAP4(x[j][1]); SWAP4(x[j][2]); SWAP4(x[j][3]);
	}
}

__device__ __forceinline__
static void RoundFunction3(uint32_t x[8][4])
{
	//uint32_t* xj = x[j];
	#pragma unroll 4
	for (int j = 1; j < 8; j += 2) {
		SWAP8(x[j][0]);
		SWAP8(x[j][1]);
		SWAP8(x[j][2]);
		SWAP8(x[j][3]);
	}
}

__device__ __forceinline__
static void RoundFunction4(uint32_t x[8][4])
{
	#pragma unroll 4
	for (int j = 1; j < 8; j += 2)
	{
		//uint32_t* xj = x[j];
		#pragma unroll
		for (int i = 0; i < 4; i++)
			SWAP16(x[j][i]);
	}
}

__device__ __forceinline__
static void RoundFunction5(uint32_t x[8][4])
{
	#pragma unroll 4
	for (int j = 1; j < 8; j += 2)
	{
		xchg(x[j][0], x[j][1]);
		xchg(x[j][2], x[j][3]);
	}
}

__device__ __forceinline__
static void RoundFunction6(uint32_t x[8][4])
{
	#pragma unroll 4
	for (int j = 1; j < 8; j += 2)
	{
		xchg(x[j][0], x[j][2]);
		xchg(x[j][1], x[j][3]);
	}
}
__device__ __forceinline__
static void Sbox_and_MDS_layer(uint32_t x[8][4], const uint32_t rnd)
{
	// Sbox and MDS layer
	uint2* cc = (uint2*) &c_E8[rnd];
	#pragma unroll 
	for (int i = 0; i < 4; i++, ++cc) {
		uint2 temp = *cc;
		Sbox(x[0][i], x[2][i], x[4][i], x[6][i], temp.x);
		Sbox(x[1][i], x[3][i], x[5][i], x[7][i], temp.y);
		L(x[0][i], x[2][i], x[4][i], x[6][i], x[1][i], x[3][i], x[5][i], x[7][i]);
	}
}

__device__
static void E8(uint32_t x[8][4])
{
	/* perform 6 loops of 7 rounds */
	for (int r = 0; r < 42; r += 7){
		Sbox_and_MDS_layer(x,r);
		RoundFunction0(x);
		Sbox_and_MDS_layer(x,r+1);
		RoundFunction1(x);
		Sbox_and_MDS_layer(x,r+2);
		RoundFunction2(x);
		Sbox_and_MDS_layer(x,r+3);
		RoundFunction3(x);
		Sbox_and_MDS_layer(x,r+4);
		RoundFunction4(x);
		Sbox_and_MDS_layer(x,r+5);
		RoundFunction5(x);
		Sbox_and_MDS_layer(x,r+6);
		RoundFunction6(x);
	}
}

__device__ __forceinline__
static void keccak_kernel(uint2* s){
	uint2 u[5],t[5], v, w;

	/*theta*/
	t[ 0] = vectorize(devectorize(s[ 0])^devectorize(s[ 5]));
	t[ 1] = vectorize(devectorize(s[ 1])^devectorize(s[ 6]));
	t[ 2] = vectorize(devectorize(s[ 2])^devectorize(s[ 7]));
	t[ 3] = vectorize(devectorize(s[ 3])^devectorize(s[ 8]));
	t[ 4] = s[4];

	for (int i = 0; i < 24; i++) {
		/*theta*/
		#pragma unroll 5
		for(int j=0;j<5;j++){
			t[ j] = vectorize(xor5(devectorize(s[ j]),devectorize(s[j+5]),devectorize(s[j+10]),devectorize(s[j+15]),devectorize(s[j+20])));
		}
		/*theta*/
		#pragma unroll 5
		for(int j=0;j<5;j++){
			u[ j] = ROL2(t[ j], 1);
		}
		s[ 4] = xor3x(s[ 4], t[3], u[ 0]);s[ 9] = xor3x(s[ 9], t[3], u[ 0]);s[14] = xor3x(s[14], t[3], u[ 0]);s[19] = xor3x(s[19], t[3], u[ 0]);s[24] = xor3x(s[24], t[3], u[ 0]);
		s[ 0] = xor3x(s[ 0], t[4], u[ 1]);s[ 5] = xor3x(s[ 5], t[4], u[ 1]);s[10] = xor3x(s[10], t[4], u[ 1]);s[15] = xor3x(s[15], t[4], u[ 1]);s[20] = xor3x(s[20], t[4], u[ 1]);
		s[ 1] = xor3x(s[ 1], t[0], u[ 2]);s[ 6] = xor3x(s[ 6], t[0], u[ 2]);s[11] = xor3x(s[11], t[0], u[ 2]);s[16] = xor3x(s[16], t[0], u[ 2]);s[21] = xor3x(s[21], t[0], u[ 2]);
		s[ 2] = xor3x(s[ 2], t[1], u[ 3]);s[ 7] = xor3x(s[ 7], t[1], u[ 3]);s[12] = xor3x(s[12], t[1], u[ 3]);s[17] = xor3x(s[17], t[1], u[ 3]);s[22] = xor3x(s[22], t[1], u[ 3]);
		s[ 3] = xor3x(s[ 3], t[2], u[ 4]);s[ 8] = xor3x(s[ 8], t[2], u[ 4]);s[13] = xor3x(s[13], t[2], u[ 4]);s[18] = xor3x(s[18], t[2], u[ 4]);s[23] = xor3x(s[23], t[2], u[ 4]);

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1]  = ROL2(s[6], 44);
		s[6]  = ROL2(s[9], 20);
		s[9]  = ROL2(s[22], 61);
		s[22] = ROL2(s[14], 39);
		s[14] = ROL2(s[20], 18);
		s[20] = ROL2(s[2], 62);
		s[2]  = ROL2(s[12], 43);
		s[12] = ROL2(s[13], 25);
		s[13] = ROL8(s[19]);
		s[19] = ROR8(s[23]);
		s[23] = ROL2(s[15], 41);
		s[15] = ROL2(s[4], 27);
		s[4]  = ROL2(s[24], 14);
		s[24] = ROL2(s[21], 2);
		s[21] = ROL2(s[8], 55);
		s[8]  = ROL2(s[16], 45);
		s[16] = ROL2(s[5], 36);
		s[5]  = ROL2(s[3], 28);
		s[3]  = ROL2(s[18], 21);
		s[18] = ROL2(s[17], 15);
		s[17] = ROL2(s[11], 10);
		s[11] = ROL2(s[7], 6);
		s[7]  = ROL2(s[10], 3);
		s[10] = ROL2(v, 1);

		#pragma unroll 5
		for(int j=0;j<25;j+=5){
			v=s[j];w=s[j + 1];s[j] = chi(v,w,s[j+2]);s[j+1] = chi(w,s[j+2],s[j+3]);s[j+2]=chi(s[j+2],s[j+3],s[j+4]);s[j+3]=chi(s[j+3],s[j+4],v);s[j+4]=chi(s[j+4],v,w);
		}

		s[0] ^= keccak_round_constants[i];
	}
}

#define TPB 512

__global__ __launch_bounds__(TPB)
void xevan_jh512_gpu_hash_128_keccak_128(uint32_t threads, uint32_t* g_hash, const uint32_t* __restrict__ g_nonceVector){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){
		const uint32_t hashPosition = (g_nonceVector != NULL) ? g_nonceVector[thread] : thread;

		uint32_t *Hash = &g_hash[hashPosition<<4];
		uint32_t hash[16];
		uint32_t x[8][4] = { /* init */
			{ 0x964bd16f, 0x17aa003e, 0x052e6a63, 0x43d5157a },{ 0x8d5e228a, 0x0bef970c, 0x591234e9, 0x61c3b3f2 },
			{ 0xc1a01d89, 0x1e806f53, 0x6b05a92a, 0x806d2bea },{ 0xdbcc8e58, 0xa6ba7520, 0x763a0fa9, 0xf73bf8ba },
			{ 0x05e66901, 0x694ae341, 0x8e8ab546, 0x5ae66f2e },{ 0xd0a74710, 0x243c84c1, 0xb1716e3b, 0x99c15a2d },
			{ 0xecf657cf, 0x56f8b19d, 0x7c8806a7, 0x56b11657 },{ 0xdffcc2e3, 0xfb1785e6, 0x78465a54, 0x4bdd8ccc }
		};

		*(uint2x4*)&hash[0] = __ldg4((uint2x4*)&Hash[0]);
		*(uint2x4*)&hash[8] = __ldg4((uint2x4*)&Hash[8]);
		
		#pragma unroll 16
		for (int i = 0; i < 16; i++){
			x[i/4][i & 3] ^= hash[i];
		}
		E8(x);

		#pragma unroll 16
		for (int i = 0; i < 16; i++){
			x[(i+16)/4][i & 3] ^= hash[i];
		}
		E8(x);

		x[0][0] ^= 0x80U;
		x[3][3] ^= 0x00040000U;
		E8(x);

		x[4][0] ^= 0x80U;
		x[7][3] ^= 0x00040000U;

///keccak
		uint2 keccak_gpu_state[25];
		uint64_t *s=(uint64_t*)&x[4][0];

		for (int i = 0; i<8; i++) {
			keccak_gpu_state[i] = vectorize(s[i]);
		}

		for (int i=8; i<25; i++) {
			keccak_gpu_state[i] = make_uint2(0, 0);
		}

		keccak_kernel(keccak_gpu_state);

		keccak_gpu_state[1] = ~keccak_gpu_state[1];
		keccak_gpu_state[2] = ~keccak_gpu_state[2];
		keccak_gpu_state[8] = ~keccak_gpu_state[8];
		keccak_gpu_state[12] = ~keccak_gpu_state[12];
		keccak_gpu_state[17] = ~keccak_gpu_state[17];
		keccak_gpu_state[20] = ~keccak_gpu_state[20];

		keccak_gpu_state[7] ^= vectorize(0x1UL);
		keccak_gpu_state[8] ^= vectorize(0x8000000000000000UL);

		keccak_gpu_state[1] ^= vectorize(0xFFFFFFFFFFFFFFFF);
		keccak_gpu_state[2] ^= vectorize(0xFFFFFFFFFFFFFFFF);
		keccak_gpu_state[8] ^= vectorize(0xFFFFFFFFFFFFFFFF);
		keccak_gpu_state[12] ^= vectorize(0xFFFFFFFFFFFFFFFF);
		keccak_gpu_state[17] ^= vectorize(0xFFFFFFFFFFFFFFFF);
		keccak_gpu_state[20] ^= vectorize(0xFFFFFFFFFFFFFFFF);

        keccak_kernel(keccak_gpu_state);

		uint64_t *inout =(uint64_t*)Hash;
		#pragma unroll 8
		for(int i=0; i<8; i++) {
			inout[i] = devectorize(keccak_gpu_state[i]);
		}
	}
}

__host__
void xevan_jh512_cpu_hash_128_keccak_128(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash)
{
	dim3 grid((threads + TPB-1)/TPB);
	dim3 block(TPB);

	xevan_jh512_gpu_hash_128_keccak_128<<<grid, block>>>(threads, d_hash, d_nonceVector);
}