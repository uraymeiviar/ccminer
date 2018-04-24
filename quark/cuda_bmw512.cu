#include <stdio.h>
#include <memory.h>

#define WANT_BMW512_80

#include "cuda_helper.h"
#include "cuda_vectors.h"
#include "cuda_vector_uint2x4.h"

__constant__ uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)

#include "cuda_bmw512_sm3.cuh"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 500
#endif

#undef SHL
#undef SHR
#undef CONST_EXP2

#define SHR(x, n) SHR2(x, n)
#define SHL(x, n) SHL2(x, n)
#define ROL(x, n) ROL2(x, n)

#define CONST_EXP2(i) \
	q[i+0] + ROL(q[i+1], 5)  + q[i+2] + ROL(q[i+3], 11) + \
	q[i+4] + ROL(q[i+5], 27) + q[i+6] + SWAPUINT2(q[i+7]) + \
	q[i+8] + ROL(q[i+9], 37) + q[i+10] + ROL(q[i+11], 43) + \
	q[i+12] + ROL(q[i+13], 53) + (SHR(q[i+14],1) ^ q[i+14]) + (SHR(q[i+15],2) ^ q[i+15])

__device__
void Compression512_64_first(uint2 *msg, uint2 *hash)
{
	// Compression ref. implementation
	uint2 q[32];
	uint2 tmp;

	tmp = (msg[5] ^ hash[5]) - (msg[7] ^ hash[7]) + (hash[10]) + (hash[13]) + (hash[14]);
	q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp, 4) ^ ROL(tmp, 37)) + hash[1];

	tmp = (msg[6] ^ hash[6]) - (msg[8] ^ hash[8]) + (hash[11]) + (hash[14]) - (msg[15] ^ hash[15]);
	q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROL(tmp, 13) ^ ROL(tmp, 43)) + hash[2];
	tmp = (msg[0] ^ hash[0]) + (msg[7] ^ hash[7]) + (hash[9]) - (hash[12]) + (msg[15] ^ hash[15]);
	q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROL(tmp, 19) ^ ROL(tmp, 53)) + hash[3];
	tmp = (msg[0] ^ hash[0]) - (msg[1] ^ hash[1]) + (msg[8] ^ hash[8]) - (hash[10]) + (hash[13]);
	q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROL(tmp, 28) ^ ROL(tmp, 59)) + hash[4];
	tmp = (msg[1] ^ hash[1]) + (msg[2] ^ hash[2]) + (hash[9]) - (hash[11]) - (hash[14]);
	q[4] = (SHR(tmp, 1) ^ tmp) + hash[5];
	tmp = (msg[3] ^ hash[3]) - (msg[2] ^ hash[2]) + (hash[10]) - (hash[12]) + (msg[15] ^ hash[15]);
	q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp, 4) ^ ROL(tmp, 37)) + hash[6];
	tmp = (msg[4] ^ hash[4]) - (msg[0] ^ hash[0]) - (msg[3] ^ hash[3]) - (hash[11]) + (hash[13]);
	q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROL(tmp, 13) ^ ROL(tmp, 43)) + hash[7];
	tmp = (msg[1] ^ hash[1]) - (msg[4] ^ hash[4]) - (msg[5] ^ hash[5]) - (hash[12]) - (hash[14]);
	q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROL(tmp, 19) ^ ROL(tmp, 53)) + hash[8];

	tmp = (msg[2] ^ hash[2]) - (msg[5] ^ hash[5]) - (msg[6] ^ hash[6]) + (hash[13]) - (msg[15] ^ hash[15]);
	q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROL(tmp, 28) ^ ROL(tmp, 59)) + hash[9];
	tmp = (msg[0] ^ hash[0]) - (msg[3] ^ hash[3]) + (msg[6] ^ hash[6]) - (msg[7] ^ hash[7]) + (hash[14]);
	q[9] = (SHR(tmp, 1) ^ tmp) + hash[10];
	tmp = (msg[8] ^ hash[8]) - (msg[1] ^ hash[1]) - (msg[4] ^ hash[4]) - (msg[7] ^ hash[7]) + (msg[15] ^ hash[15]);
	q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp, 4) ^ ROL(tmp, 37)) + hash[11];
	tmp = (msg[8] ^ hash[8]) - (msg[0] ^ hash[0]) - (msg[2] ^ hash[2]) - (msg[5] ^ hash[5]) + (hash[9]);
	q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROL(tmp, 13) ^ ROL(tmp, 43)) + hash[12];
	tmp = (msg[1] ^ hash[1]) + (msg[3] ^ hash[3]) - (msg[6] ^ hash[6]) - (hash[9]) + (hash[10]);
	q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROL(tmp, 19) ^ ROL(tmp, 53)) + hash[13];
	tmp = (msg[2] ^ hash[2]) + (msg[4] ^ hash[4]) + (msg[7] ^ hash[7]) + (hash[10]) + (hash[11]);
	q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROL(tmp, 28) ^ ROL(tmp, 59)) + hash[14];
	tmp = (msg[3] ^ hash[3]) - (msg[5] ^ hash[5]) + (msg[8] ^ hash[8]) - (hash[11]) - (hash[12]);
	q[14] = (SHR(tmp, 1) ^ tmp) + hash[15];
	tmp = (msg[12] ^ hash[12]) - (msg[4] ^ hash[4]) - (msg[6] ^ hash[6]) - (hash[9]) + (hash[13]);
	q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp, 4) ^ ROL(tmp, 37)) + hash[0];

	q[0 + 16] =
		(SHR(q[0], 1) ^ SHL(q[0], 2) ^ ROL(q[0], 13) ^ ROL(q[0], 43)) +
		(SHR(q[0 + 1], 2) ^ SHL(q[0 + 1], 1) ^ ROL(q[0 + 1], 19) ^ ROL(q[0 + 1], 53)) +
		(SHR(q[0 + 2], 2) ^ SHL(q[0 + 2], 2) ^ ROL(q[0 + 2], 28) ^ ROL(q[0 + 2], 59)) +
		(SHR(q[0 + 3], 1) ^ SHL(q[0 + 3], 3) ^ ROL(q[0 + 3], 4) ^ ROL(q[0 + 3], 37)) +
		(SHR(q[0 + 4], 1) ^ SHL(q[0 + 4], 2) ^ ROL(q[0 + 4], 13) ^ ROL(q[0 + 4], 43)) +
		(SHR(q[0 + 5], 2) ^ SHL(q[0 + 5], 1) ^ ROL(q[0 + 5], 19) ^ ROL(q[0 + 5], 53)) +
		(SHR(q[0 + 6], 2) ^ SHL(q[0 + 6], 2) ^ ROL(q[0 + 6], 28) ^ ROL(q[0 + 6], 59)) +
		(SHR(q[0 + 7], 1) ^ SHL(q[0 + 7], 3) ^ ROL(q[0 + 7], 4) ^ ROL(q[0 + 7], 37)) +
		(SHR(q[0 + 8], 1) ^ SHL(q[0 + 8], 2) ^ ROL(q[0 + 8], 13) ^ ROL(q[0 + 8], 43)) +
		(SHR(q[0 + 9], 2) ^ SHL(q[0 + 9], 1) ^ ROL(q[0 + 9], 19) ^ ROL(q[0 + 9], 53)) +
		(SHR(q[0 + 10], 2) ^ SHL(q[0 + 10], 2) ^ ROL(q[0 + 10], 28) ^ ROL(q[0 + 10], 59)) +
		(SHR(q[0 + 11], 1) ^ SHL(q[0 + 11], 3) ^ ROL(q[0 + 11], 4) ^ ROL(q[0 + 11], 37)) +
		(SHR(q[0 + 12], 1) ^ SHL(q[0 + 12], 2) ^ ROL(q[0 + 12], 13) ^ ROL(q[0 + 12], 43)) +
		(SHR(q[0 + 13], 2) ^ SHL(q[0 + 13], 1) ^ ROL(q[0 + 13], 19) ^ ROL(q[0 + 13], 53)) +
		(SHR(q[0 + 14], 2) ^ SHL(q[0 + 14], 2) ^ ROL(q[0 + 14], 28) ^ ROL(q[0 + 14], 59)) +
		(SHR(q[0 + 15], 1) ^ SHL(q[0 + 15], 3) ^ ROL(q[0 + 15], 4) ^ ROL(q[0 + 15], 37)) +
		((make_uint2(0x55555550ul,0x55555555) + ROL(msg[0], 0 + 1) +
		ROL(msg[0 + 3], 0 + 4)) ^ hash[0 + 7]);

	q[1 + 16] =
		(SHR(q[1], 1) ^ SHL(q[1], 2) ^ ROL(q[1], 13) ^ ROL(q[1], 43)) +
		(SHR(q[1 + 1], 2) ^ SHL(q[1 + 1], 1) ^ ROL(q[1 + 1], 19) ^ ROL(q[1 + 1], 53)) +
		(SHR(q[1 + 2], 2) ^ SHL(q[1 + 2], 2) ^ ROL(q[1 + 2], 28) ^ ROL(q[1 + 2], 59)) +
		(SHR(q[1 + 3], 1) ^ SHL(q[1 + 3], 3) ^ ROL(q[1 + 3], 4) ^ ROL(q[1 + 3], 37)) +
		(SHR(q[1 + 4], 1) ^ SHL(q[1 + 4], 2) ^ ROL(q[1 + 4], 13) ^ ROL(q[1 + 4], 43)) +
		(SHR(q[1 + 5], 2) ^ SHL(q[1 + 5], 1) ^ ROL(q[1 + 5], 19) ^ ROL(q[1 + 5], 53)) +
		(SHR(q[1 + 6], 2) ^ SHL(q[1 + 6], 2) ^ ROL(q[1 + 6], 28) ^ ROL(q[1 + 6], 59)) +
		(SHR(q[1 + 7], 1) ^ SHL(q[1 + 7], 3) ^ ROL(q[1 + 7], 4) ^ ROL(q[1 + 7], 37)) +
		(SHR(q[1 + 8], 1) ^ SHL(q[1 + 8], 2) ^ ROL(q[1 + 8], 13) ^ ROL(q[1 + 8], 43)) +
		(SHR(q[1 + 9], 2) ^ SHL(q[1 + 9], 1) ^ ROL(q[1 + 9], 19) ^ ROL(q[1 + 9], 53)) +
		(SHR(q[1 + 10], 2) ^ SHL(q[1 + 10], 2) ^ ROL(q[1 + 10], 28) ^ ROL(q[1 + 10], 59)) +
		(SHR(q[1 + 11], 1) ^ SHL(q[1 + 11], 3) ^ ROL(q[1 + 11], 4) ^ ROL(q[1 + 11], 37)) +
		(SHR(q[1 + 12], 1) ^ SHL(q[1 + 12], 2) ^ ROL(q[1 + 12], 13) ^ ROL(q[1 + 12], 43)) +
		(SHR(q[1 + 13], 2) ^ SHL(q[1 + 13], 1) ^ ROL(q[1 + 13], 19) ^ ROL(q[1 + 13], 53)) +
		(SHR(q[1 + 14], 2) ^ SHL(q[1 + 14], 2) ^ ROL(q[1 + 14], 28) ^ ROL(q[1 + 14], 59)) +
		(SHR(q[1 + 15], 1) ^ SHL(q[1 + 15], 3) ^ ROL(q[1 + 15], 4) ^ ROL(q[1 + 15], 37)) +
		((make_uint2(0xAAAAAAA5, 0x5AAAAAAA) + ROL(msg[1], 1 + 1) +
		ROL(msg[1 + 3], 1 + 4)) ^ hash[1 + 7]);

	q[2 + 16] = CONST_EXP2(2) +
		((make_uint2(0xFFFFFFFA, 0x5FFFFFFF) + ROL(msg[2], 2 + 1) +
		ROL(msg[2 + 3], 2 + 4) - ROL(msg[2 + 10], 2 + 11)) ^ hash[2 + 7]);
	q[3 + 16] = CONST_EXP2(3) +
		((make_uint2(0x5555554F, 0x65555555) + ROL(msg[3], 3 + 1) +
		ROL(msg[3 + 3], 3 + 4) - ROL(msg[3 + 10], 3 + 11)) ^ hash[3 + 7]);
	q[4 + 16] = CONST_EXP2(4) +
		((make_uint2(0xAAAAAAA4, 0x6AAAAAAA) +ROL(msg[4], 4 + 1) +
		ROL(msg[4 + 3], 4 + 4) - ROL(msg[4 + 10], 4 + 11)) ^ hash[4 + 7]);
	q[5 + 16] = CONST_EXP2(5) +
		((make_uint2(0xFFFFFFF9, 0x6FFFFFFF) + ROL(msg[5], 5 + 1) +
		ROL(msg[5 + 3], 5 + 4) - ROL(msg[5 + 10], 5 + 11)) ^ hash[5 + 7]);

	#pragma unroll 3
	for (int i = 6; i<9; i++) {
		q[i + 16] = CONST_EXP2(i) +
			((vectorize((i + 16)*(0x0555555555555555ull)) + ROL(msg[i], i + 1) -
			ROL(msg[i - 6], (i - 6) + 1)) ^ hash[i + 7]);
	}

	#pragma unroll 4
	for (int i = 9; i<13; i++) {
		q[i + 16] = CONST_EXP2(i) +
			((vectorize((i + 16)*(0x0555555555555555ull)) +
			ROL(msg[i + 3], i + 4) - ROL(msg[i - 6], (i - 6) + 1)) ^ hash[i - 9]);
	}

	q[13 + 16] = CONST_EXP2(13) +
		((make_uint2(0xAAAAAAA1, 0x9AAAAAAA) + ROL(msg[13], 13 + 1) +
		ROL(msg[13 - 13], (13 - 13) + 1) - ROL(msg[13 - 6], (13 - 6) + 1)) ^ hash[13 - 9]);
	q[14 + 16] = CONST_EXP2(14) +
		((make_uint2(0xFFFFFFF6, 0x9FFFFFFF) + ROL(msg[14], 14 + 1) +
		ROL(msg[14 - 13], (14 - 13) + 1) - ROL(msg[14 - 6], (14 - 6) + 1)) ^ hash[14 - 9]);
	q[15 + 16] = CONST_EXP2(15) +
		((make_uint2(0x5555554B, 0xA5555555) + ROL(msg[15], 15 + 1) +
		ROL(msg[15 - 13], (15 - 13) + 1) - ROL(msg[15 - 6], (15 - 6) + 1)) ^ hash[15 - 9]);


	uint2 XL64 = q[16] ^ q[17] ^ q[18] ^ q[19] ^ q[20] ^ q[21] ^ q[22] ^ q[23];
	uint2 XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

	hash[0] = (SHL(XH64, 5) ^ SHR(q[16], 5) ^ msg[0]) + (XL64 ^ q[24] ^ q[0]);
	hash[1] = (SHR(XH64, 7) ^ SHL(q[17], 8) ^ msg[1]) + (XL64 ^ q[25] ^ q[1]);
	hash[2] = (SHR(XH64, 5) ^ SHL(q[18], 5) ^ msg[2]) + (XL64 ^ q[26] ^ q[2]);
	hash[3] = (SHR(XH64, 1) ^ SHL(q[19], 5) ^ msg[3]) + (XL64 ^ q[27] ^ q[3]);
	hash[4] = (SHR(XH64, 3) ^ q[20] ^ msg[4]) + (XL64 ^ q[28] ^ q[4]);
	hash[5] = (SHL(XH64, 6) ^ SHR(q[21], 6) ^ msg[5]) + (XL64 ^ q[29] ^ q[5]);
	hash[6] = (SHR(XH64, 4) ^ SHL(q[22], 6) ^ msg[6]) + (XL64 ^ q[30] ^ q[6]);
	hash[7] = (SHR(XH64, 11) ^ SHL(q[23], 2) ^ msg[7]) + (XL64 ^ q[31] ^ q[7]);

	hash[8] =  ROL(hash[4], 9)  + (XH64 ^ q[24] ^ msg[8]) + (SHL(XL64, 8) ^ q[23] ^ q[8]);
	hash[9] =  ROL(hash[5], 10) + (XH64 ^ q[25]) + (SHR(XL64, 6) ^ q[16] ^ q[9]);
	hash[10] = ROL(hash[6], 11) + (XH64 ^ q[26]) + (SHL(XL64, 6) ^ q[17] ^ q[10]);
	hash[11] = ROL(hash[7], 12) + (XH64 ^ q[27]) + (SHL(XL64, 4) ^ q[18] ^ q[11]);
	hash[12] = ROL(hash[0], 13) + (XH64 ^ q[28]) + (SHR(XL64, 3) ^ q[19] ^ q[12]);
	hash[13] = ROL(hash[1], 14) + (XH64 ^ q[29]) + (SHR(XL64, 4) ^ q[20] ^ q[13]);
	hash[14] = ROL(hash[2], 15) + (XH64 ^ q[30]) + (SHR(XL64, 7) ^ q[21] ^ q[14]);
	hash[15] = ROL(hash[3], 16) + (XH64 ^ q[31] ^ msg[15]) + (SHR(XL64, 2) ^ q[22] ^ q[15]);
}

__device__
void Compression512(uint2 *msg, uint2 *hash)
{
	// Compression ref. implementation
	uint2 q[32];
	uint2 tmp;

	tmp = (msg[ 5] ^ hash[ 5]) - (msg[ 7] ^ hash[ 7]) + (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]) + (msg[14] ^ hash[14]);
	q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp,  4) ^ ROL(tmp, 37)) + hash[1];
	tmp = (msg[ 6] ^ hash[ 6]) - (msg[ 8] ^ hash[ 8]) + (msg[11] ^ hash[11]) + (msg[14] ^ hash[14]) - (msg[15] ^ hash[15]);
	q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROL(tmp, 13) ^ ROL(tmp, 43)) + hash[2];
	tmp = (msg[ 0] ^ hash[ 0]) + (msg[ 7] ^ hash[ 7]) + (msg[ 9] ^ hash[ 9]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
	q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROL(tmp, 19) ^ ROL(tmp, 53)) + hash[3];
	tmp = (msg[ 0] ^ hash[ 0]) - (msg[ 1] ^ hash[ 1]) + (msg[ 8] ^ hash[ 8]) - (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]);
	q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROL(tmp, 28) ^ ROL(tmp, 59)) + hash[4];
	tmp = (msg[ 1] ^ hash[ 1]) + (msg[ 2] ^ hash[ 2]) + (msg[ 9] ^ hash[ 9]) - (msg[11] ^ hash[11]) - (msg[14] ^ hash[14]);
	q[4] = (SHR(tmp, 1) ^ tmp) + hash[5];
	tmp = (msg[ 3] ^ hash[ 3]) - (msg[ 2] ^ hash[ 2]) + (msg[10] ^ hash[10]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
	q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp,  4) ^ ROL(tmp, 37)) + hash[6];
	tmp = (msg[ 4] ^ hash[ 4]) - (msg[ 0] ^ hash[ 0]) - (msg[ 3] ^ hash[ 3]) - (msg[11] ^ hash[11]) + (msg[13] ^ hash[13]);
	q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROL(tmp, 13) ^ ROL(tmp, 43)) + hash[7];
	tmp = (msg[ 1] ^ hash[ 1]) - (msg[ 4] ^ hash[ 4]) - (msg[ 5] ^ hash[ 5]) - (msg[12] ^ hash[12]) - (msg[14] ^ hash[14]);
	q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROL(tmp, 19) ^ ROL(tmp, 53)) + hash[8];
	tmp = (msg[ 2] ^ hash[ 2]) - (msg[ 5] ^ hash[ 5]) - (msg[ 6] ^ hash[ 6]) + (msg[13] ^ hash[13]) - (msg[15] ^ hash[15]);
	q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROL(tmp, 28) ^ ROL(tmp, 59)) + hash[9];
	tmp = (msg[ 0] ^ hash[ 0]) - (msg[ 3] ^ hash[ 3]) + (msg[ 6] ^ hash[ 6]) - (msg[ 7] ^ hash[ 7]) + (msg[14] ^ hash[14]);
	q[9] = (SHR(tmp, 1) ^ tmp) + hash[10];
	tmp = (msg[ 8] ^ hash[ 8]) - (msg[ 1] ^ hash[ 1]) - (msg[ 4] ^ hash[ 4]) - (msg[ 7] ^ hash[ 7]) + (msg[15] ^ hash[15]);
	q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp,  4) ^ ROL(tmp, 37)) + hash[11];
	tmp = (msg[ 8] ^ hash[ 8]) - (msg[ 0] ^ hash[ 0]) - (msg[ 2] ^ hash[ 2]) - (msg[ 5] ^ hash[ 5]) + (msg[ 9] ^ hash[ 9]);
	q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROL(tmp, 13) ^ ROL(tmp, 43)) + hash[12];
	tmp = (msg[ 1] ^ hash[ 1]) + (msg[ 3] ^ hash[ 3]) - (msg[ 6] ^ hash[ 6]) - (msg[ 9] ^ hash[ 9]) + (msg[10] ^ hash[10]);
	q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROL(tmp, 19) ^ ROL(tmp, 53)) + hash[13];
	tmp = (msg[ 2] ^ hash[ 2]) + (msg[ 4] ^ hash[ 4]) + (msg[ 7] ^ hash[ 7]) + (msg[10] ^ hash[10]) + (msg[11] ^ hash[11]);
	q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROL(tmp, 28) ^ ROL(tmp, 59)) + hash[14];
	tmp = (msg[ 3] ^ hash[ 3]) - (msg[ 5] ^ hash[ 5]) + (msg[ 8] ^ hash[ 8]) - (msg[11] ^ hash[11]) - (msg[12] ^ hash[12]);
	q[14] = (SHR(tmp, 1) ^ tmp) + hash[15];
	tmp = (msg[12] ^ hash[12]) - (msg[ 4] ^ hash[ 4]) - (msg[ 6] ^ hash[ 6]) - (msg[ 9] ^ hash[ 9]) + (msg[13] ^ hash[13]);
	q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp, 4) ^ ROL(tmp, 37)) + hash[0];

	q[0+16] =
		(SHR(q[0], 1) ^ SHL(q[0], 2) ^ ROL(q[0], 13) ^ ROL(q[0], 43)) +
		(SHR(q[0+1], 2) ^ SHL(q[0+1], 1) ^ ROL(q[0+1], 19) ^ ROL(q[0+1], 53)) +
		(SHR(q[0+2], 2) ^ SHL(q[0+2], 2) ^ ROL(q[0+2], 28) ^ ROL(q[0+2], 59)) +
		(SHR(q[0+3], 1) ^ SHL(q[0+3], 3) ^ ROL(q[0+3],  4) ^ ROL(q[0+3], 37)) +
		(SHR(q[0+4], 1) ^ SHL(q[0+4], 2) ^ ROL(q[0+4], 13) ^ ROL(q[0+4], 43)) +
		(SHR(q[0+5], 2) ^ SHL(q[0+5], 1) ^ ROL(q[0+5], 19) ^ ROL(q[0+5], 53)) +
		(SHR(q[0+6], 2) ^ SHL(q[0+6], 2) ^ ROL(q[0+6], 28) ^ ROL(q[0+6], 59)) +
		(SHR(q[0+7], 1) ^ SHL(q[0+7], 3) ^ ROL(q[0+7],  4) ^ ROL(q[0+7], 37)) +
		(SHR(q[0+8], 1) ^ SHL(q[0+8], 2) ^ ROL(q[0+8], 13) ^ ROL(q[0+8], 43)) +
		(SHR(q[0+9], 2) ^ SHL(q[0+9], 1) ^ ROL(q[0+9], 19) ^ ROL(q[0+9], 53)) +
		(SHR(q[0+10], 2) ^ SHL(q[0+10], 2) ^ ROL(q[0+10], 28) ^ ROL(q[0+10], 59)) +
		(SHR(q[0+11], 1) ^ SHL(q[0+11], 3) ^ ROL(q[0+11],  4) ^ ROL(q[0+11], 37)) +
		(SHR(q[0+12], 1) ^ SHL(q[0+12], 2) ^ ROL(q[0+12], 13) ^ ROL(q[0+12], 43)) +
		(SHR(q[0+13], 2) ^ SHL(q[0+13], 1) ^ ROL(q[0+13], 19) ^ ROL(q[0+13], 53)) +
		(SHR(q[0+14], 2) ^ SHL(q[0+14], 2) ^ ROL(q[0+14], 28) ^ ROL(q[0+14], 59)) +
		(SHR(q[0+15], 1) ^ SHL(q[0+15], 3) ^ ROL(q[0+15],  4) ^ ROL(q[0+15], 37)) +
		((make_uint2(0x55555550ul, 0x55555555) + ROL(msg[0], 0 + 1) +
		ROL(msg[0+3], 0+4) - ROL(msg[0+10], 0+11) ) ^ hash[0+7]);

	q[1 + 16] =
		(SHR(q[1], 1) ^ SHL(q[1], 2) ^ ROL(q[1], 13) ^ ROL(q[1], 43)) +
		(SHR(q[1 + 1], 2) ^ SHL(q[1 + 1], 1) ^ ROL(q[1 + 1], 19) ^ ROL(q[1 + 1], 53)) +
		(SHR(q[1 + 2], 2) ^ SHL(q[1 + 2], 2) ^ ROL(q[1 + 2], 28) ^ ROL(q[1 + 2], 59)) +
		(SHR(q[1 + 3], 1) ^ SHL(q[1 + 3], 3) ^ ROL(q[1 + 3], 4) ^ ROL(q[1 + 3], 37)) +
		(SHR(q[1 + 4], 1) ^ SHL(q[1 + 4], 2) ^ ROL(q[1 + 4], 13) ^ ROL(q[1 + 4], 43)) +
		(SHR(q[1 + 5], 2) ^ SHL(q[1 + 5], 1) ^ ROL(q[1 + 5], 19) ^ ROL(q[1 + 5], 53)) +
		(SHR(q[1 + 6], 2) ^ SHL(q[1 + 6], 2) ^ ROL(q[1 + 6], 28) ^ ROL(q[1 + 6], 59)) +
		(SHR(q[1 + 7], 1) ^ SHL(q[1 + 7], 3) ^ ROL(q[1 + 7], 4) ^ ROL(q[1 + 7], 37)) +
		(SHR(q[1 + 8], 1) ^ SHL(q[1 + 8], 2) ^ ROL(q[1 + 8], 13) ^ ROL(q[1 + 8], 43)) +
		(SHR(q[1 + 9], 2) ^ SHL(q[1 + 9], 1) ^ ROL(q[1 + 9], 19) ^ ROL(q[1 + 9], 53)) +
		(SHR(q[1 + 10], 2) ^ SHL(q[1 + 10], 2) ^ ROL(q[1 + 10], 28) ^ ROL(q[1 + 10], 59)) +
		(SHR(q[1 + 11], 1) ^ SHL(q[1 + 11], 3) ^ ROL(q[1 + 11], 4) ^ ROL(q[1 + 11], 37)) +
		(SHR(q[1 + 12], 1) ^ SHL(q[1 + 12], 2) ^ ROL(q[1 + 12], 13) ^ ROL(q[1 + 12], 43)) +
		(SHR(q[1 + 13], 2) ^ SHL(q[1 + 13], 1) ^ ROL(q[1 + 13], 19) ^ ROL(q[1 + 13], 53)) +
		(SHR(q[1 + 14], 2) ^ SHL(q[1 + 14], 2) ^ ROL(q[1 + 14], 28) ^ ROL(q[1 + 14], 59)) +
		(SHR(q[1 + 15], 1) ^ SHL(q[1 + 15], 3) ^ ROL(q[1 + 15], 4) ^ ROL(q[1 + 15], 37)) +
		((make_uint2(0xAAAAAAA5, 0x5AAAAAAA) + ROL(msg[1], 1 + 1) +
		ROL(msg[1 + 3], 1 + 4) - ROL(msg[1 + 10], 1 + 11)) ^ hash[1 + 7]);

	q[2 + 16] = CONST_EXP2(2) +
		((make_uint2(0xFFFFFFFA, 0x5FFFFFFF) + ROL(msg[2], 2 + 1) +
		ROL(msg[2+3], 2+4) - ROL(msg[2+10], 2+11) ) ^ hash[2+7]);
	q[3 + 16] = CONST_EXP2(3) +
		((make_uint2(0x5555554F, 0x65555555) + ROL(msg[3], 3 + 1) +
		ROL(msg[3 + 3], 3 + 4) - ROL(msg[3 + 10], 3 + 11)) ^ hash[3 + 7]);
	q[4 + 16] = CONST_EXP2(4) +
		((make_uint2(0xAAAAAAA4, 0x6AAAAAAA) + ROL(msg[4], 4 + 1) +
		ROL(msg[4 + 3], 4 + 4) - ROL(msg[4 + 10], 4 + 11)) ^ hash[4 + 7]);
	q[5 + 16] = CONST_EXP2(5) +
		((make_uint2(0xFFFFFFF9, 0x6FFFFFFF) + ROL(msg[5], 5 + 1) +
		ROL(msg[5 + 3], 5 + 4) - ROL(msg[5 + 10], 5 + 11)) ^ hash[5 + 7]);
	q[6 + 16] = CONST_EXP2(6) +
		((make_uint2(0x5555554E, 0x75555555)+ ROL(msg[6], 6 + 1) +
		ROL(msg[6 + 3], 6 + 4) - ROL(msg[6 - 6], (6 - 6) + 1)) ^ hash[6 + 7]);
	q[7 + 16] = CONST_EXP2(7) +
		((make_uint2(0xAAAAAAA3, 0x7AAAAAAA) + ROL(msg[7], 7 + 1) +
		ROL(msg[7 + 3], 7 + 4) - ROL(msg[7 - 6], (7 - 6) + 1)) ^ hash[7 + 7]);
	q[8 + 16] = CONST_EXP2(8) +
		((make_uint2(0xFFFFFFF8, 0x7FFFFFFF) + ROL(msg[8], 8 + 1) +
		ROL(msg[8 + 3], 8 + 4) - ROL(msg[8 - 6], (8 - 6) + 1)) ^ hash[8 + 7]);
	q[9 + 16] = CONST_EXP2(9) +
		((make_uint2(0x5555554D, 0x85555555) + ROL(msg[9], 9 + 1) +
		ROL(msg[9 + 3], 9 + 4) - ROL(msg[9 - 6], (9 - 6) + 1)) ^ hash[9 - 9]);
	q[10 + 16] = CONST_EXP2(10) +
		((make_uint2(0xAAAAAAA2, 0x8AAAAAAA) + ROL(msg[10], 10 + 1) +
		ROL(msg[10 + 3], 10 + 4) - ROL(msg[10 - 6], (10 - 6) + 1)) ^ hash[10 - 9]);
	q[11 + 16] = CONST_EXP2(11) +
		((make_uint2(0xFFFFFFF7, 0x8FFFFFFF) + ROL(msg[11], 11 + 1) +
		ROL(msg[11 + 3], 11 + 4) - ROL(msg[11 - 6], (11 - 6) + 1)) ^ hash[11 - 9]);
	q[12 + 16] = CONST_EXP2(12) +
		((make_uint2(0x5555554C, 0x95555555) + ROL(msg[12], 12 + 1) +
		ROL(msg[12 + 3], 12 + 4) - ROL(msg[12 - 6], (12 - 6) + 1)) ^ hash[12 - 9]);
	q[13 + 16] = CONST_EXP2(13) +
		((make_uint2(0xAAAAAAA1, 0x9AAAAAAA) + ROL(msg[13], 13 + 1) +
		ROL(msg[13 - 13], (13 - 13) + 1) - ROL(msg[13 - 6], (13 - 6) + 1)) ^ hash[13 - 9]);
	q[14 + 16] = CONST_EXP2(14) +
		((make_uint2(0xFFFFFFF6, 0x9FFFFFFF) + ROL(msg[14], 14 + 1) +
		ROL(msg[14 - 13], (14 - 13) + 1) - ROL(msg[14 - 6], (14 - 6) + 1)) ^ hash[14 - 9]);
	q[15 + 16] = CONST_EXP2(15) +
		((make_uint2(0x5555554B, 0xA5555555) + ROL(msg[15], 15 + 1) +
		ROL(msg[15 - 13], (15 - 13) + 1) - ROL(msg[15 - 6], (15 - 6) + 1)) ^ hash[15 - 9]);

	uint2 XL64 = q[16]^q[17]^q[18]^q[19]^q[20]^q[21]^q[22]^q[23];
	uint2 XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

	hash[0] = (SHL(XH64, 5) ^ SHR(q[16],5) ^ msg[ 0]) + (XL64 ^ q[24] ^ q[ 0]);
	hash[1] = (SHR(XH64, 7) ^ SHL(q[17],8) ^ msg[ 1]) + (XL64 ^ q[25] ^ q[ 1]);
	hash[2] = (SHR(XH64, 5) ^ SHL(q[18],5) ^ msg[ 2]) + (XL64 ^ q[26] ^ q[ 2]);
	hash[3] = (SHR(XH64, 1) ^ SHL(q[19],5) ^ msg[ 3]) + (XL64 ^ q[27] ^ q[ 3]);
	hash[4] = (SHR(XH64, 3) ^     q[20]    ^ msg[ 4]) + (XL64 ^ q[28] ^ q[ 4]);
	hash[5] = (SHL(XH64, 6) ^ SHR(q[21],6) ^ msg[ 5]) + (XL64 ^ q[29] ^ q[ 5]);
	hash[6] = (SHR(XH64, 4) ^ SHL(q[22],6) ^ msg[ 6]) + (XL64 ^ q[30] ^ q[ 6]);
	hash[7] = (SHR(XH64,11) ^ SHL(q[23],2) ^ msg[ 7]) + (XL64 ^ q[31] ^ q[ 7]);

	hash[ 8] = ROL(hash[4], 9) + (XH64 ^ q[24] ^ msg[ 8]) + (SHL(XL64,8) ^ q[23] ^ q[ 8]);
	hash[ 9] = ROL(hash[5],10) + (XH64 ^ q[25] ^ msg[ 9]) + (SHR(XL64,6) ^ q[16] ^ q[ 9]);
	hash[10] = ROL(hash[6],11) + (XH64 ^ q[26] ^ msg[10]) + (SHL(XL64,6) ^ q[17] ^ q[10]);
	hash[11] = ROL(hash[7],12) + (XH64 ^ q[27] ^ msg[11]) + (SHL(XL64,4) ^ q[18] ^ q[11]);
	hash[12] = ROL(hash[0],13) + (XH64 ^ q[28] ^ msg[12]) + (SHR(XL64,3) ^ q[19] ^ q[12]);
	hash[13] = ROL(hash[1],14) + (XH64 ^ q[29] ^ msg[13]) + (SHR(XL64,4) ^ q[20] ^ q[13]);
	hash[14] = ROL(hash[2],15) + (XH64 ^ q[30] ^ msg[14]) + (SHR(XL64,7) ^ q[21] ^ q[14]);
	hash[15] = ROL(hash[3],16) + (XH64 ^ q[31] ^ msg[15]) + (SHR(XL64, 2) ^ q[22] ^ q[15]);
}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(32, 16)
#else
__launch_bounds__(64, 8)
#endif
void quark_bmw512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		uint32_t hashPosition = nounce - startNounce;
		uint64_t *inpHash = &g_hash[hashPosition * 8];

		// Init
		uint2 h[16] = {
			{ 0x84858687UL, 0x80818283UL },
			{ 0x8C8D8E8FUL, 0x88898A8BUL },
			{ 0x94959697UL, 0x90919293UL },
			{ 0x9C9D9E9FUL, 0x98999A9BUL },
			{ 0xA4A5A6A7UL, 0xA0A1A2A3UL },
			{ 0xACADAEAFUL, 0xA8A9AAABUL },
			{ 0xB4B5B6B7UL, 0xB0B1B2B3UL },
			{ 0xBCBDBEBFUL, 0xB8B9BABBUL },
			{ 0xC4C5C6C7UL, 0xC0C1C2C3UL },
			{ 0xCCCDCECFUL, 0xC8C9CACBUL },
			{ 0xD4D5D6D7UL, 0xD0D1D2D3UL },
			{ 0xDCDDDEDFUL, 0xD8D9DADBUL },
			{ 0xE4E5E6E7UL, 0xE0E1E2E3UL },
			{ 0xECEDEEEFUL, 0xE8E9EAEBUL },
			{ 0xF4F5F6F7UL, 0xF0F1F2F3UL },
			{ 0xFCFDFEFFUL, 0xF8F9FAFBUL }
		};

		// Nachricht kopieren (Achtung, die Nachricht hat 64 Byte,
		// BMW arbeitet mit 128 Byte!!!
		uint2 message[16];
		#pragma unroll
		for(int i=0;i<8;i++)
			message[i] = vectorize(inpHash[i]);

		#pragma unroll 6
		for(int i=9;i<15;i++)
			message[i] = make_uint2(0,0);

		// Padding einf\FCgen (Byteorder?!?)
		message[8] = make_uint2(0x80,0);
		// L\E4nge (in Bits, d.h. 64 Byte * 8 = 512 Bits
		message[15] = make_uint2(512,0);

		// Compression 1
		Compression512_64_first(message, h);

		// Final
		#pragma unroll
		for(int i=0;i<16;i++)
		{
			message[i].y = 0xaaaaaaaa;
			message[i].x = 0xaaaaaaa0ul + (uint32_t)i;
		}
		Compression512(h, message);

		// fertig
		uint64_t *outpHash = &g_hash[hashPosition * 8];

		#pragma unroll
		for(int i=0;i<8;i++)
			outpHash[i] = devectorize(message[i+8]);
	}
}

__global__ __launch_bounds__(256, 2)
void quark_bmw512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = startNounce + thread;

		// Init
		uint2 h[16] = {
			{ 0x84858687UL, 0x80818283UL },
			{ 0x8C8D8E8FUL, 0x88898A8BUL },
			{ 0x94959697UL, 0x90919293UL },
			{ 0x9C9D9E9FUL, 0x98999A9BUL },
			{ 0xA4A5A6A7UL, 0xA0A1A2A3UL },
			{ 0xACADAEAFUL, 0xA8A9AAABUL },
			{ 0xB4B5B6B7UL, 0xB0B1B2B3UL },
			{ 0xBCBDBEBFUL, 0xB8B9BABBUL },
			{ 0xC4C5C6C7UL, 0xC0C1C2C3UL },
			{ 0xCCCDCECFUL, 0xC8C9CACBUL },
			{ 0xD4D5D6D7UL, 0xD0D1D2D3UL },
			{ 0xDCDDDEDFUL, 0xD8D9DADBUL },
			{ 0xE4E5E6E7UL, 0xE0E1E2E3UL },
			{ 0xECEDEEEFUL, 0xE8E9EAEBUL },
			{ 0xF4F5F6F7UL, 0xF0F1F2F3UL },
			{ 0xFCFDFEFFUL, 0xF8F9FAFBUL }
		};
		// Nachricht kopieren (Achtung, die Nachricht hat 64 Byte,
		// BMW arbeitet mit 128 Byte!!!
		uint2 message[16];
#pragma unroll 16
		for(int i=0;i<16;i++)
			message[i] = vectorize(c_PaddedMessage80[i]);

		// die Nounce durch die thread-spezifische ersetzen
		message[9].y = cuda_swab32(nounce);	//REPLACE_HIDWORD(message[9], cuda_swab32(nounce));

		// Compression 1
		Compression512(message, h);

#pragma unroll 16
		for(int i=0;i<16;i++)
			message[i] = make_uint2(0xaaaaaaa0+i,0xaaaaaaaa);


		Compression512(h, message);

		// fertig
		uint64_t *outpHash = &g_hash[thread * 8];

#pragma unroll 8
		for(int i=0;i<8;i++)
			outpHash[i] = devectorize(message[i+8]);
	}
}

__host__
void quark_bmw512_cpu_setBlock_80(void *pdata)
{
	unsigned char PaddedMessage[128];
	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage+80, 0, 48);
	uint64_t *message = (uint64_t*)PaddedMessage;
	message[10] = SPH_C64(0x80);
	message[15] = SPH_C64(640);
	cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 16*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

__host__
void quark_bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	int dev_id = device_map[thr_id];

	if (device_sm[dev_id] > 300 && cuda_arch[dev_id] > 300)
		quark_bmw512_gpu_hash_80<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash);
	else
		quark_bmw512_gpu_hash_80_30<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash);
}

__host__
void quark_bmw512_cpu_init(int thr_id, uint32_t threads)
{
	cuda_get_arch(thr_id);
}

__host__
void quark_bmw512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = 32;
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	int dev_id = device_map[thr_id];
	if (device_sm[dev_id] > 300 && cuda_arch[dev_id] > 300)
		quark_bmw512_gpu_hash_64<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
	else
		quark_bmw512_gpu_hash_64_30<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
}

//========================================

#define TPB_BMW 256
#define BMW_SH 64
// Wolf's BMW512, loosely based on SPH's implementation
#define as_uint2(x) (x)
#define FAST_ROTL64_LO ROTL64
#define FAST_ROTL64_HI ROTL64

#undef CONST_EXP2

#define CONST_EXP2  q[i+0] + FAST_ROTL64_LO(as_uint2(q[i+1]), 5)  + q[i+2] + FAST_ROTL64_LO(as_uint2(q[i+3]), 11) + \
                    q[i+4] + FAST_ROTL64_LO(as_uint2(q[i+5]), 27) + q[i+6] + as_ulong(as_uint2(q[i+7]).s10) + \
                    q[i+8] + FAST_ROTL64_HI(as_uint2(q[i+9]), 37) + q[i+10] + FAST_ROTL64_HI(as_uint2(q[i+11]), 43) + \
                    q[i+12] + FAST_ROTL64_HI(as_uint2(q[i+13]), 53) + (SHR(q[i+14],1) ^ q[i+14]) + (SHR(q[i+15],2) ^ q[i+15])

#undef SHL
#undef SHR
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

__global__ __launch_bounds__(TPB_BMW,2)
void quark_bmw512_gpu_hash_64x(uint32_t threads, uint64_t *const __restrict__ g_hash, const uint32_t *const __restrict__ g_nonceVector){
	const  uint64_t BMW512_IV[16] =
	{
		0x8081828384858687UL, 0x88898A8B8C8D8E8FUL, 0x9091929394959697UL, 0x98999A9B9C9D9E9FUL,
		0xA0A1A2A3A4A5A6A7UL, 0xA8A9AAABACADAEAFUL, 0xB0B1B2B3B4B5B6B7UL, 0xB8B9BABBBCBDBEBFUL,
		0xC0C1C2C3C4C5C6C7UL, 0xC8C9CACBCCCDCECFUL, 0xD0D1D2D3D4D5D6D7UL, 0xD8D9DADBDCDDDEDFUL,
		0xE0E1E2E3E4E5E6E7UL, 0xE8E9EAEBECEDEEEFUL, 0xF0F1F2F3F4F5F6F7UL, 0xF8F9FAFBFCFDFEFFUL
	};

	const  uint64_t BMW512_FINAL[16] =
	{
		0xAAAAAAAAAAAAAAA0UL, 0xAAAAAAAAAAAAAAA1UL, 0xAAAAAAAAAAAAAAA2UL, 0xAAAAAAAAAAAAAAA3UL,
		0xAAAAAAAAAAAAAAA4UL, 0xAAAAAAAAAAAAAAA5UL, 0xAAAAAAAAAAAAAAA6UL, 0xAAAAAAAAAAAAAAA7UL,
		0xAAAAAAAAAAAAAAA8UL, 0xAAAAAAAAAAAAAAA9UL, 0xAAAAAAAAAAAAAAAAUL, 0xAAAAAAAAAAAAAAABUL,
		0xAAAAAAAAAAAAAAACUL, 0xAAAAAAAAAAAAAAADUL, 0xAAAAAAAAAAAAAAAEUL, 0xAAAAAAAAAAAAAAAFUL
	};

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads){
		const uint32_t hashPosition = (g_nonceVector == NULL) ? thread : g_nonceVector[thread];
		uint64_t *inpHash = &g_hash[8 * hashPosition];
		uint64_t msg[8];

		uint2x4* phash = (uint2x4*)inpHash;
		uint2x4* outpt = (uint2x4*)msg;
		outpt[0] = __ldg4(&phash[0]);
		outpt[1] = __ldg4(&phash[1]);

		uint64_t msg0[16] = { 0 }, msg1[16] = { 0 };
		uint64_t q[32];

		for(int i = 0; i < 8; ++i) msg0[i] = (msg[i]);

		msg1[0] = 0x80UL;
		msg1[15] = 1024UL;

		BMW_Compression(msg0, BMW512_IV, q);
		BMW_Compression(msg1, msg0,q);
		BMW_Compression(msg1, BMW512_FINAL,q);
	
		for(int i = 0; i < 8; ++i) inpHash[i] = (msg1[i + 8]);
	}
}

__host__ void quark_bmw512_cpu_hash_64x(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash)
{
    const uint32_t threadsperblock = TPB_BMW;

    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    quark_bmw512_gpu_hash_64x<<<grid, block>>>(threads, (uint64_t*)d_hash, d_nonceVector);
}
