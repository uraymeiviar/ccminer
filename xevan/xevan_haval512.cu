#include "cuda_helper.h"
#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

#define F1(x6, x5, x4, x3, x2, x1, x0) \
	(((x1) & ((x0) ^ (x4))) ^ ((x2) & (x5)) ^ ((x3) & (x6)) ^ (x0))

#define F2(x6, x5, x4, x3, x2, x1, x0) \
	(((x2) & (((x1) & ~(x3)) ^ ((x4) & (x5)) ^ (x6) ^ (x0))) ^ ((x4) & ((x1) ^ (x5))) ^ ((x3 & (x5)) ^ (x0)))

#define F3(x6, x5, x4, x3, x2, x1, x0) \
	(((x3) & (((x1) & (x2)) ^ (x6) ^ (x0))) ^ ((x1) & (x4)) ^ ((x2) & (x5)) ^ (x0))

#define F4(x6, x5, x4, x3, x2, x1, x0) \
	(((x3) & (((x1) & (x2)) ^ ((x4) | (x6)) ^ (x5))) ^ ((x4) & ((~(x2) & (x5)) ^ (x1) ^ (x6) ^ (x0))) ^ ((x2) & (x6)) ^ (x0))

#define F5(x6, x5, x4, x3, x2, x1, x0) \
	(((x0) & ~(((x1) & (x2) & (x3)) ^ (x5))) ^ ((x1) & (x4)) ^ ((x2) & (x5)) ^ ((x3) & (x6)))

#define FP5_1(x6, x5, x4, x3, x2, x1, x0) \
	F1(x3, x4, x1, x0, x5, x2, x6)
#define FP5_2(x6, x5, x4, x3, x2, x1, x0) \
	F2(x6, x2, x1, x0, x3, x4, x5)
#define FP5_3(x6, x5, x4, x3, x2, x1, x0) \
	F3(x2, x6, x0, x4, x3, x1, x5)
#define FP5_4(x6, x5, x4, x3, x2, x1, x0) \
	F4(x1, x5, x3, x2, x0, x4, x6)
#define FP5_5(x6, x5, x4, x3, x2, x1, x0) \
    F5(x2, x5, x0, x6, x4, x3, x1)
    
#define STEP(n, p, x7, x6, x5, x4, x3, x2, x1, x0, w, c) { \
	uint32_t t = FP ## n ## _ ## p(x6, x5, x4, x3, x2, x1, x0); \
	(x7) = (uint32_t)(ROTR32(t, 7) + ROTR32((x7), 11) + (w) + (c)); \
}

#define PASS1(n, in) { \
	STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, in[ 0], 0U); \
	STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, in[ 1], 0U); \
	STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, in[ 2], 0U); \
	STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, in[ 3], 0U); \
	STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, in[ 4], 0U); \
	STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, in[ 5], 0U); \
	STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, in[ 6], 0U); \
	STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, in[ 7], 0U); \
 \
	STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, in[ 8], 0U); \
	STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, in[ 9], 0U); \
	STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, in[10], 0U); \
	STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, in[11], 0U); \
	STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, in[12], 0U); \
	STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, in[13], 0U); \
	STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, in[14], 0U); \
	STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, in[15], 0U); \
 \
	STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, in[16], 0U); \
	STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, in[17], 0U); \
	STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, in[18], 0U); \
	STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, in[19], 0U); \
	STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, in[20], 0U); \
	STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, in[21], 0U); \
	STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, in[22], 0U); \
	STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, in[23], 0U); \
 \
	STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, in[24], 0U); \
	STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, in[25], 0U); \
	STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, in[26], 0U); \
	STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, in[27], 0U); \
	STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, in[28], 0U); \
	STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, in[29], 0U); \
	STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, in[30], 0U); \
	STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, in[31], 0U); \
}

#define PASS2(n, in) { \
	STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, in[ 5], 0x452821E6); \
	STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, in[14], 0x38D01377); \
	STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, in[26], 0xBE5466CF); \
	STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, in[18], 0x34E90C6C); \
	STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, in[11], 0xC0AC29B7); \
	STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, in[28], 0xC97C50DD); \
	STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, in[ 7], 0x3F84D5B5); \
	STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, in[16], 0xB5470917); \
 \
	STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, in[ 0], 0x9216D5D9); \
	STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, in[23], 0x8979FB1B); \
	STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, in[20], 0xD1310BA6); \
	STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, in[22], 0x98DFB5AC); \
	STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, in[ 1], 0x2FFD72DB); \
	STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, in[10], 0xD01ADFB7); \
	STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, in[ 4], 0xB8E1AFED); \
	STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, in[ 8], 0x6A267E96); \
 \
	STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, in[30], 0xBA7C9045); \
	STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, in[ 3], 0xF12C7F99); \
	STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, in[21], 0x24A19947); \
	STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, in[ 9], 0xB3916CF7); \
	STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, in[17], 0x0801F2E2); \
	STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, in[24], 0x858EFC16); \
	STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, in[29], 0x636920D8); \
	STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, in[ 6], 0x71574E69); \
 \
	STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, in[19], 0xA458FEA3); \
	STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, in[12], 0xF4933D7E); \
	STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, in[15], 0x0D95748F); \
	STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, in[13], 0x728EB658); \
	STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, in[ 2], 0x718BCD58); \
	STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, in[25], 0x82154AEE); \
	STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, in[31], 0x7B54A41D); \
	STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, in[27], 0xC25A59B5); \
}

#define PASS3(n, in) { \
	STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, in[19], 0x9C30D539); \
	STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, in[ 9], 0x2AF26013); \
	STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, in[ 4], 0xC5D1B023); \
	STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, in[20], 0x286085F0); \
	STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, in[28], 0xCA417918); \
	STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, in[17], 0xB8DB38EF); \
	STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, in[ 8], 0x8E79DCB0); \
	STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, in[22], 0x603A180E); \
 \
	STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, in[29], 0x6C9E0E8B); \
	STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, in[14], 0xB01E8A3E); \
	STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, in[25], 0xD71577C1); \
	STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, in[12], 0xBD314B27); \
	STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, in[24], 0x78AF2FDA); \
	STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, in[30], 0x55605C60); \
	STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, in[16], 0xE65525F3); \
	STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, in[26], 0xAA55AB94); \
 \
	STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, in[31], 0x57489862); \
	STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, in[15], 0x63E81440); \
	STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, in[ 7], 0x55CA396A); \
	STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, in[ 3], 0x2AAB10B6); \
	STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, in[ 1], 0xB4CC5C34); \
	STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, in[ 0], 0x1141E8CE); \
	STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, in[18], 0xA15486AF); \
	STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, in[27], 0x7C72E993); \
 \
	STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, in[13], 0xB3EE1411); \
	STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, in[ 6], 0x636FBC2A); \
	STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, in[21], 0x2BA9C55D); \
	STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, in[10], 0x741831F6); \
	STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, in[23], 0xCE5C3E16); \
	STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, in[11], 0x9B87931E); \
	STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, in[ 5], 0xAFD6BA33); \
	STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, in[ 2], 0x6C24CF5C); \
}

#define PASS4(n, in) { \
	STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, in[24], 0x7A325381); \
	STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, in[ 4], 0x28958677); \
	STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, in[ 0], 0x3B8F4898); \
	STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, in[14], 0x6B4BB9AF); \
	STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, in[ 2], 0xC4BFE81B); \
	STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, in[ 7], 0x66282193); \
	STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, in[28], 0x61D809CC); \
	STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, in[23], 0xFB21A991); \
 \
	STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, in[26], 0x487CAC60); \
	STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, in[ 6], 0x5DEC8032); \
	STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, in[30], 0xEF845D5D); \
	STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, in[20], 0xE98575B1); \
	STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, in[18], 0xDC262302); \
	STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, in[25], 0xEB651B88); \
	STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, in[19], 0x23893E81); \
	STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, in[ 3], 0xD396ACC5); \
 \
	STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, in[22], 0x0F6D6FF3); \
	STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, in[11], 0x83F44239); \
	STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, in[31], 0x2E0B4482); \
	STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, in[21], 0xA4842004); \
	STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, in[ 8], 0x69C8F04A); \
	STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, in[27], 0x9E1F9B5E); \
	STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, in[12], 0x21C66842); \
	STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, in[ 9], 0xF6E96C9A); \
 \
	STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, in[ 1], 0x670C9C61); \
	STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, in[29], 0xABD388F0); \
	STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, in[ 5], 0x6A51A0D2); \
	STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, in[15], 0xD8542F68); \
	STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, in[17], 0x960FA728); \
	STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, in[10], 0xAB5133A3); \
	STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, in[16], 0x6EEF0B6C); \
	STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, in[13], 0x137A3BE4); \
}

#define PASS5(n, in) { \
	STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, in[27], 0xBA3BF050); \
	STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, in[ 3], 0x7EFB2A98); \
	STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, in[21], 0xA1F1651D); \
	STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, in[26], 0x39AF0176); \
	STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, in[17], 0x66CA593E); \
	STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, in[11], 0x82430E88); \
	STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, in[20], 0x8CEE8619); \
	STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, in[29], 0x456F9FB4); \
 \
	STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, in[19], 0x7D84A5C3); \
	STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, in[ 0], 0x3B8B5EBE); \
	STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, in[12], 0xE06F75D8); \
	STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, in[ 7], 0x85C12073); \
	STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, in[13], 0x401A449F); \
	STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, in[ 8], 0x56C16AA6); \
	STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, in[31], 0x4ED3AA62); \
	STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, in[10], 0x363F7706); \
 \
	STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, in[ 5], 0x1BFEDF72); \
	STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, in[ 9], 0x429B023D); \
	STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, in[14], 0x37D0D724); \
	STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, in[30], 0xD00A1248); \
	STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, in[18], 0xDB0FEAD3); \
	STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, in[ 6], 0x49F1C09B); \
	STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, in[28], 0x075372C9); \
	STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, in[24], 0x80991B7B); \
 \
	STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, in[ 2], 0x25D479D8); \
	STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, in[23], 0xF6E8DEF7); \
	STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, in[16], 0xE3FE501A); \
	STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, in[22], 0xB6794C3B); \
	STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, in[ 4], 0x976CE0BD); \
	STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, in[ 1], 0x04C006BA); \
	STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, in[25], 0xC1A94FB6); \
	STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, in[15], 0x409F60C4); \
}

#define PASS5_final(n, in) { \
	STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, in[27], 0xBA3BF050); \
	STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, in[ 3], 0x7EFB2A98); \
	STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, in[21], 0xA1F1651D); \
	STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, in[26], 0x39AF0176); \
	STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, in[17], 0x66CA593E); \
	STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, in[11], 0x82430E88); \
	STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, in[20], 0x8CEE8619); \
	STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, in[29], 0x456F9FB4); \
 \
	STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, in[19], 0x7D84A5C3); \
	STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, in[ 0], 0x3B8B5EBE); \
	STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, in[12], 0xE06F75D8); \
	STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, in[ 7], 0x85C12073); \
	STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, in[13], 0x401A449F); \
	STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, in[ 8], 0x56C16AA6); \
	STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, in[31], 0x4ED3AA62); \
	STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, in[10], 0x363F7706); \
 \
	STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, in[ 5], 0x1BFEDF72); \
	STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, in[ 9], 0x429B023D); \
	STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, in[14], 0x37D0D724); \
	STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, in[30], 0xD00A1248); \
	STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, in[18], 0xDB0FEAD3); \
	STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, in[ 6], 0x49F1C09B); \
	STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, in[28], 0x075372C9); \
	STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, in[24], 0x80991B7B); \
 \
	STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, in[ 2], 0x25D479D8); \
	STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, in[23], 0xF6E8DEF7); \
}

#define sph_u32 uint32_t

#define H_SAVE_STATE \
	sph_u32 u0, u1, u2, u3, u4, u5, u6, u7; \
	do { \
		u0 = s0; \
		u1 = s1; \
		u2 = s2; \
		u3 = s3; \
		u4 = s4; \
		u5 = s5; \
		u6 = s6; \
		u7 = s7; \
	} while (0)

#define H_UPDATE_STATE   do { \
		s0 = SPH_T32(s0 + u0); \
		s1 = SPH_T32(s1 + u1); \
		s2 = SPH_T32(s2 + u2); \
		s3 = SPH_T32(s3 + u3); \
		s4 = SPH_T32(s4 + u4); \
		s5 = SPH_T32(s5 + u5); \
		s6 = SPH_T32(s6 + u6); \
		s7 = SPH_T32(s7 + u7); \
	} while (0)

#define CORE5(in)  do { \
	H_SAVE_STATE; \
	PASS1(5, in); \
	PASS2(5, in); \
	PASS3(5, in); \
	PASS4(5, in); \
	PASS5(5, in); \
	H_UPDATE_STATE; \
} while (0)

#define CORE5_F(in)  do { \
	H_SAVE_STATE; \
	PASS1(5, in); \
	PASS2(5, in); \
	PASS3(5, in); \
	PASS4(5, in); \
	PASS5_final(5, in); \
	H_UPDATE_STATE; \
} while (0)

#define TPB 512

__global__ __launch_bounds__(TPB, 2)
void xevan_haval512_gpu_hash_128(const uint32_t threads,const uint64_t*  g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
        uint32_t *pHash = (uint32_t *)&g_hash[thread<<3];
// haval
		sph_u32 s0 = SPH_C32(0x243F6A88);
		sph_u32 s1 = SPH_C32(0x85A308D3);
		sph_u32 s2 = SPH_C32(0x13198A2E);
		sph_u32 s3 = SPH_C32(0x03707344);
		sph_u32 s4 = SPH_C32(0xA4093822);
		sph_u32 s5 = SPH_C32(0x299F31D0);
		sph_u32 s6 = SPH_C32(0x082EFA98);
		sph_u32 s7 = SPH_C32(0xEC4E6C89);

		sph_u32 X_var[32];

		uint2x4* phash = (uint2x4*)pHash;
		uint2x4* outpt = (uint2x4*)X_var;
		outpt[0] = __ldg4(&phash[0]);
		outpt[1] = __ldg4(&phash[1]);

		#pragma unroll 16
		for (int i = 16; i < 32; i++){
			X_var[i] = 0;
		}

  		CORE5(X_var);

  		X_var[0] = 0x00000001U;

		#pragma unroll 28
		for (int i = 1; i < 29; i++){
			X_var[i] = 0;
		}

		X_var[29] = 0x40290000U;
		X_var[30] = 0x00000400U;
		X_var[31] = 0x00000000U;

		CORE5(X_var);

		pHash[0] = s0;
		pHash[1] = s1;
		pHash[2] = s2;
		pHash[3] = s3;
		pHash[4] = s4;
		pHash[5] = s5;
		pHash[6] = s6;
		pHash[7] = s7;

		pHash[8] = 0;
		pHash[9] = 0;
		pHash[10] = 0;
		pHash[11] = 0;
		pHash[12] = 0;
		pHash[13] = 0;
		pHash[14] = 0;
		pHash[15] = 0;
	}
}

__host__
void xevan_haval512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	dim3 grid((threads + TPB-1)/TPB);
	dim3 block(TPB);
	xevan_haval512_gpu_hash_128 <<<grid, block>>> (threads, (uint64_t*)d_hash);
}

#define TPB_F 512

__global__ __launch_bounds__(TPB_F, 4)
void xevan_haval512_gpu_hash_128_final(const uint32_t threads,const uint64_t* __restrict__ g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		uint32_t *pHash = (uint32_t *)&g_hash[thread<<3];
		// haval
		sph_u32 s0 = SPH_C32(0x243F6A88);
		sph_u32 s1 = SPH_C32(0x85A308D3);
		sph_u32 s2 = SPH_C32(0x13198A2E);
		sph_u32 s3 = SPH_C32(0x03707344);
		sph_u32 s4 = SPH_C32(0xA4093822);
		sph_u32 s5 = SPH_C32(0x299F31D0);
		sph_u32 s6 = SPH_C32(0x082EFA98);
		sph_u32 s7 = SPH_C32(0xEC4E6C89);

  		sph_u32 X_var[32];

		uint2x4* phash = (uint2x4*)pHash;
		uint2x4* outpt = (uint2x4*)X_var;
		outpt[0] = __ldg4(&phash[0]);
		outpt[1] = __ldg4(&phash[1]);

		//#pragma unroll 16
		for (int i = 16; i < 32; i++){
			X_var[i] = 0;
		}

		CORE5(X_var);

		X_var[0] = 0x00000001U;

		#pragma unroll 28
		for (int i = 1; i < 29; i++){
			X_var[i] = 0;
		}
		X_var[29] = 0x40290000U;
		X_var[30] = 0x00000400U;
		X_var[31] = 0x00000000U;

		CORE5_F(X_var);

		X_var[0] = s6;
		X_var[1] = s7;
	}
}

__host__
void xevan_haval512_cpu_hash_128_final(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	dim3 grid((threads + TPB_F-1)/TPB_F);
	dim3 block(TPB_F);

	xevan_haval512_gpu_hash_128_final <<<grid, block>>> (threads, (uint64_t*)d_hash);
}