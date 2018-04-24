/*
 * luffa_for_32.c
 * Version 2.0 (Sep 15th 2009)
 *
 * Copyright (C) 2008-2009 Hitachi, Ltd. All rights reserved.
 *
 * Hitachi, Ltd. is the owner of this software and hereby grant
 * the U.S. Government and any interested party the right to use
 * this software for the purposes of the SHA-3 evaluation process,
 * notwithstanding that this software is copyrighted.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include "cuda_helper.h"
#include "cuda_vectors.h"
#include "cuda_vectors_alexis.h"

typedef unsigned char BitSequence;

typedef struct {
    uint32_t buffer[8]; /* Buffer to be hashed */
    uint32_t chainv[40];   /* Chaining values */
} hashState;

#define MULT2(a,j)\
    tmp = a[7+(8*j)];\
    a[7+(8*j)] = a[6+(8*j)];\
    a[6+(8*j)] = a[5+(8*j)];\
    a[5+(8*j)] = a[4+(8*j)];\
    a[4+(8*j)] = a[3+(8*j)] ^ tmp;\
    a[3+(8*j)] = a[2+(8*j)] ^ tmp;\
    a[2+(8*j)] = a[1+(8*j)];\
    a[1+(8*j)] = a[0+(8*j)] ^ tmp;\
    a[0+(8*j)] = tmp;

#if __CUDA_ARCH__ < 350
#define LROT(x,bits) ((x << bits) | (x >> (32 - bits)))
#else
#define LROT(x, bits) __funnelshift_l(x, x, bits)
#endif

#define TWEAK(a0,a1,a2,a3,j)\
    a0 = LROT(a0,j);\
    a1 = LROT(a1,j);\
    a2 = LROT(a2,j);\
    a3 = LROT(a3,j);

#define STEP(c0,c1)\
    SUBCRUMB(chainv[0],chainv[1],chainv[2],chainv[3],tmp);\
    SUBCRUMB(chainv[5],chainv[6],chainv[7],chainv[4],tmp);\
    MIXWORD(chainv[0],chainv[4]);\
    MIXWORD(chainv[1],chainv[5]);\
    MIXWORD(chainv[2],chainv[6]);\
    MIXWORD(chainv[3],chainv[7]);\
    ADD_CONSTANT(chainv[0],chainv[4],c0,c1);

#define SUBCRUMB(a0,a1,a2,a3,a4)\
    a4  = a0;\
    a0 |= a1;\
    a2 ^= a3;\
    a1  = ~a1;\
    a0 ^= a3;\
    a3 &= a4;\
    a1 ^= a3;\
    a3 ^= a2;\
    a2 &= a0;\
    a0  = ~a0;\
    a2 ^= a1;\
    a1 |= a3;\
    a4 ^= a1;\
    a3 ^= a2;\
    a2 &= a1;\
    a1 ^= a0;\
    a0  = a4;

#define MIXWORD(a0,a4)\
    a4 ^= a0;\
    a0  = LROT(a0,2);\
    a0 ^= a4;\
    a4  = LROT(a4,14);\
    a4 ^= a0;\
    a0  = LROT(a0,10);\
    a0 ^= a4;\
    a4  = LROT(a4,1);

#define ADD_CONSTANT(a0,b0,c0,c1)\
    a0 ^= c0;\
    b0 ^= c1;

/* initial values of chaining variables */
__device__ __constant__ uint32_t c_IV[40];
const uint32_t h_IV[40] = {
    0x6d251e69,0x44b051e0,0x4eaa6fb4,0xdbf78465,
    0x6e292011,0x90152df4,0xee058139,0xdef610bb,
    0xc3b44b95,0xd9d2f256,0x70eee9a0,0xde099fa3,
    0x5d9b0557,0x8fc944b3,0xcf1ccf0e,0x746cd581,
    0xf7efc89d,0x5dba5781,0x04016ce5,0xad659c05,
    0x0306194f,0x666d1836,0x24aa230a,0x8b264ae7,
    0x858075d5,0x36d79cce,0xe571f7d7,0x204b1f67,
    0x35870c6a,0x57e9e923,0x14bcb808,0x7cde72ce,
    0x6c68e9be,0x5ec41e22,0xc825b7c7,0xaffb4363,
    0xf5df3999,0x0fc688f1,0xb07224cc,0x03e86cea};

__constant__ const uint32_t c_CNS[80] = {
		0x303994a6,0xe0337818,0xc0e65299,0x441ba90d, 0x6cc33a12,0x7f34d442,0xdc56983e,0x9389217f, 0x1e00108f,0xe5a8bce6,0x7800423d,0x5274baf4, 0x8f5b7882,0x26889ba7,0x96e1db12,0x9a226e9d,
		0xb6de10ed,0x01685f3d,0x70f47aae,0x05a17cf4, 0x0707a3d4,0xbd09caca,0x1c1e8f51,0xf4272b28, 0x707a3d45,0x144ae5cc,0xaeb28562,0xfaa7ae2b, 0xbaca1589,0x2e48f1c1,0x40a46f3e,0xb923c704,
		0xfc20d9d2,0xe25e72c1,0x34552e25,0xe623bb72, 0x7ad8818f,0x5c58a4a4,0x8438764a,0x1e38e2e7, 0xbb6de032,0x78e38b9d,0xedb780c8,0x27586719, 0xd9847356,0x36eda57f,0xa2c78434,0x703aace7,
		0xb213afa5,0xe028c9bf,0xc84ebe95,0x44756f91, 0x4e608a22,0x7e8fce32,0x56d858fe,0x956548be, 0x343b138f,0xfe191be2,0xd0ec4e3d,0x3cb226e5, 0x2ceb4882,0x5944a28e,0xb3ad2208,0xa1c4c355,
		0xf0d2e9e3,0x5090d577,0xac11d7fa,0x2d1925ab, 0x1bcb66f2,0xb46496ac,0x6f2d9bc9,0xd1925ab0, 0x78602649,0x29131ab6,0x8edae952,0x0fc053c3, 0x3b6ba548,0x3f014f0c,0xedae9520,0xfc053c31
	};

static uint32_t h_CNS[80] = {
		0x303994a6,0xe0337818,0xc0e65299,0x441ba90d, 0x6cc33a12,0x7f34d442,0xdc56983e,0x9389217f, 0x1e00108f,0xe5a8bce6,0x7800423d,0x5274baf4, 0x8f5b7882,0x26889ba7,0x96e1db12,0x9a226e9d,
		0xb6de10ed,0x01685f3d,0x70f47aae,0x05a17cf4, 0x0707a3d4,0xbd09caca,0x1c1e8f51,0xf4272b28, 0x707a3d45,0x144ae5cc,0xaeb28562,0xfaa7ae2b, 0xbaca1589,0x2e48f1c1,0x40a46f3e,0xb923c704,
		0xfc20d9d2,0xe25e72c1,0x34552e25,0xe623bb72, 0x7ad8818f,0x5c58a4a4,0x8438764a,0x1e38e2e7, 0xbb6de032,0x78e38b9d,0xedb780c8,0x27586719, 0xd9847356,0x36eda57f,0xa2c78434,0x703aace7,
		0xb213afa5,0xe028c9bf,0xc84ebe95,0x44756f91, 0x4e608a22,0x7e8fce32,0x56d858fe,0x956548be, 0x343b138f,0xfe191be2,0xd0ec4e3d,0x3cb226e5, 0x2ceb4882,0x5944a28e,0xb3ad2208,0xa1c4c355,
		0xf0d2e9e3,0x5090d577,0xac11d7fa,0x2d1925ab, 0x1bcb66f2,0xb46496ac,0x6f2d9bc9,0xd1925ab0, 0x78602649,0x29131ab6,0x8edae952,0x0fc053c3, 0x3b6ba548,0x3f014f0c,0xedae9520,0xfc053c31
	};


/***************************************************/
__device__ __forceinline__
void rnd512(hashState *state)
{
    int i,j;
    uint32_t t[40];
    uint32_t chainv[8];
    uint32_t tmp;

#pragma unroll 8
    for(i=0;i<8;i++) {
        t[i]=0;
#pragma unroll 5
        for(j=0;j<5;j++) {
            t[i] ^= state->chainv[i+8*j];
        }
    }

    MULT2(t, 0);

#pragma unroll 5
    for(j=0;j<5;j++) {
#pragma unroll 8
        for(i=0;i<8;i++) {
            state->chainv[i+8*j] ^= t[i];
        }
    }

#pragma unroll 5
    for(j=0;j<5;j++) {
#pragma unroll 8
        for(i=0;i<8;i++) {
            t[i+8*j] = state->chainv[i+8*j];
        }
    }

#pragma unroll 5
    for(j=0;j<5;j++) {
        MULT2(state->chainv, j);
    }

#pragma unroll 5
    for(j=0;j<5;j++) {
#pragma unroll 8
        for(i=0;i<8;i++) {
            state->chainv[8*j+i] ^= t[8*((j+1)%5)+i];
        }
    }

#pragma unroll 5
    for(j=0;j<5;j++) {
#pragma unroll 8
        for(i=0;i<8;i++) {
            t[i+8*j] = state->chainv[i+8*j];
        }
    }

#pragma unroll 5
    for(j=0;j<5;j++) {
        MULT2(state->chainv, j);
    }

#pragma unroll 5
    for(j=0;j<5;j++) {
#pragma unroll 8
        for(i=0;i<8;i++) {
            state->chainv[8*j+i] ^= t[8*((j+4)%5)+i];
        }
    }

#pragma unroll 5
    for(j=0;j<5;j++) {
#pragma unroll 8
        for(i=0;i<8;i++) {
            state->chainv[i+8*j] ^= state->buffer[i];
        }
        MULT2(state->buffer, 0);
    }

#pragma unroll 8
    for(i=0;i<8;i++) {
        chainv[i] = state->chainv[i];
    }

#pragma unroll 8
    for(i=0;i<8;i++) {
        STEP(c_CNS[(2*i)],c_CNS[(2*i)+1]);
    }

#pragma unroll 8
    for(i=0;i<8;i++) {
        state->chainv[i] = chainv[i];
        chainv[i] = state->chainv[i+8];
    }

    TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],1);

#pragma unroll 8
    for(i=0;i<8;i++) {
        STEP(c_CNS[(2*i)+16],c_CNS[(2*i)+16+1]);
    }

#pragma unroll 8
    for(i=0;i<8;i++) {
        state->chainv[i+8] = chainv[i];
        chainv[i] = state->chainv[i+16];
    }

    TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],2);

#pragma unroll 8
    for(i=0;i<8;i++) {
        STEP(c_CNS[(2*i)+32],c_CNS[(2*i)+32+1]);
    }

#pragma unroll 8
    for(i=0;i<8;i++) {
        state->chainv[i+16] = chainv[i];
        chainv[i] = state->chainv[i+24];
    }

    TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],3);

#pragma unroll 8
    for(i=0;i<8;i++) {
        STEP(c_CNS[(2*i)+48],c_CNS[(2*i)+48+1]);
    }

#pragma unroll 8
    for(i=0;i<8;i++) {
        state->chainv[i+24] = chainv[i];
        chainv[i] = state->chainv[i+32];
    }

    TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],4);

#pragma unroll 8
    for(i=0;i<8;i++) {
        STEP(c_CNS[(2*i)+64],c_CNS[(2*i)+64+1]);
    }

#pragma unroll 8
    for(i=0;i<8;i++) {
        state->chainv[i+32] = chainv[i];
    }
}


__device__ __forceinline__
void Update512(hashState *state, const BitSequence *data)
{
#pragma unroll 8
    for(int i=0;i<8;i++) state->buffer[i] = cuda_swab32(((uint32_t*)data)[i]);
    rnd512(state);

#pragma unroll 8
    for(int i=0;i<8;i++) state->buffer[i] = cuda_swab32(((uint32_t*)(data+32))[i]);
    rnd512(state);
}


/***************************************************/
__device__ __forceinline__
void finalization512(hashState *state, uint32_t *b)
{
    int i,j;

    state->buffer[0] = 0x80000000;
#pragma unroll 7
    for(int i=1;i<8;i++) state->buffer[i] = 0;
    rnd512(state);

    /*---- blank round with m=0 ----*/
#pragma unroll 8
    for(i=0;i<8;i++) state->buffer[i] =0;
    rnd512(state);

#pragma unroll 8
    for(i=0;i<8;i++) {
        b[i] = 0;
#pragma unroll 5
        for(j=0;j<5;j++) {
            b[i] ^= state->chainv[i+8*j];
        }
        b[i] = cuda_swab32((b[i]));
    }

#pragma unroll 8
    for(i=0;i<8;i++) state->buffer[i]=0;
    rnd512(state);

#pragma unroll 8
    for(i=0;i<8;i++) {
        b[8+i] = 0;
#pragma unroll 5
        for(j=0;j<5;j++) {
            b[8+i] ^= state->chainv[i+8*j];
        }
        b[8 + i] = cuda_swab32((b[8 + i]));
    }
}


/***************************************************/
// Die Hash-Funktion
__global__ void x11_luffa512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint32_t *Hash = (uint32_t*)&g_hash[8 * hashPosition];

        hashState state;
#pragma unroll 40
        for(int i=0;i<40;i++) state.chainv[i] = c_IV[i];
#pragma unroll 8
        for(int i=0;i<8;i++) state.buffer[i] = 0;
        Update512(&state, (BitSequence*)Hash);
        finalization512(&state, (uint32_t*)Hash);
    }
}


// Setup Function
__host__
void x11_luffa512_cpu_init(int thr_id, uint32_t threads)
{
    CUDA_CALL_OR_RET(cudaMemcpyToSymbol(c_IV, h_IV, sizeof(h_IV), 0, cudaMemcpyHostToDevice));
    CUDA_CALL_OR_RET(cudaMemcpyToSymbol(c_CNS, h_CNS, sizeof(h_CNS), 0, cudaMemcpyHostToDevice));
}

__host__ void x11_luffa512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    const uint32_t threadsperblock = 256;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    x11_luffa512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
    //MyStreamSynchronize(NULL, order, thr_id);
}

//=================================================================

#define MULT0(a) {\
	tmp = a[7]; \
	a[7] = a[6]; \
	a[6] = a[5]; \
	a[5] = a[4]; \
	a[4] = a[3] ^ tmp; \
	a[3] = a[2] ^ tmp; \
	a[2] = a[1]; \
	a[1] = a[0] ^ tmp; \
	a[0] = tmp; \
}

__device__ __forceinline__
void STEP2(uint32_t *t, const uint2 c0, const uint2 c1){
	uint32_t temp[ 4];
	temp[ 0] = t[ 0];
	temp[ 1] = t[ 5];	
	temp[ 2] = t[0+8];
	temp[ 3] = t[8+5];
	t[ 2] ^= t[ 3];
	t[ 7] ^= t[ 4];		
	t[8+2] ^= t[8+3];
	t[8+7] ^= t[8+4];
	t[ 0] |= t[ 1];	
	t[ 5] |= t[ 6];
	t[8+0]|= t[8+1];
	t[8+5]|= t[8+6];
	t[ 1]  = ~t[ 1];
	t[ 6]  = ~t[ 6];
	t[8+1] = ~t[8+1];
	t[8+6] = ~t[8+6];
	t[ 0] ^= t[ 3];
	t[ 5] ^= t[ 4];
	t[8+0]^= t[8+3];	
	t[8+5]^= t[8+4];
	t[ 3] &= temp[ 0];
	t[ 4] &= temp[ 1];
	t[8+3]&= temp[ 2];
	t[8+4]&= temp[ 3];
	t[ 1] ^= t[ 3];	
	t[ 6] ^= t[ 4];
	t[8+1]^= t[8+3];
	t[8+6]^= t[8+4];
	t[ 3] ^= t[ 2];	
	t[ 4] ^= t[ 7];	
	t[8+3]^= t[8+2];
	t[8+4]^= t[8+7];
	t[ 2] &= t[ 0];	
	t[ 7] &= t[ 5];	
	t[8+2]&= t[8+0];
	t[8+7]&= t[8+5];
	t[ 0]  = ~t[ 0];
	t[ 5]  = ~t[ 5];
	t[8+0] = ~t[8+0];
	t[8+5] = ~t[8+5];
	t[ 2] ^= t[ 1];
	t[ 7] ^= t[ 6];
	t[8+2]^= t[8+1];
	t[8+7]^= t[8+6];
	t[ 1] |= t[ 3];	
	t[ 6] |= t[ 4];	
	t[8+1]|= t[8+3];
	t[8+6]|= t[8+4];
	
	temp[ 0] ^= t[ 1];
	temp[ 1] ^= t[ 6];
	temp[ 2] ^= t[8+1];
	temp[ 3] ^= t[8+6];
	
	t[ 3] ^= t[ 2];
	t[ 4] ^= t[ 7] ^ temp[ 0];
	t[8+3]^= t[8+2];	
	t[8+4]^= t[8+7] ^ temp[ 2];
	t[ 2] &= t[ 1];
	t[ 7]  = (t[ 7] & t[ 6]) ^ t[ 3];
	t[8+2]&= t[8+1];
	t[ 1] ^= t[ 0];
	t[8+7] = (t[8+6] & t[8+7]) ^ t[8+3];
	t[ 6] ^= t[ 5] ^ t[ 2];		
	t[8+1]^= t[8+0];	
	t[8+6]^= t[8+2]^ t[8+5];
	t[ 5]  = t[ 1] ^ temp[ 1];
	t[ 0]  = t[ 4] ^ ROTL32(temp[ 0],2);
	t[8+5] = t[8+1]^ temp[ 3];	
	t[8+0] = t[8+4]^ ROTL32(temp[ 2],2);
	t[ 1]  = t[ 5] ^ ROTL32(t[ 1],2);
	t[ 2]  = t[ 6] ^ ROTL32(t[ 2],2);
	t[8+1] = t[8+5]^ ROTL32(t[8+1],2);
	t[8+2] = t[8+6]^ ROTL32(t[8+2],2);
	t[ 3]  = t[ 7] ^ ROTL32(t[ 3],2);
	t[ 4]  = t[ 0] ^ ROTL32(t[ 4],14);
	t[8+3] = t[8+7] ^ ROTL32(t[8+3],2);
	t[8+4] = t[8+0] ^ ROTL32(t[8+4],14);
	t[ 5]  = t[ 1] ^ ROTL32(t[ 5],14);
	t[ 6]  = t[ 2] ^ ROTL32(t[ 6],14);
	t[8+5] = t[8+1] ^ ROTL32(t[8+5],14);
	t[8+6] = t[8+2] ^ ROTL32(t[8+6],14);
	t[ 7]  = t[ 3] ^ ROTL32(t[ 7],14);
	t[ 0]  = t[ 4] ^ ROTL32(t[ 0],10) ^ c0.x;
	t[8+7] = t[8+3]^ ROTL32(t[8+7],14);
	t[8+0] = t[8+4]^ ROTL32(t[8+0],10) ^ c1.x;
	t[ 1]  = t[ 5] ^ ROTL32(t[ 1],10);
	t[ 2]  = t[ 6] ^ ROTL32(t[ 2],10);
	t[8+1] = t[8+5]^ ROTL32(t[8+1],10);
	t[8+2] = t[8+6]^ ROTL32(t[8+2],10);
	t[ 3]  = t[ 7] ^ ROTL32(t[ 3],10);
	t[ 4]  = ROTL32(t[ 4],1) ^ c0.y;
	t[8+3] = t[8+7] ^ ROTL32(t[8+3],10);
	t[8+4] = ROTL32(t[8+4],1) ^ c1.y;
	t[ 5]  = ROTL32(t[ 5],1);
	t[ 6]  = ROTL32(t[ 6],1);
	t[8+5] = ROTL32(t[8+5],1);
	t[8+6] = ROTL32(t[8+6],1);
	t[ 7]  = ROTL32(t[ 7],1);
	t[8+7] = ROTL32(t[8+7],1);
}

__device__ __forceinline__
void STEP1(uint32_t *t, const uint2 c){
	uint32_t temp[ 2];
	temp[ 0] = t[ 0];			temp[ 1] = t[ 5];
	t[ 2] ^= t[ 3];				t[ 7] ^= t[ 4];
	t[ 0] |= t[ 1];				t[ 5] |= t[ 6];
	t[ 1]  = ~t[ 1];			t[ 6]  = ~t[ 6];
	t[ 0] ^= t[ 3];				t[ 5] ^= t[ 4];
	t[ 3] &= temp[ 0];			t[ 4] &= temp[ 1];
	t[ 1] ^= t[ 3];				t[ 6] ^= t[ 4];
	t[ 3] ^= t[ 2];				t[ 4] ^= t[ 7];
	t[ 2] &= t[ 0];				t[ 7] &= t[ 5];
	t[ 0]  = ~t[ 0];			t[ 5]  = ~t[ 5];
	t[ 2] ^= t[ 1];				t[ 7] ^= t[ 6];
	t[ 1] |= t[ 3];				t[ 6] |= t[ 4];
	temp[ 0] ^= t[ 1];			temp[ 1] ^= t[ 6];
	t[ 3] ^= t[ 2];				t[ 4] ^= t[ 7] ^ temp[ 0];
	t[ 2] &= t[ 1];				t[ 7]  = (t[ 7] & t[ 6]) ^ t[ 3];
	t[ 1] ^= t[ 0];				t[ 6] ^= t[ 5] ^ t[ 2];
	t[ 5]  = t[ 1] ^ temp[ 1];		t[ 0]  = t[ 4] ^ ROTL32(temp[ 0],2);
	t[ 1]  = t[ 5] ^ ROTL32(t[ 1],2);	t[ 2]  = t[ 6] ^ ROTL32(t[ 2],2);
	t[ 3]  = t[ 7] ^ ROTL32(t[ 3],2);	t[ 4]  = t[ 0] ^ ROTL32(t[ 4],14);
	t[ 5]  = t[ 1] ^ ROTL32(t[ 5],14);	t[ 6]  = t[ 2] ^ ROTL32(t[ 6],14);
	t[ 7]  = t[ 3] ^ ROTL32(t[ 7],14);	t[ 0]  = t[ 4] ^ ROTL32(t[ 0],10) ^ c.x;
	t[ 1]  = t[ 5] ^ ROTL32(t[ 1],10);	t[ 2]  = t[ 6] ^ ROTL32(t[ 2],10);
	t[ 3]  = t[ 7] ^ ROTL32(t[ 3],10);	t[ 4]  = ROTL32(t[ 4],1) ^ c.y;
	t[ 5]  = ROTL32(t[ 5],1);		t[ 6]  = ROTL32(t[ 6],1);
						t[ 7]  = ROTL32(t[ 7],1);
}

__device__
static void rnd512(uint32_t *const __restrict__ statebuffer, uint32_t *const __restrict__ statechainv){
	uint32_t t[40];
	uint32_t tmp;

	tmp = statechainv[ 7] ^ statechainv[7 + 8] ^ statechainv[7 +16] ^ statechainv[7 +24] ^ statechainv[7 +32];
	t[7] = statechainv[ 6] ^ statechainv[6 + 8] ^ statechainv[6 +16] ^ statechainv[6 +24] ^ statechainv[6 +32];
	t[6] = statechainv[ 5] ^ statechainv[5 + 8] ^ statechainv[5 +16] ^ statechainv[5 +24] ^ statechainv[5 +32];
	t[5] = statechainv[ 4] ^ statechainv[4 + 8] ^ statechainv[4 +16] ^ statechainv[4 +24] ^ statechainv[4 +32];
	t[4] = statechainv[ 3] ^ statechainv[3 + 8] ^ statechainv[3 +16] ^ statechainv[3 +24] ^ statechainv[3 +32] ^ tmp;
	t[3] = statechainv[ 2] ^ statechainv[2 + 8] ^ statechainv[2 +16] ^ statechainv[2 +24] ^ statechainv[2 +32] ^ tmp;
	t[2] = statechainv[ 1] ^ statechainv[1 + 8] ^ statechainv[1 +16] ^ statechainv[1 +24] ^ statechainv[1 +32];
	t[1] = statechainv[ 0] ^ statechainv[0 + 8] ^ statechainv[0 +16] ^ statechainv[0 +24] ^ statechainv[0 +32] ^ tmp;
	t[0] = tmp;

	#pragma unroll 8
	for(int i=0;i<8;i++){
		statechainv[i] ^= t[i];
    }

	#pragma unroll 4
	for (int j=1;j<5;j++) {
		#pragma unroll 8
		for(int i=0;i<8;i++){
			statechainv[i+(j<<3)] ^= t[i];
        }

		#pragma unroll 8
		for(int i=0;i<8;i++){
			t[i+(j<<3)] = statechainv[i+(j<<3)];
        }
	}

	#pragma unroll 8
	for(int i=0;i<8;i++){
		t[i] = statechainv[i];
    }

	MULT0(statechainv);

	#pragma unroll 4
	for (int j=1;j<5;j++){
		MULT2(statechainv, j);
    }

	#pragma unroll 5
	for (int j=0;j<5;j++){
		#pragma unroll 8
		for(int i=0;i<8;i++){
			statechainv[i+8*j] ^= t[i+(8*((j+1)%5))];
        }
    }

	#pragma unroll 5
	for (int j=0;j<5;j++){
		*(uint2x4*)&t[8*j] = *(uint2x4*)&statechainv[8*j];
    }

	MULT0(statechainv);
	#pragma unroll 4
	for (int j=1;j<5;j++){
		MULT2(statechainv, j);
    }

	#pragma unroll 5
	for (int j=0;j<5;j++){
		*(uint2x4*)&statechainv[8*j] ^= *(uint2x4*)&t[8*((j+4)%5)];
    }

	#pragma unroll 5
	for (int j=0;j<5;j++) {
		*(uint2x4*)&statechainv[8*j] ^= *(uint2x4*)statebuffer;
		MULT0(statebuffer);
	}

	TWEAK(statechainv[12], statechainv[13], statechainv[14], statechainv[15], 1);
	TWEAK(statechainv[20], statechainv[21], statechainv[22], statechainv[23], 2);
	TWEAK(statechainv[28], statechainv[29], statechainv[30], statechainv[31], 3);
	TWEAK(statechainv[36], statechainv[37], statechainv[38], statechainv[39], 4);

	for (int i = 0; i<8; i++){
		STEP2( statechainv    ,*(uint2*)&c_CNS[(2 * i) +  0], *(uint2*)&c_CNS[(2 * i) + 16]);
		STEP2(&statechainv[16],*(uint2*)&c_CNS[(2 * i) + 32], *(uint2*)&c_CNS[(2 * i) + 48]);
		STEP1(&statechainv[32],*(uint2*)&c_CNS[(2 * i) + 64]);
	}
}

__device__
static void rnd512_first(uint32_t *const __restrict__ state, uint32_t *const __restrict__ buffer)
{
	#pragma unroll 5
	for (int j = 0; j<5; j++) {
		uint32_t tmp;
		#pragma unroll 8
		for(int i=0;i<8;i++)
			state[i+(j<<3)] ^= buffer[i];
		MULT0(buffer);
	}
	TWEAK(state[12], state[13], state[14], state[15], 1);
	TWEAK(state[20], state[21], state[22], state[23], 2);
	TWEAK(state[28], state[29], state[30], state[31], 3);
	TWEAK(state[36], state[37], state[38], state[39], 4);
	
	for (int i = 0; i<8; i++) {
		STEP2(&state[ 0],*(uint2*)&c_CNS[(2 * i) +  0],*(uint2*)&c_CNS[(2 * i) + 16]);
		STEP2(&state[16],*(uint2*)&c_CNS[(2 * i) + 32],*(uint2*)&c_CNS[(2 * i) + 48]);
		STEP1(&state[32],*(uint2*)&c_CNS[(2 * i) + 64]);
	}
}

__device__ __forceinline__
static void rnd512_nullhash(uint32_t *const __restrict__ state){

	uint32_t t[40];
	uint32_t tmp;

	tmp = state[ 7] ^ state[7 + 8] ^ state[7 +16] ^ state[7 +24] ^ state[7 +32];
	t[7] = state[ 6] ^ state[6 + 8] ^ state[6 +16] ^ state[6 +24] ^ state[6 +32];
	t[6] = state[ 5] ^ state[5 + 8] ^ state[5 +16] ^ state[5 +24] ^ state[5 +32];
	t[5] = state[ 4] ^ state[4 + 8] ^ state[4 +16] ^ state[4 +24] ^ state[4 +32];
	t[4] = state[ 3] ^ state[3 + 8] ^ state[3 +16] ^ state[3 +24] ^ state[3 +32] ^ tmp;
	t[3] = state[ 2] ^ state[2 + 8] ^ state[2 +16] ^ state[2 +24] ^ state[2 +32] ^ tmp;
	t[2] = state[ 1] ^ state[1 + 8] ^ state[1 +16] ^ state[1 +24] ^ state[1 +32];
	t[1] = state[ 0] ^ state[0 + 8] ^ state[0 +16] ^ state[0 +24] ^ state[0 +32] ^ tmp;
	t[0] = tmp;
	
	#pragma unroll 5
	for (int j = 0; j<5; j++){
		*(uint2x4*)&state[8*j] ^= *(uint2x4*)t;
	}
	
	#pragma unroll 5
	for (int j = 0; j<5; j++){
		*(uint2x4*)&t[8*j] = *(uint2x4*)&state[8*j];
	}
	#pragma unroll 5
	for (int j = 0; j<5; j++) {
		MULT2(state, j);
	}

	#pragma unroll 5
	for (int j = 0; j<5; j++) {
		*(uint2x4*)&state[8*j] ^= *(uint2x4*)&t[8 * ((j + 1) % 5)];
	}

	#pragma unroll 5
	for (int j = 0; j<5; j++) {
		*(uint2x4*)&t[8*j] = *(uint2x4*)&state[8*j];
	}

	#pragma unroll 5
	for (int j = 0; j<5; j++) {
		MULT2(state, j);
	}

	#pragma unroll 5
	for (int j = 0; j<5; j++) {
		#pragma unroll 8
		for(int i=0;i<8;i++)
			state[i+8*j] ^= t[i+(8 * ((j + 4) % 5))];
	}

	TWEAK(state[12], state[13], state[14], state[15], 1);
	TWEAK(state[20], state[21], state[22], state[23], 2);
	TWEAK(state[28], state[29], state[30], state[31], 3);
	TWEAK(state[36], state[37], state[38], state[39], 4);
	
	for (int i = 0; i<8; i++) {
		STEP2(&state[ 0],*(uint2*)&c_CNS[(2 * i) +  0],*(uint2*)&c_CNS[(2 * i) + 16]);
		STEP2(&state[16],*(uint2*)&c_CNS[(2 * i) + 32],*(uint2*)&c_CNS[(2 * i) + 48]);
		STEP1(&state[32],*(uint2*)&c_CNS[(2 * i) + 64]);
	}
}

__device__ __forceinline__
static void Update512(uint32_t *statebuffer, uint32_t *statechainv, const uint32_t *data)
{
	#pragma unroll
	for (int i = 0; i < 8; i++) statebuffer[i] = cuda_swab32(data[i]);
	rnd512_first(statechainv, statebuffer);

	#pragma unroll
	for (int i = 0; i < 8; i++) statebuffer[i] = cuda_swab32(data[i + 8]);
	rnd512(statebuffer, statechainv);
}

__device__ __forceinline__
static void finalization512(uint32_t *statebuffer, uint32_t *statechainv, uint32_t *outHash)
{
    rnd512_nullhash(statechainv);
    rnd512_nullhash(statechainv);

	statebuffer[0] = 0x80000000;
	#pragma unroll 7
	for(int i=1;i<8;i++) statebuffer[i] = 0;
	rnd512(statebuffer, statechainv);
	rnd512_nullhash(statechainv);

	*(uint2x4*)&outHash[ 0] = swapvec(*(uint2x4*)&statechainv[ 0] ^ *(uint2x4*)&statechainv[ 8] ^ *(uint2x4*)&statechainv[16] ^ *(uint2x4*)&statechainv[24] ^ *(uint2x4*)&statechainv[32]);
	rnd512_nullhash(statechainv);
	*(uint2x4*)&outHash[ 8] = swapvec(*(uint2x4*)&statechainv[ 0] ^ *(uint2x4*)&statechainv[ 8] ^ *(uint2x4*)&statechainv[16] ^ *(uint2x4*)&statechainv[24] ^ *(uint2x4*)&statechainv[32]);
}

#define TPB_L 384

__global__
__launch_bounds__(TPB_L,2)
void x11_luffa512_gpu_hash_128(uint32_t threads, uint32_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t statechainv[40] = {
			0x8bb0a761, 0xc2e4aa8b, 0x2d539bc9, 0x381408f8,
			0x478f6633, 0x255a46ff, 0x581c37f7, 0x601c2e8e,
			0x266c5f9d, 0xc34715d8, 0x8900670e, 0x51a540be,
			0xe4ce69fb, 0x5089f4d4, 0x3cc0a506, 0x609bcb02,
			0xa4e3cd82, 0xd24fd6ca, 0xc0f196dc, 0xcf41eafe,
			0x0ff2e673, 0x303804f2, 0xa7b3cd48, 0x677addd4,
			0x66e66a8a, 0x2303208f, 0x486dafb4, 0xc0d37dc6,
			0x634d15af, 0xe5af6747, 0x10af7e38, 0xee7e6428,
			0x01262e5d, 0xc92c2e64, 0x82fee966, 0xcea738d3,
			0x867de2b0, 0xe0714818, 0xda6e831f, 0xa7062529
		};

		uint32_t statebuffer[8];
		uint32_t *const Hash = &g_hash[thread * 16U];

		Update512(statebuffer, statechainv, Hash);
		finalization512(statebuffer, statechainv, Hash);
	}
}

__host__
void x11_luffa512_cpu_hash_128(int thr_id, uint32_t threads,uint32_t *d_hash)
{
    const uint32_t threadsperblock = TPB_L;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    x11_luffa512_gpu_hash_128<<<grid, block>>>(threads,d_hash);
}
