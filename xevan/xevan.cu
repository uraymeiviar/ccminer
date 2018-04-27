/**
 * Xevan algorithm double(X15 + sha512 + haval256)
 */

 extern "C" {
    #include "sph/sph_blake.h"
    #include "sph/sph_bmw.h"
    #include "sph/sph_groestl.h"
    #include "sph/sph_skein.h"
    #include "sph/sph_jh.h"
    #include "sph/sph_keccak.h"

    #include "sph/sph_luffa.h"
    #include "sph/sph_cubehash.h"
    #include "sph/sph_shavite.h"
    #include "sph/sph_simd.h"
    #include "sph/sph_echo.h"

    #include "sph/sph_hamsi.h"
    #include "sph/sph_fugue.h"

    #include "sph/sph_shabal.h"
    #include "sph/sph_whirlpool.h"

    #include "sph/sph_sha2.h"
    #include "sph/sph_haval.h"
}

#include "miner.h"
#include "cuda_helper.h"

__constant__ uint32_t pTarget[8]; // 32 bytes

// store MAX_GPUS device arrays of 8 nonces
static uint32_t* h_resNonces[MAX_GPUS] = { NULL };
static uint32_t* d_resNonces[MAX_GPUS] = { NULL };
static __thread bool init_done = false;

__host__
void xevan_check_cpu_init(int thr_id, uint32_t threads)
{
    CUDA_CALL_OR_RET(cudaMalloc(&d_resNonces[thr_id], sizeof(uint32_t) * MAX_NONCES));
    CUDA_SAFE_CALL(cudaMallocHost(&h_resNonces[thr_id], sizeof(uint32_t) * MAX_NONCES));
    init_done = true;
}

__host__
void xevan_check_cpu_free(int thr_id)
{
	if (!init_done) return;
	cudaFree(d_resNonces[thr_id]);
	cudaFreeHost(h_resNonces[thr_id]);
	d_resNonces[thr_id] = NULL;
	h_resNonces[thr_id] = NULL;
	init_done = false;
}

__host__
void xevan_check_cpu_setTarget(const void *ptarget)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(pTarget, ptarget, 32, 0, cudaMemcpyHostToDevice));
}

__global__ __launch_bounds__(512, 4)
void xevan_checkhash_128(uint32_t threads, uint32_t startNounce, const uint64_t *hash, uint32_t *resNonces)
{
    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    uint32_t validCount = 0;
	if (thread < threads)
	{
        const uint32_t *inpHash = (const uint32_t *)&hash[thread<<3];

		if (resNonces[thread % MAX_NONCES] == UINT32_MAX) {
            const uint64_t checkhash = *(const uint64_t*)&inpHash[6];
            const uint64_t target    = *(const uint64_t*)&pTarget[6];
            if (checkhash <= target){
                resNonces[thread % MAX_NONCES] = (startNounce + thread);
            }
		}
	}
}

__host__
void xevan_check_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash)
{
    cudaMemset(d_resNonces[thr_id], 0xff, sizeof(uint32_t)*MAX_NONCES);

	const uint32_t threadsperblock = 512;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	xevan_checkhash_128 <<<grid, block>>> (threads, startNounce, (uint64_t*)d_inputHash, d_resNonces[thr_id]);
	cudaThreadSynchronize();

	cudaMemcpy(h_resNonces[thr_id], d_resNonces[thr_id], sizeof(uint32_t)*MAX_NONCES, cudaMemcpyDeviceToHost);
}

static uint32_t *d_hash[MAX_GPUS];

extern void xevan_blake512_cpu_hash_80_bmw_128(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void xevan_blake512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_outputHash);
extern void xevan_blake512_cpu_setBlock_80(int thr_id, uint32_t *pdata);
extern void xevan_skein512_cpu_hash_128(int thr_id,uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void xevan_jh512_cpu_hash_128_keccak_128(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void xevan_hamsi512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void xevan_shavite512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash);
extern int  xevan_simd512_cpu_init(int thr_id, uint32_t threads);
extern void xevan_simd512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void xevan_simd512_cpu_free(int thr_id);
extern void xevan_whirlpool_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void xevan_whirlpool_cpu_init(int thr_id, uint32_t threads);
extern void xevan_luffa512_cpu_hash_128(int thr_id, uint32_t threads,uint32_t *d_hash);
extern void xevan_echo512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void xevan_fugue512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void xevan_shabal512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void xevan_groestl512_cpu_hash_128(int thr_id, uint32_t threads,  uint32_t *d_hash);
extern void xevan_cubehash512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void xevan_sha512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void xevan_haval512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void xevan_haval512_cpu_hash_128_final(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void xevan_bmw512_cpu_hash_64x(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash);

extern "C" void xevanhash(void *output, const void *input)
{
    uint32_t _ALIGN(64) hash[32]; // 128 bytes required
    const int dataLen = 128;

    sph_blake512_context     ctx_blake;
    sph_bmw512_context       ctx_bmw;
    sph_groestl512_context   ctx_groestl;
    sph_skein512_context     ctx_skein;
    sph_jh512_context        ctx_jh;
    sph_keccak512_context    ctx_keccak;
    sph_luffa512_context     ctx_luffa;
    sph_cubehash512_context  ctx_cubehash;
    sph_shavite512_context   ctx_shavite;
    sph_simd512_context      ctx_simd;
    sph_echo512_context      ctx_echo;
    sph_hamsi512_context     ctx_hamsi;
    sph_fugue512_context     ctx_fugue;
    sph_shabal512_context    ctx_shabal;
    sph_whirlpool_context    ctx_whirlpool;
    sph_sha512_context       ctx_sha512;
    sph_haval256_5_context   ctx_haval;

    sph_blake512_init(&ctx_blake);
    sph_blake512(&ctx_blake, input, 80);
    sph_blake512_close(&ctx_blake, hash);

    memset(&hash[16], 0, 64);

    sph_bmw512_init(&ctx_bmw);
    sph_bmw512(&ctx_bmw, hash, dataLen);
    sph_bmw512_close(&ctx_bmw, hash);

    sph_groestl512_init(&ctx_groestl);
    sph_groestl512(&ctx_groestl, hash, dataLen);
    sph_groestl512_close(&ctx_groestl, hash);

    sph_skein512_init(&ctx_skein);
    sph_skein512(&ctx_skein, hash, dataLen);
    sph_skein512_close(&ctx_skein, hash);

    sph_jh512_init(&ctx_jh);
    sph_jh512(&ctx_jh, hash, dataLen);
    sph_jh512_close(&ctx_jh, hash);

    sph_keccak512_init(&ctx_keccak);
    sph_keccak512(&ctx_keccak, hash, dataLen);
    sph_keccak512_close(&ctx_keccak, hash);

    sph_luffa512_init(&ctx_luffa);
    sph_luffa512(&ctx_luffa, hash, dataLen);
    sph_luffa512_close(&ctx_luffa, hash);

    sph_cubehash512_init(&ctx_cubehash);
    sph_cubehash512(&ctx_cubehash, hash, dataLen);
    sph_cubehash512_close(&ctx_cubehash, hash);

    sph_shavite512_init(&ctx_shavite);
    sph_shavite512(&ctx_shavite, hash, dataLen);
    sph_shavite512_close(&ctx_shavite, hash);

    sph_simd512_init(&ctx_simd);
    sph_simd512(&ctx_simd, hash, dataLen);
    sph_simd512_close(&ctx_simd, hash);

    sph_echo512_init(&ctx_echo);
    sph_echo512(&ctx_echo, hash, dataLen);
    sph_echo512_close(&ctx_echo, hash);
            
    sph_hamsi512_init(&ctx_hamsi);
    sph_hamsi512(&ctx_hamsi, hash, dataLen);
    sph_hamsi512_close(&ctx_hamsi, hash);

    sph_fugue512_init(&ctx_fugue);
    sph_fugue512(&ctx_fugue, hash, dataLen);
    sph_fugue512_close(&ctx_fugue, hash);

    sph_shabal512_init(&ctx_shabal);
    sph_shabal512(&ctx_shabal, hash, dataLen);
    sph_shabal512_close(&ctx_shabal, hash);

    sph_whirlpool_init(&ctx_whirlpool);
    sph_whirlpool(&ctx_whirlpool, hash, dataLen);
    sph_whirlpool_close(&ctx_whirlpool, hash);

    sph_sha512_init(&ctx_sha512);
    sph_sha512(&ctx_sha512,(const void*) hash, dataLen);
    sph_sha512_close(&ctx_sha512,(void*) hash);

    sph_haval256_5_init(&ctx_haval);
    sph_haval256_5(&ctx_haval,(const void*) hash, dataLen);
    sph_haval256_5_close(&ctx_haval, hash);

    memset(&hash[8], 0, dataLen - 32);

    sph_blake512_init(&ctx_blake);
    sph_blake512(&ctx_blake, hash, dataLen);
    sph_blake512_close(&ctx_blake, hash);

    sph_bmw512_init(&ctx_bmw);
    sph_bmw512(&ctx_bmw, hash, dataLen);
    sph_bmw512_close(&ctx_bmw, hash);

    sph_groestl512_init(&ctx_groestl);
    sph_groestl512(&ctx_groestl, hash, dataLen);
    sph_groestl512_close(&ctx_groestl, hash);

    sph_skein512_init(&ctx_skein);
    sph_skein512(&ctx_skein, hash, dataLen);
    sph_skein512_close(&ctx_skein, hash);

    sph_jh512_init(&ctx_jh);
    sph_jh512(&ctx_jh, hash, dataLen);
    sph_jh512_close(&ctx_jh, hash);

    sph_keccak512_init(&ctx_keccak);
    sph_keccak512(&ctx_keccak, hash, dataLen);
    sph_keccak512_close(&ctx_keccak, hash);

    sph_luffa512_init(&ctx_luffa);
    sph_luffa512(&ctx_luffa, hash, dataLen);
    sph_luffa512_close(&ctx_luffa, hash);

    sph_cubehash512_init(&ctx_cubehash);
    sph_cubehash512(&ctx_cubehash, hash, dataLen);
    sph_cubehash512_close(&ctx_cubehash, hash);

    sph_shavite512_init(&ctx_shavite);
    sph_shavite512(&ctx_shavite, hash, dataLen);
    sph_shavite512_close(&ctx_shavite, hash);
            
    sph_simd512_init(&ctx_simd);
    sph_simd512(&ctx_simd, hash, dataLen);
    sph_simd512_close(&ctx_simd, hash);

    sph_echo512_init(&ctx_echo);
    sph_echo512(&ctx_echo, hash, dataLen);
    sph_echo512_close(&ctx_echo, hash);
    
    sph_hamsi512_init(&ctx_hamsi);
    sph_hamsi512(&ctx_hamsi, hash, dataLen);
    sph_hamsi512_close(&ctx_hamsi, hash);

    sph_fugue512_init(&ctx_fugue);
    sph_fugue512(&ctx_fugue, hash, dataLen);
    sph_fugue512_close(&ctx_fugue, hash);

    sph_shabal512_init(&ctx_shabal);
    sph_shabal512(&ctx_shabal, hash, dataLen);
    sph_shabal512_close(&ctx_shabal, hash);

    sph_whirlpool_init(&ctx_whirlpool);
    sph_whirlpool(&ctx_whirlpool, hash, dataLen);
    sph_whirlpool_close(&ctx_whirlpool, hash);

    sph_sha512_init(&ctx_sha512);
    sph_sha512(&ctx_sha512,(const void*) hash, dataLen);
    sph_sha512_close(&ctx_sha512,(void*) hash);

    sph_haval256_5_init(&ctx_haval);
    sph_haval256_5(&ctx_haval,(const void*) hash, dataLen);
    sph_haval256_5_close(&ctx_haval, hash);

    memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_xevan(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done){

    int dev_id = device_map[thr_id];

    uint32_t *pdata = work->data;
    uint32_t *ptarget = work->target;
    const uint32_t first_nonce = pdata[19];

    uint32_t default_throughput;
    if(device_sm[dev_id]<=500) default_throughput = 1<<20;
    else if(device_sm[dev_id]<=520) default_throughput = 1<<21;
    else if(device_sm[dev_id]>520) default_throughput = (1<<22) + (1<<21);
    default_throughput = 1<<20;
    if((strstr(device_name[dev_id], "1070")))default_throughput = 1<<20;
    if((strstr(device_name[dev_id], "1080")))default_throughput = 1<<20;
    
    uint32_t throughput = cuda_default_throughput(thr_id, default_throughput); // 19=256*256*8;
    if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

    throughput&=0xFFFFFF70; //multiples of 128 due to simd_echo kernel

    if (opt_benchmark)
        ((uint32_t*)ptarget)[7] = 0xff;

    if (!init[thr_id])
    {
        cudaSetDevice(device_map[thr_id]);
        if (opt_cudaschedule == -1 && gpu_threads == 1) {
            cudaDeviceReset();
            // reduce cpu usage
            cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        }
        gpulog(LOG_INFO,thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

        xevan_simd512_cpu_init(thr_id, throughput);

        CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput), 0);

        xevan_check_cpu_init(thr_id, throughput);
        init[thr_id] = true;
    }

    xevan_check_cpu_setTarget(ptarget);

    uint32_t _ALIGN(64) endiandata[20];
    for (int k=0; k < 20; k++)
        be32enc(&endiandata[k], pdata[k]);

    xevan_blake512_cpu_setBlock_80(thr_id, endiandata);
    int warn = 0;
    do {
        xevan_blake512_cpu_hash_80_bmw_128(thr_id, throughput, pdata[19], d_hash[thr_id]);
        xevan_groestl512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_skein512_cpu_hash_128(thr_id, throughput, NULL, d_hash[thr_id]);
        xevan_jh512_cpu_hash_128_keccak_128(thr_id, throughput, NULL, d_hash[thr_id]);
        xevan_luffa512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_cubehash512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_shavite512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_simd512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_echo512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_hamsi512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_fugue512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_shabal512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_whirlpool_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_sha512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_haval512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);

        xevan_blake512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_bmw512_cpu_hash_64x(thr_id, throughput, NULL, d_hash[thr_id]);
        xevan_groestl512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_skein512_cpu_hash_128(thr_id, throughput, NULL, d_hash[thr_id]);
        xevan_jh512_cpu_hash_128_keccak_128(thr_id, throughput,  NULL, d_hash[thr_id]);
        xevan_luffa512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_cubehash512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_shavite512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_simd512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_echo512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_hamsi512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_fugue512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_shabal512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_whirlpool_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_sha512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]);
        xevan_haval512_cpu_hash_128_final(thr_id, throughput, d_hash[thr_id]);

        *hashes_done = pdata[19] - first_nonce + throughput;
        xevan_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);

        work->valid_nonces = 0;
        for(int n=0 ; n<MAX_NONCES ; n++){
            uint32_t nonce = h_resNonces[thr_id][n];
            if (nonce != UINT32_MAX)
            {
                const uint32_t Htarg = ptarget[7];
                uint32_t _ALIGN(64) vhash[8];
                work->nonces[work->valid_nonces] = nonce;
                be32enc(&endiandata[19], nonce);
                xevanhash(vhash, endiandata);
    
                if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
                    bn_set_target_ratio(work, vhash, work->valid_nonces);
                    if(work->nonces[work->valid_nonces] + 1 > pdata[19]){
                        pdata[19] = work->nonces[work->valid_nonces] + 1; 
                    }
                    work->valid_nonces++;
                }
                else if (vhash[7] > Htarg) {
                    // x11+ coins could do some random error, but not on retry
                    gpu_increment_reject(thr_id);
                    if (!warn) {
                        warn++;
                        pdata[19] = work->nonces[work->valid_nonces] + 1;
                        continue;
                    } else {
                        if (!opt_quiet)
                            gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU! ( %08x > %08x )", nonce,vhash[7],Htarg);
                        warn = 0;
                    }
                }
            }
        }

        if(work->valid_nonces > 0){
            if (work->valid_nonces > 1)
			    applog(LOG_WARNING, "Found multiple nonces : %d, from GPU #%d (%s)", work->valid_nonces, thr_id, device_name[dev_id]);
            return work->valid_nonces;
        }

        if ((uint64_t)throughput + pdata[19] >= max_nonce) {
            pdata[19] = max_nonce;
            break;
        }

        pdata[19] += throughput;

    } while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

    *hashes_done = pdata[19] - first_nonce;
    return 0;
}

// cleanup
extern "C" void free_xevan(int thr_id)
{
    if (!init[thr_id])
        return;

    cudaDeviceSynchronize();
    cudaFree(d_hash[thr_id]);

    xevan_simd512_cpu_free(thr_id);
    xevan_check_cpu_free(thr_id);

    cudaDeviceSynchronize();
    init[thr_id] = false;
}
