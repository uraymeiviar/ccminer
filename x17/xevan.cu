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
    
    #define NBN 2

    static uint32_t *d_hash[MAX_GPUS];
    static uint32_t *d_resNonce[MAX_GPUS];
    static uint32_t *h_resNonce[MAX_GPUS];
    
    extern void quark_blake512_cpu_init(int thr_id, uint32_t threads);
    extern void quark_blake512_cpu_free(int thr_id);
    extern void quark_blake512_cpu_hash_80_bmw_128(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
    extern void quark_blake512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_outputHash);
    extern void xevan_blake512_cpu_setBlock_80(int thr_id, uint32_t *pdata);

    extern void quark_skein512_cpu_init(int thr_id, uint32_t threads);
    extern void quark_skein512_cpu_hash_128(int thr_id,uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash);

    extern void quark_jh512_cpu_hash_128_keccak_128(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash);

    extern void quark_keccak512_cpu_init(int thr_id, uint32_t threads);
    extern void quark_keccak512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash);

    extern void xevan_hamsi512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash);

    extern void xevan_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

    extern int  xevan_simd512_cpu_init(int thr_id, uint32_t threads);
    extern void xevan_simd512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash);
    extern void xevan_simd512_cpu_free(int thr_id);

    extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int mode);
    extern void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
    extern void x15_whirlpool_cpu_free(int thr_id);

    extern void x11_luffa512_cpu_hash_128(int thr_id, uint32_t threads,uint32_t *d_hash);

    extern void xevan_echo512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash);

    extern void xevan_fugue512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash);
    extern void xevan_shabal512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash);

    extern void quark_groestl512_cpu_hash_128(int thr_id, uint32_t threads,  uint32_t *d_hash);
    extern void groestl512_cpu_init(int thr_id, uint32_t threads);

    extern void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

    extern void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

    extern void x17_haval512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash);
    extern void x17_haval512_cpu_hash_128_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *resNonce, uint64_t target);

    extern void quark_bmw512_cpu_hash_64x(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash);

    // X17 CPU Hash (Validation)
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
        /*
        sph_whirlpool_init(&ctx_whirlpool);
        sph_whirlpool(&ctx_whirlpool, hash, dataLen);
        sph_whirlpool_close(&ctx_whirlpool, hash);
    
        sph_sha512_init(&ctx_sha512);
        sph_sha512(&ctx_sha512,(const void*) hash, dataLen);
        sph_sha512_close(&ctx_sha512,(void*) hash);
    */
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
        /*
        sph_whirlpool_init(&ctx_whirlpool);
        sph_whirlpool(&ctx_whirlpool, hash, dataLen);
        sph_whirlpool_close(&ctx_whirlpool, hash);
    
        sph_sha512_init(&ctx_sha512);
        sph_sha512(&ctx_sha512,(const void*) hash, dataLen);
        sph_sha512_close(&ctx_sha512,(void*) hash);
        */
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

            /*
            quark_blake512_cpu_init(thr_id, throughput);
            x11_simd512_cpu_init(thr_id, throughput);
            quark_jh512_cpu_init(thr_id, throughput);
            quark_keccak512_cpu_init(thr_id, throughput);
            x15_whirlpool_cpu_init(thr_id, throughput, 0);
            x11_luffa512_cpu_init(thr_id, throughput);
            x11_echo512_cpu_init(thr_id, throughput);
            
            x13_fugue512_cpu_init(thr_id, throughput);
            x14_shabal512_cpu_init(thr_id, throughput);
            x17_haval256_cpu_init(thr_id, throughput);
*/
            quark_skein512_cpu_init(thr_id, throughput);
            groestl512_cpu_init(thr_id, throughput);
            xevan_simd512_cpu_init(thr_id, throughput);

            CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 8 * sizeof(uint64_t) * throughput));
            CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
            h_resNonce[thr_id] = (uint32_t*) malloc(NBN  * 8 * sizeof(uint32_t));
            if(h_resNonce[thr_id] == NULL){
                gpulog(LOG_ERR,thr_id,"Host memory allocation failed");
                exit(EXIT_FAILURE);
            }		
            init[thr_id] = true;
        }
    
        uint32_t _ALIGN(64) endiandata[20];
        for (int k=0; k < 20; k++)
            be32enc(&endiandata[k], pdata[k]);
    
        xevan_blake512_cpu_setBlock_80(thr_id, endiandata);
        cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));

        int warn = 0;
    
        do {
            int order = 0;

            // Hash with CUDA
            quark_blake512_cpu_hash_80_bmw_128(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
            quark_groestl512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]); order++;
            quark_skein512_cpu_hash_128(thr_id, throughput, NULL, d_hash[thr_id]); order++;
            quark_jh512_cpu_hash_128_keccak_128(thr_id, throughput, NULL, d_hash[thr_id]); order++;
            x11_luffa512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]); order++;
            x11_cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
            xevan_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
            xevan_simd512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]); order++;
            xevan_echo512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]); order++;
            xevan_hamsi512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]); order++;
            xevan_fugue512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]); order++;
            xevan_shabal512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]); order++;
            /*
            x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
            x17_sha512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
            */
            x17_haval512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]); order++;

            quark_blake512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]); order++;
            quark_bmw512_cpu_hash_64x(thr_id, throughput, NULL, d_hash[thr_id]); order++;
            quark_groestl512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]); order++;
            quark_skein512_cpu_hash_128(thr_id, throughput, NULL, d_hash[thr_id]); order++;
            quark_jh512_cpu_hash_128_keccak_128(thr_id, throughput,  NULL, d_hash[thr_id]); order++;
            x11_luffa512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]); order++;
            x11_cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
            xevan_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
            xevan_simd512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]); order++;
            xevan_echo512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]); order++;
            xevan_hamsi512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]); order++;
            xevan_fugue512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]); order++;
            xevan_shabal512_cpu_hash_128(thr_id, throughput, d_hash[thr_id]); order++;
             /*
            x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
            x17_sha512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
                */
            x17_haval512_cpu_hash_128_final(thr_id, throughput, d_hash[thr_id],d_resNonce[thr_id],*(uint64_t*)&ptarget[6]); order++;

            cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);

            if (h_resNonce[thr_id][0] != UINT32_MAX){
                const uint32_t Htarg = ptarget[7];
                const uint32_t startNounce = pdata[19];
                uint32_t vhash64[8];
                be32enc(&endiandata[19], startNounce + h_resNonce[thr_id][0]);
                xevanhash(vhash64, endiandata);

                if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
                    int res = 1;
                    *hashes_done = pdata[19] - first_nonce + throughput + 1;
                    work_set_target_ratio(work, vhash64);
                    pdata[19] = startNounce + h_resNonce[thr_id][0];
                    if (h_resNonce[thr_id][1] != UINT32_MAX) {
                        pdata[21] = startNounce+h_resNonce[thr_id][1];
                        if(!opt_quiet)
                            gpulog(LOG_BLUE,dev_id,"Found 2nd nonce: %08x", pdata[21]);
                        be32enc(&endiandata[19], pdata[21]);
                        xevanhash(vhash64, endiandata);
                        if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio[0]){
                            work_set_target_ratio(work, vhash64);
                            xchg(pdata[19],pdata[21]);
                        }
                        res++;
                    }
                    return res;
                }
                else {
                    gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", h_resNonce[thr_id][0]);
                    cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));				
                }
            }

            pdata[19] += throughput;
        } while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > (uint64_t)throughput + pdata[19]));

        *hashes_done = pdata[19] - first_nonce + 1;

        return 0;
    }
    
    // cleanup
    extern "C" void free_xevan(int thr_id)
    {
        if (!init[thr_id])
            return;
    
        cudaDeviceSynchronize();
    
        cudaFree(d_hash[thr_id]);

    	quark_blake512_cpu_free(thr_id);
        xevan_simd512_cpu_free(thr_id);
        //x13_fugue512_cpu_free(thr_id);
        x15_whirlpool_cpu_free(thr_id);
        xevan_simd512_cpu_free(thr_id);

        cuda_check_cpu_free(thr_id);

        cudaDeviceSynchronize();
        init[thr_id] = false;
    }
    