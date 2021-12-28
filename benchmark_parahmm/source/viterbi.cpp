/* find the most probable sequence */
void viterbi(int *data, int len, int nstates,int nobvs, float *prior, float *trans, float *obvs)
{
    float *lambda = (float *)aligned_alloc(32, 3 * nstates * sizeof(float));
    int *backtrace = (int *)aligned_alloc(32, len * nstates * sizeof(int));
    int *stack = (int *)malloc(len * sizeof(int));

    double startTime = CycleTimer::currentSeconds();
    for (int i = 0; i < nstates; i++) {
        lambda[i] = prior[i] + obvs[IDXT(i,data[0],nstates)];
        backtrace[i] = -1;       /* -1 is starting point */
    }
    
    /*for (int i = 1; i < len; i++) {
        for (int j = 0; j < nstates; j++) {
            obvs_AVX = _mm256_set1_ps(obvs[IDX(j,data[i],nobvs)]);
            for (int k = 0; k < nstates; k+=8) {
                lambda_AVX = _mm256_load_ps(lambda + (i-1) * nstates + k);
                trans_AVX = _mm256_load_ps(trans + j * nstates + k);
                result = _mm256_add_ps(lambda_AVX, trans_AVX);
                result = _mm256_add_ps(result, obvs_AVX);
                float* p = (float*)&result;
                for (int m = 0; m < 8; m++) {
                    if (p[m] > lambda[i * nstates + j]) {
                        lambda[i * nstates + j] = p[m];
                        backtrace[i * nstates + j] = k+m;
                    }
                }
            }
        }
    }*/

    for (int i = 1; i < len; i++) {
        /*Use SIMD to compute lambda[i][j..j+16] simultaneously.*/
        /*Loop Unrolling*/
        #pragma omp parallel for
        for (int j = 0; j < nstates; j+=16) {
            __m256 max = _mm256_set1_ps(-INFINITY);
            __m256 max2 = _mm256_set1_ps(-INFINITY);
            __m256 kMax = _mm256_setzero_ps();
            __m256 kMax2 = _mm256_setzero_ps();
            __m256 obvs_AVX = _mm256_load_ps(obvs + data[i] * nstates + j);
            __m256 obvs_AVX2 = _mm256_load_ps(obvs + data[i] * nstates + j + 8);
            /*lambda[i][j] = max{k}(lambda[i-1][k] * trans[k][j] * obvs[j][o]*/
            for (int k = 0; k < nstates; k++) {
                __m256 kIndex = _mm256_set1_ps((float)k);
                __m256 trans_AVX = _mm256_load_ps(trans + k * nstates + j);
                __m256 trans_AVX2 = _mm256_load_ps(trans + k * nstates + j + 8);
                __m256 lambda_AVX = _mm256_set1_ps(lambda[((i+1)%2) * nstates + k]);
                __m256 result = _mm256_add_ps(lambda_AVX, trans_AVX);
                __m256 result2 = _mm256_add_ps(lambda_AVX, trans_AVX2);
                result = _mm256_add_ps(result, obvs_AVX);
                result2 = _mm256_add_ps(result2, obvs_AVX2);
                __m256 mask = _mm256_cmp_ps(max, result, _CMP_LT_OQ);
                __m256 mask2 = _mm256_cmp_ps(max2, result2, _CMP_LT_OQ);
                kMax = _mm256_blendv_ps(kMax, kIndex, mask);
                kMax2 = _mm256_blendv_ps(kMax2, kIndex, mask2);
                max = _mm256_max_ps(max, result);
                max2 = _mm256_max_ps(max2, result2);
            }
            _mm256_store_ps(lambda + (i % 2) * nstates + j, max);
            _mm256_store_ps(lambda + (i % 2) * nstates + j + 8, max2);
            __m256i kMaxi = _mm256_cvtps_epi32(kMax);
            __m256i kMaxi2 = _mm256_cvtps_epi32(kMax2);
            _mm256_store_si256((__m256i *)(backtrace + i * nstates + j), kMaxi);
            _mm256_store_si256((__m256i *)(backtrace + i * nstates + j + 8), kMaxi2);
        }
    }

    int k = 0;
    float p = 0;
    /* backtrace */
    for (int i = 0; i < nstates; i++) {
        if (i == 0 || lambda[((len-1) % 2) * nstates + i] > p) {
            p = lambda[((len-1) % 2) * nstates + i];
            k = i;
        }
    }
    stack[len - 1] = k;
    for (int i = 1; i < len; i++) {
        stack[len - 1 - i] = backtrace[(len - i) * nstates + stack[len - i]];
    }
    for (int i = 0; i < len; i++) {
        printf("%d ", stack[i]);
    }
    printf("\n");
    double endTime = CycleTimer::currentSeconds();
    printf("Time taken %0.4f milliseconds\n",  (endTime - startTime) * 1000);

    free(lambda);
    free(backtrace);
    free(stack);
}
