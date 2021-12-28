/* forward algoritm: return observation likelihood */
float forward(int *data, int len, int nstates, int nobvs,
        float *prior, float * trans, float *obvs)
{
    /* construct trellis */
    // float alpha[len][nstates];
    // float beta[len][nstates];
    float *alpha = (float *)aligned_alloc(32, len * nstates * sizeof(float));

    float loglik;

    double startTime = CycleTimer::currentSeconds();
    #pragma omp parallel
    {
        /* forward pass */
        #pragma omp for
        for (int i = 0; i < nstates; i++) {
            alpha[i] = prior[i] + obvs[IDXT(i,data[0],nstates)];
        }

        __m256 result_AVX;
        __m256 result_AVX2;
        __m256 alpha_AVX; 
        __m256 alpha_AVX2; 
        __m256 trans_AVX, obvs_AVX;
        __m256 trans_AVX2, obvs_AVX2;

        for (int i = 1; i < len; i++) {
            #pragma omp for
            //for (int j = 0; j < nstates; j++) {
            //    for (int k = 0; k < nstates; k++) {
            //        float p = alpha[(i-1) * nstates + k] + trans[IDXT(k,j,nstates)] + obvs[IDX(j,data[i],nobvs)];
            //        alpha[i * nstates + j] = logadd(alpha[i * nstates + j], p);
            //    }
            //}
            for (int j = 0; j < nstates; j+=16) {
                result_AVX = _mm256_set1_ps(-INFINITY);
                result_AVX2 = _mm256_set1_ps(-INFINITY);
                obvs_AVX = _mm256_load_ps(obvs + data[i] * nstates + j);
                obvs_AVX2 = _mm256_load_ps(obvs + data[i] * nstates + j + 8);
                for (int k = 0; k < nstates; k++) {
                    alpha_AVX = _mm256_set1_ps(alpha[(i-1) * nstates + k]);
                    alpha_AVX2 = _mm256_set1_ps(alpha[(i-1) * nstates + k]);
                    trans_AVX = _mm256_load_ps(trans + k*nstates + j);
                    trans_AVX2 = _mm256_load_ps(trans + k*nstates + j + 8);
                    // calculate p
                    alpha_AVX = _mm256_add_ps(alpha_AVX, trans_AVX);
                    alpha_AVX2 = _mm256_add_ps(alpha_AVX2, trans_AVX2);
                    alpha_AVX = _mm256_add_ps(alpha_AVX, obvs_AVX);
                    alpha_AVX2 = _mm256_add_ps(alpha_AVX2, obvs_AVX2);
                    result_AVX = logadd(result_AVX,  alpha_AVX);
                    result_AVX2 = logadd(result_AVX2,  alpha_AVX2);
                }
                _mm256_store_ps(alpha + i*nstates + j, result_AVX);
                _mm256_store_ps(alpha + i*nstates + j + 8, result_AVX2);
            }
        }
    }

    loglik = -INFINITY;
    for (int i = 0; i < nstates; i++) {
        loglik = logadd(loglik, alpha[(len-1) * nstates + i]);
    }
    double endTime = CycleTimer::currentSeconds();
    printf("Time taken %.4f milliseconds\n",  (endTime - startTime) * 1000);

    free(alpha);
    return loglik;
}

inline __m256 logadd(__m256 a, __m256 b) {
    __m256 max_AVX, min_AVX; 
    __m256 all_one = _mm256_set1_ps(1);
    max_AVX = _mm256_max_ps(a, b);
    min_AVX = _mm256_min_ps(a, b);
    min_AVX = _mm256_sub_ps(min_AVX, max_AVX);
    min_AVX = exp256_ps(min_AVX);
    min_AVX = _mm256_add_ps(min_AVX, all_one);
    min_AVX = log256_ps(min_AVX);
    return _mm256_add_ps(max_AVX,  min_AVX);
}
