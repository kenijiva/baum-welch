/* backward algoritm: return observation likelihood */
float backward(int *data, int len, int nstates, int nobvs,
        float *prior, float * trans, float *obvs)
{
    /* construct trellis */
    float *beta = (float *)aligned_alloc(32, len * nstates * sizeof(float));

    float loglik;

    double startTime = CycleTimer::currentSeconds();

    for (int i = 0; i < nstates; i++) {
        beta[(len-1) * nstates + i] = 0;         /* 0 = log (1.0) */
    }

    for (int i = 1; i < len; i++) {
        #pragma omp parallel for 
        for (int j = 0; j < nstates; j+=8) {
            __m256 beta_AVX = _mm256_set1_ps(-INFINITY);
            for (int k = 0; k < nstates; k++) {
                __m256 p_AVX = _mm256_set1_ps(beta[(len-i) * nstates + k]);
                __m256 trans_AVX = _mm256_load_ps(trans + k * nstates + j);
                __m256 obvs_AVX = _mm256_set1_ps(obvs[data[len-i] * nstates + k]);
                p_AVX = _mm256_add_ps(p_AVX, trans_AVX);
                p_AVX = _mm256_add_ps(p_AVX, obvs_AVX);
                beta_AVX = logadd(beta_AVX, p_AVX); 

            }
            // Store beta and xi
            _mm256_store_ps(beta + (len-1-i) * nstates + j, beta_AVX);

        }
    }

    loglik = -INFINITY;
    for (int i = 0; i < nstates; i++) {
        loglik = logadd(loglik, prior[i] + beta[i] + obvs[IDX(i,data[0],nobvs)]);
    }
    double endTime = CycleTimer::currentSeconds();
    printf("Time taken %.4f milliseconds\n",  (endTime - startTime) * 1000);
    free(beta);
    return loglik;
}
