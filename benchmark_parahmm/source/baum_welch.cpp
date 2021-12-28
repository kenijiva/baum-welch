float *gmm = NULL;             /* gamma */
float *xi = NULL;              /* xi */
float *pi = NULL;              /* pi */

void printAVX(__m256 vect, char* title){
    float *point = (float *)&vect;
    printf("%s:%f\n", title, point[0]);
}
/* forward backward algoritm: return observation likelihood */
float forward_backward(int *data, int len, int nstates, int nobvs, float *prior, float *trans, float *transT, float *obvs)
{
    /* construct trellis */
    float *alpha = (float *)aligned_alloc(32, len * nstates * sizeof(float));
    float *beta = (float *)aligned_alloc(32, len * nstates * sizeof(float));

    float loglik;

    /* forward pass */
    for (int i = 0; i < nstates; i++) {
        alpha[i] = prior[i] + obvs[IDXT(i,data[0],nstates)];
    }
    
    __m256 all_Inf = _mm256_set1_ps(-INFINITY);

    double startTime = CycleTimer::currentSeconds();
    for (int i = 1; i < len; i++) {
#pragma omp parallel for
       // for (int j = 0; j < nstates; j++) {
       //     for (int k = 0; k < nstates; k++) {
       //         float p = alpha[(i-1) * nstates + k] + trans[IDX(k,j,nstates)] + obvs[IDXT(j,data[i],nstates)];
       //         alpha[i * nstates + j] = logadd(alpha[i * nstates + j], p);
       //     }
       // }
       
        for (int j = 0; j < nstates; j+=16) {
            __m256 result_AVX = _mm256_set1_ps(-INFINITY);
            __m256 result_AVX2 = _mm256_set1_ps(-INFINITY);
            __m256 obvs_AVX = _mm256_load_ps(obvs + data[i] * nstates + j);
            __m256 obvs_AVX2 = _mm256_load_ps(obvs + data[i] * nstates + j + 8);
            for (int k = 0; k < nstates; k++) {
                __m256 alpha_AVX = _mm256_set1_ps(alpha[(i-1) * nstates + k]);
                __m256 alpha_AVX2 = _mm256_set1_ps(alpha[(i-1) * nstates + k]);
                __m256 trans_AVX = _mm256_load_ps(trans + k*nstates + j);
                __m256 trans_AVX2 = _mm256_load_ps(trans + k*nstates + j + 8);
                // calculate p
                alpha_AVX = _mm256_add_ps(alpha_AVX, trans_AVX);
                alpha_AVX2 = _mm256_add_ps(alpha_AVX2, trans_AVX2);
                alpha_AVX = _mm256_add_ps(alpha_AVX, obvs_AVX);
                alpha_AVX2 = _mm256_add_ps(alpha_AVX2, obvs_AVX2);
                result_AVX = logadd(result_AVX, alpha_AVX);
                result_AVX2 = logadd(result_AVX2, alpha_AVX2);
            }
            _mm256_store_ps(alpha + i*nstates + j, result_AVX);
            _mm256_store_ps(alpha + i*nstates + j + 8, result_AVX2);
        }
    }
    // printf("Forward taken %.4f milliseconds\n",  (CycleTimer::currentSeconds() - startTime) * 1000);
    loglik = -INFINITY;
    int thread_num = omp_get_max_threads();
#pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        float local_loglik = -INFINITY;
#pragma omp for
        for (int i = 0; i < nstates; i++) {
            local_loglik = logadd(local_loglik, alpha[(len-1) * nstates + i]);
        }
#pragma omp critical
        {
            loglik = logadd(local_loglik, loglik);
        }
    }

    /* backward pass & update counts */
    for (int i = 0; i < nstates; i++) {
        beta[(len-1) * nstates + i] = 0;         /* 0 = log (1.0) */
    }

    __m256 loglik_AVX = _mm256_set1_ps(loglik);
    startTime = CycleTimer::currentSeconds();
    for (int i = 1; i < len; i++) {
        #pragma omp parallel for
       // for (int j = 0; j < nstates; j++) {

       //     float e = alpha[(len-i) * nstates + j] + beta[(len-i) * nstates + j] - loglik;
       //     gmm[IDXT(j,data[len-i],nstates)] = logadd(gmm[IDXT(j,data[len-i],nstates)], e);

       //     for (int k = 0; k < nstates; k++) {
       //         float p = beta[(len-i) * nstates + k] + transT[IDXT(j,k,nstates)] + obvs[IDXT(k,data[len-i],nstates)];
       //         beta[(len-1-i) * nstates + j] = logadd(beta[(len-1-i) * nstates + j], p);

       //         e = alpha[(len-1-i) * nstates + j] + beta[(len-i) * nstates + k]
       //             + transT[IDXT(j,k,nstates)] + obvs[IDXT(k,data[len-i],nstates)] - loglik;
       //         xi[IDXT(j,k,nstates)] = logadd(xi[IDXT(j,k,nstates)], e);
       //     }
       // }
        for (int j = 0; j < nstates; j+=16) {
            __m256 gmm_AVX = _mm256_load_ps(gmm + data[len-i]*nstates + j);
            __m256 gmm_AVX2 = _mm256_load_ps(gmm + data[len-i]*nstates + j + 8);
            __m256 alpha_AVX = _mm256_load_ps(alpha + (len-i) * nstates + j);
            __m256 alpha_AVX2 = _mm256_load_ps(alpha + (len-i) * nstates + j + 8);
            __m256 beta_AVX = _mm256_load_ps(beta + (len-i) * nstates + j);
            __m256 beta_AVX2 = _mm256_load_ps(beta + (len-i) * nstates + j + 8);
            alpha_AVX = _mm256_add_ps(alpha_AVX, beta_AVX);
            alpha_AVX2 = _mm256_add_ps(alpha_AVX2, beta_AVX2);
            alpha_AVX = _mm256_sub_ps(alpha_AVX, loglik_AVX);
            alpha_AVX2 = _mm256_sub_ps(alpha_AVX2, loglik_AVX);
            gmm_AVX = logadd(gmm_AVX, alpha_AVX);
            gmm_AVX2 = logadd(gmm_AVX2, alpha_AVX2);
            _mm256_store_ps(gmm + data[len-i] * nstates + j, gmm_AVX);
            _mm256_store_ps(gmm + data[len-i] * nstates + j + 8, gmm_AVX2);

            beta_AVX = _mm256_set1_ps(-INFINITY);
            beta_AVX2 = _mm256_set1_ps(-INFINITY);
            alpha_AVX = _mm256_load_ps(alpha + (len-1-i) * nstates + j);
            alpha_AVX2 = _mm256_load_ps(alpha + (len-1-i) * nstates + j + 8);
            for (int k = 0; k < nstates; k++) {
                __m256 p_AVX = _mm256_set1_ps(beta[(len-i) * nstates + k]);
                __m256 trans_AVX = _mm256_load_ps(transT + k * nstates + j);
                __m256 trans_AVX2 = _mm256_load_ps(transT + k * nstates + j + 8);
                __m256 obvs_AVX = _mm256_set1_ps(obvs[data[len-i] * nstates + k]);
                __m256 p_AVX1 = _mm256_add_ps(p_AVX, trans_AVX);
                __m256 p_AVX2 = _mm256_add_ps(p_AVX, trans_AVX2);
                p_AVX1 = _mm256_add_ps(p_AVX1, obvs_AVX);
                p_AVX2 = _mm256_add_ps(p_AVX2, obvs_AVX);
                beta_AVX = logadd(beta_AVX, p_AVX1); 
                beta_AVX2 = logadd(beta_AVX2, p_AVX2); 

                __m256 xi_AVX = _mm256_load_ps(xi + k*nstates + j);
                __m256 xi_AVX2 = _mm256_load_ps(xi + k*nstates + j + 8);
                __m256 e_AVX = _mm256_set1_ps(beta[(len-i) * nstates + k]);
                __m256 e_AVX2 = _mm256_set1_ps(beta[(len-i) * nstates + k]);
                // Use old transittion
                // Use old obvs
                e_AVX = _mm256_add_ps(e_AVX, alpha_AVX);
                e_AVX2 = _mm256_add_ps(e_AVX2, alpha_AVX2);
                e_AVX = _mm256_add_ps(e_AVX, trans_AVX);
                e_AVX2 = _mm256_add_ps(e_AVX2, trans_AVX2);
                e_AVX = _mm256_add_ps(e_AVX, obvs_AVX);
                e_AVX2 = _mm256_add_ps(e_AVX2, obvs_AVX);
                e_AVX = _mm256_sub_ps(e_AVX, loglik_AVX);
                e_AVX2 = _mm256_sub_ps(e_AVX2, loglik_AVX);
                xi_AVX = logadd(xi_AVX, e_AVX);
                xi_AVX2 = logadd(xi_AVX2, e_AVX2);
                _mm256_store_ps(xi + k*nstates + j, xi_AVX);
                _mm256_store_ps(xi + k*nstates + j + 8, xi_AVX2);
            }
            // Store beta and xi
            _mm256_store_ps(beta + (len-1-i) * nstates + j, beta_AVX);
            _mm256_store_ps(beta + (len-1-i) * nstates + j + 8, beta_AVX2);

        }
        //Store back to gmm
    }
    // ("Backward taken %.4f milliseconds\n",  (CycleTimer::currentSeconds() - startTime) * 1000);
    float p = -INFINITY;

#pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        float local_p = -INFINITY;
#pragma omp for
        for (int i = 0; i < nstates; i++) {
            local_p = logadd(local_p, prior[i] + beta[i] + obvs[IDXT(i,data[0],nstates)]);
        }
#pragma omp critical
        {
            p = logadd(local_p, p);
        }
    }

#pragma omp parallel for
    for (int i = 0; i < nstates; i++) {
        float e = alpha[i] + beta[i] - loglik;
        gmm[IDXT(i,data[0],nstates)] = logadd(gmm[IDXT(i,data[0],nstates)], e);

        pi[i] = logadd(pi[i], e);
    }

#ifdef DEBUG
    /* verify if forward prob == backward prob */
    if (fabs(p - loglik) > 1e-5) {
        fprintf(stderr, "Error: forward and backward incompatible: %lf, %lf\n", loglik, p);
    }
#endif

    return loglik;
}

double baum_welch(int *data, int nseq, int iterations, int length, int nstates, int nobvs, float *prior, float *trans, float *transT, float *obvs)
{

    double timing;

    float *loglik = (float *) malloc(sizeof(float) * nseq);
    if (loglik == NULL) handle_error("malloc");
    for (int i = 0; i < iterations; i++) {
        double startTime = CycleTimer::currentSeconds();
        init_count();
        for (int j = 0; j < nseq; j++) {
            loglik[j] = forward_backward(data + length * j, length, nstates, nobvs, prior, trans, transT, obvs);
        }
        float p = sum(loglik, nseq);

        double update_Time = CycleTimer::currentSeconds();
        update_prob(nstates, nobvs, prior, trans, transT, obvs);
        // ("update taken %.4f milliseconds\n",  (CycleTimer::currentSeconds() - update_Time) * 1000);

        // printf("iteration %d log-likelihood: %.4lf\n", i + 1, p);
        // printf("updated parameters:\n");
        //printf("# initial state probability\n");
        //for (int j = 0; j < nstates; j++) {
        //    printf(" %.4f", exp(prior[j]));
        //}
        //printf("\n");
        //printf("# state transition probability\n");
        //for (int j = 0; j < nstates; j++) {
        //    for (int k = 0; k < nstates; k++) {
        //        printf(" %.4f", exp(trans[IDX(j,k,nstates)]));
        //    }
        //    printf("\n");
        //}
        //printf("# state output probility\n");
        //for (int j = 0; j < nstates; j++) {
        //    for (int k = 0; k < nobvs; k++) {
        //        printf(" %.4f", exp(obvs[IDX(j,k,nobvs)]));
        //    }
        //    printf("\n");
        //}
        //printf("\n");
        double endTime = CycleTimer::currentSeconds();
        timing = (endTime - startTime) * 1000;
        //printf("======Time taken %.4f milliseconds======\n",  (endTime - startTime) * 1000);
    }
    free(loglik);

    return timing;
}

void update_prob(int nstates, int nobvs, float *prior, float *trans, float *transT, float *obvs) {
    float pisum = - INFINITY;
    int thread_num = omp_get_max_threads();
    float gmmsum[nstates];
    float xisum[nstates];
    //size_t i, j;

    for (int i = 0; i < nstates; i++) {
        gmmsum[i] = - INFINITY;
        xisum[i] = - INFINITY;

        pisum  = logadd(pisum, pi[i]);
    }

//#pragma omp parallel 
//    {
//        int tid = omp_get_thread_num();
//        float local_pisum = -INFINITY;
//        for (int i = tid; i < nstates; i+=thread_num) {
//            local_pisum = logadd(local_pisum, pi[i]);
//        }
//#pragma omp critical
//        {
//            pisum = logadd(local_pisum, pisum);
//        }
//    }

    for (int i = 0; i < nstates; i++) {
        prior[i] = pi[i] - pisum;
    }

    #pragma omp parallel for
    for (int i = 0; i < nstates; i++) {
        for (int j = 0; j < nstates; j++) {
            xisum[i] = logadd(xisum[i], xi[IDXT(i,j,nstates)]);
        }
    //}
    //for (int i = 0; i < nstates; i++) {
        for (int j = 0; j < nobvs; j++) {
            gmmsum[i] = logadd(gmmsum[i], gmm[IDXT(i,j,nstates)]);
        }
    }

    /* May need to blocking!!!*/
    for (int i = 0; i < nstates; i++) {
        for (int j = 0; j < nstates; j++) {
            trans[IDX(i,j,nstates)] = xi[IDXT(i,j,nstates)] - xisum[i];
            transT[IDXT(i,j,nstates)] = xi[IDXT(i,j,nstates)] - xisum[i];
        }
        for (int j = 0; j < nobvs; j++) {
            obvs[IDXT(i,j,nstates)] = gmm[IDXT(i,j,nstates)] - gmmsum[i];
        }
    }
}
