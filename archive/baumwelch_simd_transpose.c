/* Uses SIMD transposition. WARNING: Crashes if N % 4 != 0 */
void baum_welch_simd_transpose_no_aliasing(double *a, double* alpha, double *b, double *beta, double *pi, int *o, int N, int M, int T, double *a_, double *b_, double *pi_){
    
    // allocate resources
    double* c = static_cast<double *>(calloc(T, sizeof(double)));
    double* b_beta = static_cast<double *>(malloc(T * N * sizeof(double)));
    double* alpha_beta = static_cast<double *>(malloc(N * sizeof(double)));
    
    /* 
     *  Forward procedure
     */
     
    // Base case
    int index = o[0];
    double tmp = 0;
    for(int i = 0; i < N; i++){
        alpha[i] = pi[i] * b[index*N + i];
        tmp += alpha[i];
    }
    c[0] = 1 / tmp;

    // Step case
    for(int t = 0; t < T - 1; t++){
        index = o[t+1];
        double ctmp = c[t+1];
        for(int j = 0; j < N; j++){

            double sum = 0;
            for(int i = 0; i < N; i++){
                sum += alpha[t*N + i] * a[j*N + i];
            }
            
            double alpha_tmp = sum * b[index*N + j];
            alpha[(t+1)*N + j] = alpha_tmp;
            ctmp += alpha_tmp;
        }
        c[t+1] = 1 / ctmp;

        for(int j = 0; j < N; j++){
            alpha[(t+1)*N + j] *= c[t+1];
        }
    }
    
    /*
        Backward prodcedure
    */
    
    // Base case
    for(int i = 0; i < N; i++){
        beta[(T-1)*N + i] = c[T-1];
    }

    // Step case
    for(int t = T - 2; t >= 0; t--){
        for(int i = 0; i < N; i++){
            beta[t*N + i] = 0;
        }
    }

    for(int t = T - 2; t >= 0; t--){
        index = o[t+1];
        for(int j = 0; j < N; j++){
            double tmp = b[index*N + j] * beta[(t+1)*N +j];
            double tmp2 = tmp * c[t];
            b_beta[t*N + j] = tmp;
            for(int i = 0; i < N; i++){
                beta[t*N + i] += tmp2 * a[j*N  + i];
            }
        }
    }

    for (int i = 0; i < N; i++) {
        alpha_beta[i] = alpha[i] * beta[i];
    }
    
    transpose_simd(alpha, N, T);
    transpose_simd(beta, N, T);
    transpose_simd(b_beta, N, T);

    for (int t = 0; t < T; t++) {
        c[t] = 1 / c[t];
    }

    double c0 = c[0];
    double cT = c[T-1];
    int index0 = o[0];
    int indexT = o[T-1];
    for (int i = 0; i < N; i++) {
        /*
         *    Update pi
         */
        double alpha_beta_temp = alpha_beta[i];
        double gamma = alpha_beta_temp * c0;
        pi_[i] = gamma;
        
        for(int z = 0; z < M; z++)
            b_[z*N + i] = 0;
        
        // Precompute gamma
        b_[index0*N + i] += gamma;
        for(int t = 1; t < T-1; t++){
            index = o[t];
            alpha_beta_temp = alpha[i*T + t] * beta[i*T + t] * c[t];
            b_[index*N + i] += alpha_beta_temp;
            gamma += alpha_beta_temp;
        }

        /*
        *  Update A
        */
        double gamma_recipr = 1/gamma;
        for (int j = 0; j < N; j++) {
            double zeta = 0;
            for(int t = 0; t < T-1; t++){
                zeta += alpha[i*T + t] * b_beta[j*T + t];  
            }

            a_[j*N + i] = zeta * gamma_recipr;
        }

        /*
        *  Update B
        */
        alpha_beta_temp = alpha[i*T + (T-1)] * beta[i*T + (T-1)] * cT;
        b_[indexT*N + i] += alpha_beta_temp;
        alpha_beta[i] = 1/(gamma + alpha_beta_temp);
    }

    /*
    *  Update B
    */
    for (int k = 0; k < M; k++) {
        for (int i = 0; i < N; i++) {
            b_[k*N + i] *= alpha_beta[i];
        }
    }
    
    // Complete update A
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            a_[j*N + i] *= a[j*N  + i];
        }
    }
    
    free(c);
    c = NULL;

    free(b_beta);
    b_beta = NULL;

    free(alpha_beta);
    alpha_beta = NULL;
}
