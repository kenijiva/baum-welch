#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include "common.h"
#include "transpose_SIMD.h"


/*
    Prints a given matrix
*/
void print_matrix(int N, int M, double *v){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            printf("%f ", v[i*M  + j]);
        }
        printf("\n");
    }

    printf("\n");
}

/*
 *  Transposes a MxN matrix 
 */
inline void transpose_matrix(double *v, int N, int M){
    double *transpose = (double*) malloc(N* M * sizeof(double));
    
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            transpose[i*M + j] = v[j*N + i];
        }
    }
    
    memcpy(v, transpose, N * M * sizeof(double));
    free(transpose);

}

void baum_welch_column_major(double *a, double* alpha, double *b, double *beta, double *pi, int *o, int N, int M, int T, double *a_, double *b_, double *pi_){
    
    // allocate resources
    double* b_beta = static_cast<double *>(aligned_alloc(32, (T-1) * N * sizeof(double)));
    
    /* 
     *  Forward procedure
     */
     
    // Base case
    for(int i = 0; i < N; i++){
        alpha[i] = pi[i] * b[o[0]*N + i];
    }

    // Step case
    for(int t = 0; t < T - 1; t++){
        for(int j = 0; j < N; j++){

            double sum = 0;
            for(int i = 0; i < N; i++){
                sum += alpha[t*N + i] * a[j*N + i];
            }
            
            alpha[(t+1)*N + j] = sum * b[o[t+1]*N + j];
        }
    }
    
    /*
        Backward prodcedure
    */
    
    // Base case
    for(int i = 0; i < N; i++){
        beta[(T-1)*N + i] = 1;
    }

    // Step case
    for(int t = T - 2; t >= 0; t--){
        for(int i = 0; i < N; i++){
            beta[t*N + i] = 0;
	    }
    }

    for(int t = T - 2; t >= 0; t--){
        for(int j = 0; j < N; j++){
            b_beta[t*N + j] = b[o[t+1]*N + j] * beta[(t+1)*N +j];
            for(int i = 0; i < N; i++){
                beta[t*N + i] += b_beta[t*N+j] * a[j*N  + i];
            }
        }
    }
    
    // P(O|\lambda)
    double total_prob = 0;
    for(int j = 0; j < N; j++){
        total_prob += alpha[(T-1)*N + j];
    }
    double p = 1 / total_prob;
    
    for (int i = 0; i < N; i++) {
        double alpha_beta = alpha[i] * beta[i];
        /*
         *    Update pi
         */
        pi_[i] = alpha_beta * p;
        
        for(int z = 0; z < M; z++)
            b_[z*N + i] = 0;
        
        // Precompute gamma
        b_[o[0]*N + i] += alpha_beta;
        double gamma = alpha_beta;
        for(int t = 1; t < T-1; t++){
            alpha_beta = alpha[t*N + i] * beta[t*N + i];
            b_[o[t]*N + i] += alpha_beta;
            gamma += alpha_beta;
        }

        /*
        *  Update A
        */
       double gamma_recipr = 1/gamma;
        for (int j = 0; j < N; j++) {
            double zeta = 0;
            for(int t = 0; t < T-1; t++){
                zeta += alpha[t*N + i] * b_beta[t*N + j];    
            }
            
            zeta *= a[j*N  + i];
            a_[j*N + i] = zeta * gamma_recipr;
        }
    
        /*
         *  Update B
         */
        alpha_beta = alpha[(T-1)*N + i] * beta[(T-1)*N + i];
        gamma += alpha_beta;
        gamma_recipr = 1/gamma;
        b_[o[T-1]*N + i] += alpha_beta;
        for(int k = 0; k < M; k++){
            b_[k*N + i] *= gamma_recipr;
        }
    }
}

/* Transposes alpha, beta and b_beta in the middle */
void baum_welch_column_major_transpose_in_middle(double *a, double* alpha, double *b, double *beta, double *pi, int *o, int N, int M, int T, double *a_, double *b_, double *pi_){
    
    // allocate resources
    double* b_beta = static_cast<double *>(aligned_alloc(32, (T-1) * N * sizeof(double)));
    double* alpha_beta = static_cast<double *>(aligned_alloc(32, N * sizeof(double)));

    /* 
     *  Forward procedure
     */
     
    // Base case
    for(int i = 0; i < N; i++){
        alpha[i] = pi[i] * b[o[0]*N + i];
    }

    // Step case
    for(int t = 0; t < T - 1; t++){
        for(int j = 0; j < N; j++){

            double sum = 0;
            for(int i = 0; i < N; i++){
                sum += alpha[t*N + i] * a[j*N + i];
            }
            
            alpha[(t+1)*N + j] = sum * b[o[t+1]*N + j];
        }
    }
    
    /*
        Backward prodcedure
    */
    
    // Base case
    for(int i = 0; i < N; i++){
        beta[(T-1)*N + i] = 1;
    }

    // Step case
    for(int t = T - 2; t >= 0; t--){
        for(int i = 0; i < N; i++){
            beta[t*N + i] = 0;
        }
    }

    for(int t = T - 2; t >= 0; t--){
        for(int j = 0; j < N; j++){
            b_beta[t*N + j] = b[o[t+1]*N + j] * beta[(t+1)*N +j];
            for(int i = 0; i < N; i++){
                beta[t*N + i] += b_beta[t*N+j] * a[j*N  + i];
            }
        }
    }

    // P(O|\lambda)
    double total_prob = 0;
    for(int j = 0; j < N; j++){
        total_prob += alpha[(T-1)*N + j];
    }
    double p = 1 / total_prob;

    for (int i = 0; i < N; i++) {
        alpha_beta[i] = alpha[i] * beta[i];
    }
    
    transpose_matrix(alpha, N, T);
    transpose_matrix(beta, N, T);
    transpose_matrix(b_beta, N, (T-1));

    for (int i = 0; i < N; i++) {
        /*
         *    Update pi
         */
        pi_[i] = alpha_beta[i] * p;
        
        for(int z = 0; z < M; z++)
            b_[z*N + i] = 0;
        
        // Precompute gamma
        b_[o[0]*N + i] += alpha_beta[i];
        double gamma = alpha_beta[i];
        for(int t = 1; t < T-1; t++){
            alpha_beta[i] = alpha[i*T + t] * beta[i*T + t];
            b_[o[t]*N + i] += alpha_beta[i];
            gamma += alpha_beta[i];
        }

        /*
        *  Update A
        */
        double gamma_recipr = 1/gamma;
        for (int j = 0; j < N; j++) {
            double zeta = 0;
            for(int t = 0; t < T-1; t++){
                zeta += alpha[i*T + t] * b_beta[j*(T-1) + t];  
            }

            a_[j*N + i] = zeta * gamma_recipr;
        }
    
        /*
         *  Update B
         */
        alpha_beta[i] = alpha[i*T + (T-1)] * beta[i*T + (T-1)];
        gamma += alpha_beta[i];
        gamma_recipr = 1/gamma;
        b_[o[T-1]*N + i] += alpha_beta[i];
        for(int k = 0; k < M; k++){
            b_[k*N + i] *= gamma_recipr;
        }
    }
    
    // Complet update A
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            a_[j*N + i] *= a[j*N  + i];
        }
    }
    
}

/* Uses SIMD transposition. WARNING: Crashes if N % 4 != 0 */
void baum_welch_simd_transpose(double *a, double* alpha, double *b, double *beta, double *pi, int *o, int N, int M, int T, double *a_, double *b_, double *pi_){
    
    // allocate resources
    double* b_beta = static_cast<double *>(aligned_alloc(32, T * N * sizeof(double)));
    double* alpha_beta = static_cast<double *>(aligned_alloc(32, N * sizeof(double)));
    
    /* 
     *  Forward procedure
     */
     
    // Base case
    for(int i = 0; i < N; i++){
        alpha[i] = pi[i] * b[o[0]*N + i];
    }

    // Step case
    for(int t = 0; t < T - 1; t++){
        for(int j = 0; j < N; j++){

            double sum = 0;
            for(int i = 0; i < N; i++){
                sum += alpha[t*N + i] * a[j*N + i];
            }
            
            alpha[(t+1)*N + j] = sum * b[o[t+1]*N + j];
        }
    }
    
    /*
        Backward prodcedure
    */
    
    // Base case
    for(int i = 0; i < N; i++){
        beta[(T-1)*N + i] = 1;
    }

    // Step case
    for(int t = T - 2; t >= 0; t--){
        for(int i = 0; i < N; i++){
            beta[t*N + i] = 0;
        }
    }

    for(int t = T - 2; t >= 0; t--){
        for(int j = 0; j < N; j++){
            b_beta[t*N + j] = b[o[t+1]*N + j] * beta[(t+1)*N +j];
            for(int i = 0; i < N; i++){
                beta[t*N + i] += b_beta[t*N+j] * a[j*N  + i];
            }
        }
    }

    // P(O|\lambda)
    double total_prob = 0;
    for(int j = 0; j < N; j++){
        total_prob += alpha[(T-1)*N + j];
    }
    double p = 1 / total_prob;

    for (int i = 0; i < N; i++) {
        alpha_beta[i] = alpha[i] * beta[i];
    }
    
    transpose_simd(alpha, N, T);
    transpose_simd(beta, N, T);
    transpose_simd(b_beta, N, T);

    for (int i = 0; i < N; i++) {
        /*
         *    Update pi
         */
        double alpha_beta_temp = alpha_beta[i];
        pi_[i] = alpha_beta_temp * p;
        
        for(int z = 0; z < M; z++)
            b_[z*N + i] = 0;
        
        // Precompute gamma
        b_[o[0]*N + i] += alpha_beta_temp;
        double gamma = alpha_beta_temp;
        for(int t = 1; t < T-1; t++){
            alpha_beta_temp = alpha[i*T + t] * beta[i*T + t];
            b_[o[t]*N + i] += alpha_beta_temp;
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
        alpha_beta_temp = alpha[i*T + (T-1)] * beta[i*T + (T-1)];
        b_[o[T-1]*N + i] += alpha_beta_temp;
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
    
    // Complet update A
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            a_[j*N + i] *= a[j*N  + i];
        }
    }
    
}

void baum_welch_all_row_access_column_major(double *a, double* alpha, double *b, double *beta, double *pi, int *o, int N, int M, int T, double *a_, double *b_, double *pi_){
    
    // allocate resources
    double* b_beta = static_cast<double *>(aligned_alloc(32, (T-1) * N * sizeof(double)));
    double* gamma = static_cast<double *>(aligned_alloc(32, N * sizeof(double)));
    double* gamma_ = static_cast<double *>(aligned_alloc(32, N * sizeof(double)));

    /* 
     *  Forward procedure
     */
     
    // Base case
    for(int i = 0; i < N; i++){
        alpha[i] = pi[i] * b[o[0]*N + i];
    }

    // Step case
    for(int t = 0; t < T - 1; t++){
        for(int j = 0; j < N; j++){

            double sum = 0;
            for(int i = 0; i < N; i++){
                sum += alpha[t*N + i] * a[j*N + i];
            }
            
            alpha[(t+1)*N + j] = sum * b[o[t+1]*N + j];
        }
    }
    
    /*
        Backward prodcedure
    */
    
    // Base case
    for(int i = 0; i < N; i++){
        beta[(T-1)*N + i] = 1;
    }

    // Step case
    for(int t = T - 2; t >= 0; t--){
        for(int i = 0; i < N; i++){
            beta[t*N + i] = 0;
	    }
    }

    for(int t = T - 2; t >= 0; t--){
        for(int j = 0; j < N; j++){
            b_beta[t*N + j] = b[o[t+1]*N + j] * beta[(t+1)*N +j];
            for(int i = 0; i < N; i++){
                beta[t*N + i] += b_beta[t*N+j] * a[j*N  + i];
            }
        }
    }
    
    for(int t = 0; t < T; t++){
        for(int i = 0; i < N; i++){
            beta[t*N + i] *= alpha[t*N + i];
        }
    }
    
    // P(O|\lambda)
    double total_prob = 0;
    for(int j = 0; j < N; j++){
        total_prob += alpha[(T-1)*N + j];
    }
    double p = 1 / total_prob;
    
    /*
     *    Update pi
     */
    for (int i = 0; i < N; i++) {
        pi_[i] = beta[i] * p;
    }
    
    for(int j = 0; j < M; j++){
        for(int i = 0; i < N; i++){  
            b_[j*N + i] = 0;
        }
    }
    
    // Precompute gamma
    for(int i = 0; i < N; i++){
        gamma[i] = 0;
    }

    for(int t = 0; t < T-1; t++){
        for(int i = 0; i < N; i++){        
            b_[o[t]*N + i] += beta[t*N + i];
            gamma[i] += beta[t*N + i];
        }
    }
    
    /*
     *  Update A
     */
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            a_[i*N + j] = 0;
        }
    }
    
    for(int t = 0; t < T-1; t++){
        for(int j = 0; j < N; j++){
            for(int i = 0; i < N; i++){
                a_[j*N + i] += alpha[t*N + i] * b_beta[t*N + j]; 
            }
        }
    }
    
    for(int i = 0; i < N; i++){
        gamma_[i] = 1/gamma[i];
    }
    
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            a_[j*N + i] *= a[j*N + i];
            a_[j*N + i] *= gamma_[i];
        }
    }
    
    /*
     *  Update B
     */
    for(int i = 0; i < N; i++){ 
        gamma[i] = 1/(gamma[i] + beta[(T-1)*N + i]);
        b_[o[T-1]*N + i] += beta[(T-1)*N + i];
    }
    
    for(int k = 0; k < M; k++){
        for(int i = 0; i < N; i++){
            b_[k*N + i] *= gamma[i];
        }
    }
}

void register_functions() {
  add_function(&baum_welch_column_major, "column_major");
  add_function(&baum_welch_column_major_transpose_in_middle, "column_major + transpose alpha & beta & b_beta in the middle");
  add_function(&baum_welch_simd_transpose, "transpose with SIMD");
  add_function(&baum_welch_all_row_access_column_major, "all_row_access_column_major");
}
