#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "common.h"


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


void baum_welch_basic(double *a, double* alpha, double *b, double *beta, double *pi, int *o, int N, int M, int T, double *a_, double *b_, double *pi_){
    /* 
     *  Forward procedure
     */
     
    // Base case
    for(int i = 0; i < N; i++){
        alpha[i] = pi[i] * b[i*M  + o[0]];
    }

    // Step case
    for(int t = 0; t < T - 1; t++){
        for(int j = 0; j < N; j++){

            double sum = 0;
            for(int i = 0; i < N; i++){
                sum += alpha[t*N + i] * a[i*N  + j];
            }
            
            alpha[(t+1)*N + j] = sum * b[j*M + o[t+1]];
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
            for(int j = 0; j < N; j++){
                beta[t*N + i] += beta[(t+1)*N + j] * a[i*N  + j] * b[j*M + o[t+1]];
            }
        }
    }
    
    /*
     *    Update pi
     */
    for(int i = 0; i < N; i++){
        double sum = 0;
        for(int j = 0; j < N; j++){
            sum += alpha[(T-1)*N + j];
        }

        pi_[i] = (alpha[i] * beta[i]) / sum;
    }
    
    /*
     *  Update A
     */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double zeta = 0;
            double gamma = 0;
            double zeta_temp;
            double gamma_temp;
            for (int t = 0; t < T-1; t++) {
                zeta_temp = alpha[t*N + i] * a[i*N  + j] * b[j*M + o[t+1]] * beta[(t+1)*N +j];

                double sum_zeta = 0;
                for (int k = 0; k < N; k++) {
                    sum_zeta += alpha[t*N + k] * beta[t*N + k];
                }

                zeta += zeta_temp / sum_zeta;

                double sum_gamma = 0;
                gamma_temp = alpha[t*N + i] * beta[t*N + i];
                for (int k = 0; k < N; k++) {
                    sum_gamma += alpha[t*N + k] * beta[t*N + k];
                }
                gamma += gamma_temp / sum_gamma;

            }
            a_[i*N  + j] = zeta / gamma;
        }
    }
    
    /*
     *  Update B
     */

    for(int j = 0; j < N; j++){
        for(int k = 0; k < M; k++){

            double total = 0;
            double sum = 0;

            for(int t = 0; t < T; t++){                
                double rowsum = 0;
                for(int j = 0; j < N; j++){
                    rowsum += alpha[(T-1)*N + j];
                }
                double gamma =  (alpha[t*N + j] * beta[t*N + j]) / rowsum;

                if(k == o[t]) sum += gamma;

                total += gamma;
            }
            b_[j*M + k] = sum / total;
        }
    }
    
}


void baum_welch_a_bit_less_flops(double *a, double* alpha, double *b, double *beta, double *pi, int *o, int N, int M, int T, double *a_, double *b_, double *pi_){
    /* 
     *  Forward procedure
     */
     
    // Base case
    for(int i = 0; i < N; i++){
        alpha[i] = pi[i] * b[i*M  + o[0]];
    }

    // Step case
    for(int t = 0; t < T - 1; t++){
        for(int j = 0; j < N; j++){

            double sum = 0;
            for(int i = 0; i < N; i++){
                sum += alpha[t*N + i] * a[i*N  + j];
            }
            
            alpha[(t+1)*N + j] = sum * b[j*M + o[t+1]];
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
            double temp = beta[(t+1)*N + j] * b[j*M + o[t+1]];
            for(int i = 0; i < N; i++){
                beta[t*N + i] += temp * a[i*N  + j];
            }
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
    for(int i = 0; i < N; i++){
        pi_[i] = alpha[i] * beta[i] * p;
    }
    
    /*
     *  Update A
     */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double zeta = 0;
            double gamma = 0;
            
            for(int t = 0; t < T-1; t++){
                zeta += alpha[t*N + i] * b[j*M + o[t+1]] * beta[(t+1)*N +j];
                gamma += alpha[t*N + i] * beta[t*N + i];      
            }
            zeta *= a[i*N  + j];
            a_[i*N  + j] = zeta / gamma;
        }
    }
    
    /*
     *  Update B
     */

    for(int j = 0; j < N; j++){
        for(int k = 0; k < M; k++){

            double total = 0;
            double sum = 0;

            for(int t = 0; t < T; t++){
                double gamma =  alpha[t*N + j] * beta[t*N + j];

                if(k == o[t]) sum += gamma;

                total += gamma;
            }
            b_[j*M + k] = sum / total;
        }
    }
    
}




void baum_welch_less_flops(double *a, double* alpha, double *b, double *beta, double *pi, int *o, int N, int M, int T, double *a_, double *b_, double *pi_){
    /* 
     *  Forward procedure
     */
     
    // Base case
    for(int i = 0; i < N; i++){
        alpha[i] = pi[i] * b[i*M  + o[0]];
    }

    // Step case
    for(int t = 0; t < T - 1; t++){
        for(int j = 0; j < N; j++){

            double sum = 0;
            for(int i = 0; i < N; i++){
                sum += alpha[t*N + i] * a[i*N  + j];
            }
            
            alpha[(t+1)*N + j] = sum * b[j*M + o[t+1]];
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
            double temp = beta[(t+1)*N + j] * b[j*M + o[t+1]];
            for(int i = 0; i < N; i++){
                beta[t*N + i] += temp * a[i*N  + j];
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
            b_[i*M + z] = 0;
        
        // Precompute gamma
        b_[i*M + o[0]] += alpha_beta;
        double gamma = alpha_beta;
        for(int t = 1; t < T-1; t++){
            alpha_beta = alpha[t*N + i] * beta[t*N + i];
            b_[i*M + o[t]] += alpha_beta;
            gamma += alpha_beta;
        }

        /*
        *  Update A
        */
        for (int j = 0; j < N; j++) {
            double zeta = 0;
            for(int t = 0; t < T-1; t++){
                zeta += alpha[t*N + i] * b[j*M + o[t+1]] * beta[(t+1)*N +j];    
            }
            
            zeta *= a[i*N  + j];
            a_[i*N  + j] = zeta / gamma;
        }
    
        /*
         *  Update B
         */
        alpha_beta = alpha[(T-1)*N + i] * beta[(T-1)*N + i];
        gamma += alpha_beta;
        b_[i*M + o[T-1]] += alpha_beta;
        for(int k = 0; k < M; k++){
            b_[i*M + k] /= gamma;
        }
    }
    
}


void baum_welch_reduced_flops(double *a, double* alpha, double *b, double *beta, double *pi, int *o, int N, int M, int T, double *a_, double *b_, double *pi_){
    /* 
     *  Forward procedure
     */
     
    // Base case
    for(int i = 0; i < N; i++){
        alpha[i] = pi[i] * b[i*M  + o[0]];
    }

    // Step case
    for(int t = 0; t < T - 1; t++){
        for(int j = 0; j < N; j++){

            double sum = 0;
            for(int i = 0; i < N; i++){
                sum += alpha[t*N + i] * a[i*N  + j];
            }
            
            alpha[(t+1)*N + j] = sum * b[j*M + o[t+1]];
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
            double beta_b = beta[(t+1)*N + j] * b[j*M + o[t+1]];
            for(int i = 0; i < N; i++){
                beta[t*N + i] += beta_b * a[i*N  + j];
            }
        }
    }
    
    // P(O|\lambda)
    double total_prob = 0;
    for(int j = 0; j < N; j++){
        total_prob += alpha[(T-1)*N + j];
    }
    double p = 1 / total_prob;
    
    double b_beta[N*(T-1)];
    for(int t = 0; t < T-1; t++){
        for (int j = 0; j < N; j++) {
                b_beta[t*N + j] = b[j*M + o[t+1]] * beta[(t+1)*N +j];
            }
    }
    
    for (int i = 0; i < N; i++) {
        double alpha_beta = alpha[i] * beta[i];
        /*
         *    Update pi
         */
        pi_[i] = alpha_beta * p;
        
        for(int z = 0; z < M; z++)
            b_[i*M + z] = 0;
        
        // Precompute gamma
        b_[i*M + o[0]] += alpha_beta;
        double gamma = alpha_beta;
        for(int t = 1; t < T-1; t++){
            alpha_beta = alpha[t*N + i] * beta[t*N + i];
            b_[i*M + o[t]] += alpha_beta;
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
            
            zeta *= a[i*N  + j];
            a_[i*N  + j] = zeta * gamma_recipr;
        }
    
        /*
         *  Update B
         */
        alpha_beta = alpha[(T-1)*N + i] * beta[(T-1)*N + i];
        gamma += alpha_beta;
        gamma_recipr = 1/gamma;
        b_[i*M + o[T-1]] += alpha_beta;
        for(int k = 0; k < M; k++){
            b_[i*M + k] *= gamma_recipr;
        }
    }
    
}

void baum_welch_more_reduced_flops(double *a, double* alpha, double *b, double *beta, double *pi, int *o, int N, int M, int T, double *a_, double *b_, double *pi_){
    /* 
     *  Forward procedure
     */
     
    // Base case
    for(int i = 0; i < N; i++){
        alpha[i] = pi[i] * b[i*M  + o[0]];
    }

    // Step case
    for(int t = 0; t < T - 1; t++){
        for(int j = 0; j < N; j++){

            double sum = 0;
            for(int i = 0; i < N; i++){
                sum += alpha[t*N + i] * a[i*N  + j];
            }
            
            alpha[(t+1)*N + j] = sum * b[j*M + o[t+1]];
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

    double b_beta[N*(T-1)];
    for(int t = T - 2; t >= 0; t--){
        for(int j = 0; j < N; j++){
            b_beta[t*N + j] = b[j*M + o[t+1]] * beta[(t+1)*N +j];
            for(int i = 0; i < N; i++){
                beta[t*N + i] += b_beta[t*N+j] * a[i*N  + j];
            }
        }
    }
    
    // P(O|\lambda)


void register_functions() {
  add_function(&baum_welch_basic, "base");
  add_function(&baum_welch_a_bit_less_flops, "a_bit_less_flops");
  add_function(&baum_welch_less_flops, "less_flops");
  add_function(&baum_welch_reduced_flops, "reduced_flops");
  add_function(&baum_welch_more_reduced_flops, "more_reduced_flops");
}
