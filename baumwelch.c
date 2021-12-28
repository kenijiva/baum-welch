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

void print_vector(__m256d vector){
    double data[4];
    _mm256_storeu_pd(data, vector);
    
    printf("VEC: %f %f %f %f \n", data[0], data[1], data[2], data[3]);
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

void baum_welch_basic(double *a, double* alpha, double *b, double *beta, double *pi, int *o, int N, int M, int T, double *a_, double *b_, double *pi_){
    
    double* c = static_cast<double *>(calloc(T, sizeof(double)));

    /* 
     *  Forward procedure
     */

    // Base case
    for(int i = 0; i < N; i++){
        alpha[i] = pi[i] * b[i*M  + o[0]];
        c[0] += alpha[i];
    }
    c[0] = 1 / c[0];
    
    for(int i = 0; i < N; i++){
        alpha[i] = alpha[i] * c[0];
    }

    // Step case
    for(int t = 0; t < T - 1; t++){
        for(int j = 0; j < N; j++){

            double sum = 0;
            for(int i = 0; i < N; i++){
                sum += alpha[t*N + i] * a[i*N  + j];
            }
            
            alpha[(t+1)*N + j] = sum * b[j*M + o[t+1]];
            c[t+1] += alpha[(t+1)*N + j];
        }
        c[t+1] = 1 / c[t+1];
        
        for(int j = 0; j < N; j++){
            alpha[(t+1)*N + j] *= c[t+1];
        }
    }
    
    /*
        Backward procedure
    */
    
    // Base case
    for(int i = 0; i < N; i++){
        beta[(T-1)*N + i] = c[T-1];
    }

    // Step case
    for(int t = T - 2; t >= 0; t--){
        for(int i = 0; i < N; i++){
            beta[t*N + i] = 0;
            for(int j = 0; j < N; j++){
                beta[t*N + i] += beta[(t+1)*N + j] * a[i*N  + j] * b[j*M + o[t+1]];
            }
            beta[t*N + i] *= c[t];
        }
    }
    
    /*
     *    Update pi
     */
    for(int i = 0; i < N; i++){
        pi_[i] = (alpha[i] * beta[i]) / c[0];
    }
    
    /*
     *  Update A
     */
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            double zeta = 0;
            double gamma = 0;

            for(int t = 0; t < T-1; t++){                
                zeta += alpha[t*N + i] * a[i*N  + j] * b[j*M + o[t+1]] * beta[(t+1)*N +j];
                gamma += alpha[t*N + i] * beta[t*N + i] / c[t];            
            }
            a_[i*N + j] = zeta / gamma;
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
                double gamma =  alpha[t*N + j] * (beta[t*N + j] /  c[t]) ;

                if(k == o[t]) sum += gamma;

                total += gamma;
            }
            b_[j*M + k] = sum / total;
        }
    }
    
    free(c);
    c = NULL;
}

void baum_welch_reduced_flops(double *a, double* alpha, double *b, double *beta, double *pi, int *o, int N, int M, int T, double *a_, double *b_, double *pi_){

    double* c = static_cast<double *>(calloc(T, sizeof(double)));
    double* b_beta = static_cast<double *>(malloc((T-1) * N * sizeof(double)));
    double* gamma = static_cast<double *>(calloc(N, sizeof(double)));

    /* 
     * Forward procedure
     */

    // Base case
    for(int i = 0; i < N; i++){
        alpha[i] = pi[i] * b[i*M  + o[0]];
        c[0] += alpha[i];
    }
    c[0] = 1 / c[0];
    
    for(int i = 0; i < N; i++){
        alpha[i] = alpha[i] * c[0];
    }

    // Step case
    for(int t = 0; t < T - 1; t++){
        for(int j = 0; j < N; j++){

            double sum = 0;
            for(int i = 0; i < N; i++){
                sum += alpha[t*N + i] * a[i*N  + j];
            }
            
            alpha[(t+1)*N + j] = sum * b[j*M + o[t+1]];
            c[t+1] += alpha[(t+1)*N + j];
        }
        c[t+1] = 1 / c[t+1];
        
        for(int j = 0; j < N; j++){
            alpha[(t+1)*N + j] *= c[t+1];
        }
    }
    
    /*
     * Backward procedure
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
        for(int j = 0; j < N; j++){
            b_beta[t*N + j] = b[j*M + o[t+1]] * beta[(t+1)*N +j];
            double tmp = b_beta[t*N +j] * c[t];
            for(int i = 0; i < N; i++){
                beta[t*N + i] += tmp * a[i*N  + j];
            }
        }
    }
    
    /*
     * Update pi
     */
    double c_recipr = 1 / c[0];
    for(int i = 0; i < N; i++){
        pi_[i] = alpha[i] * beta[i] * c_recipr;
    }
    
    /*
     * Update A
     */
    for(int t = 0; t < M; t++){
        for(int j = 0; j < N; j++){
            b_[j*M + t] = 0; 
        }
    }
    for(int t = 0; t < T-1; t++){                
        c_recipr = 1 / c[t];
        for(int j = 0; j < N; j++){
            double gamma1 =  alpha[t*N + j] * beta[t*N + j] * c_recipr;
            gamma[j] += gamma1;
            b_[j*M + o[t]] += gamma1;
        }
    }

    double gamma_recipr;
    for(int i = 0; i < N; i++){
        gamma_recipr = 1 / gamma[i];
        for(int j = 0; j < N; j++){
            double zeta = 0;

            for(int t = 0; t < T-1; t++){
                zeta += alpha[t*N + i] * b_beta[t*N +j];
            }
            a_[i*N + j] = zeta * a[i*N  + j] * gamma_recipr;
        }
    }

    
    /*
     *  Update B
     */
    c_recipr = 1 / c[T-1];
    for(int i = 0; i < N; i++){
        double gamma1 = alpha[(T-1)*N + i] * beta[(T-1)*N + i] * c_recipr;
        gamma[i] += gamma1; 
        b_[i*M + o[T-1]] += gamma1;
    }

    for(int j = 0; j < N; j++) {
        gamma_recipr = 1 / gamma[j];
        for(int k = 0; k < M; k++) {
            b_[j*M + k] *= gamma_recipr;
        }
    }
    
    free(c);
    c = NULL;

    free(b_beta);
    b_beta = NULL;

    free(gamma);
    gamma = NULL;
}


void baum_welch_all_row_access_column_major_no_aliasing(double *a, double* alpha, double *b, double *beta, double *pi, int *o, int N, int M, int T, double *a_, double *b_, double *pi_){
    
    // allocate resources
    double* c = static_cast<double *>(calloc(T, sizeof(double)));
    double* c_ = static_cast<double *>(calloc(T, sizeof(double)));
    double* b_beta = static_cast<double *>(malloc((T-1) * N * sizeof(double)));
    double* gamma = static_cast<double *>(calloc(N, sizeof(double)));
    double* gamma_ = static_cast<double *>(malloc(N * sizeof(double)));

    /* 
     *  Forward procedure
     */
     
    // Base case
    int index = o[0];
    double tmp = 0;
    for(int i = 0; i < N; i++){
        double pii = pi[i];
        double bindexNi = b[index*N + i];
        
        alpha[i] = pii * bindexNi;
        tmp += pii * bindexNi;
    }
    c_[0] = tmp;
    c[0] = 1 / tmp;

    for(int i = 0; i < N; i++){
        alpha[i] = alpha[i] * c[0];
    }

    // Step case
    for(int t = 0; t < T - 1; t++){
        index = o[t+1];
        double ctmp = c[t+1];
        for(int j = 0; j < N; j++){
            double sum = 0;
            for(int i = 0; i < N; i++){
                sum += alpha[t*N + i] * a[j*N + i];
            }
            
            alpha[(t+1)*N + j] = sum * b[index*N + j];
            ctmp += sum * b[index*N + j];
        }
        c[t+1] = 1 / ctmp;
        c_[t+1] = ctmp;

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

    for(int t = T - 2; t >= 0; t-=1){
        index = o[t+1];
        double ct = c[t];
        for(int j = 0; j < N; j++){
            double tmp = b[index*N + j] * beta[(t+1)*N +j];
            double tmp2 = tmp * ct;
            b_beta[t*N + j] = tmp;
            for(int i = 0; i < N; i++){
                beta[t*N + i] += tmp2 * a[j*N  + i];
            }
        }
    }
    
    for(int t = 0; t < T; t++){
        for(int i = 0; i < N; i++){
            beta[t*N + i] *= alpha[t*N + i];
        }
    }
    
    /*
     *    Update pi
     */
    double c_recipr = c_[0];
    for (int i = 0; i < N; i++) {
        pi_[i] = beta[i] * c_recipr;
    }
    
    for(int j = 0; j < M; j++){
        for(int i = 0; i < N; i++){  
            b_[j*N + i] = 0;
        }
    }

    for(int t = 0; t < T-1; t++){
        index = o[t];
        double ct = c_[t];
        for(int i = 0; i < N; i++){
            double betatNi = beta[t*N + i];

            b_[index*N + i] += betatNi * ct;
            gamma[i] += betatNi * ct;
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
            double tmp = b_beta[t*N + j];
            for(int i = 0; i < N; i++){
                a_[j*N + i] += alpha[t*N + i] * tmp;
            }
        }
    }
    
    for(int i = 0; i < N; i++){
        gamma_[i] = 1/gamma[i];
    }
    
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            a_[j*N + i] *= a[j*N + i] * gamma_[i];
        }
    }
    
    /*
     *  Update B
     */
    index = o[T-1];
    double c_T = c_[T-1];
    for(int i = 0; i < N; i++){
        double betaTNi = beta[(T-1)*N + i];

        gamma[i] = 1/(gamma[i] + betaTNi * c_T);
        b_[index*N + i] += betaTNi * c_T;
    }
    
    for(int k = 0; k < M; k++){
        for(int i = 0; i < N; i++){
            b_[k*N + i] *= gamma[i];
        }
    }

    free(c);
    c = NULL;
    
    free(c_);
    c_ = NULL;

    free(b_beta);
    b_beta = NULL;

    free(gamma);
    gamma = NULL;

    free(gamma_);
    gamma_ = NULL;
}



void baum_welch_loop_unrolling(double *a, double* alpha, double *b, double *beta, double *pi, int *o, int N, int M, int T, double *a_, double *b_, double *pi_){
    
    // allocate resources
    double* c = static_cast<double *>(calloc(T, sizeof(double)));
    double* c_ = static_cast<double *>(calloc(T, sizeof(double)));
    double* b_beta = static_cast<double *>(malloc((T-1) * N * sizeof(double)));
    double* gamma = static_cast<double *>(calloc(N, sizeof(double)));
    double* gamma_ = static_cast<double *>(malloc(N * sizeof(double)));

    /* 
     *  Forward procedure
     */
     
    // Base case
    int index = o[0];
    double tmp = 0;
    double tmp1 = 0;
    double tmp2 = 0;
    double tmp3 = 0;

    double pii, pii2, pii3, pii1;
    double bindexNi, bindexNi1, bindexNi2, bindexNi3;
    for(int i = 0; i < N; i+=4){
        pii = pi[i];
        pii1 = pi[i+1];
        pii2 = pi[i+2];
        pii3 = pi[i+3];
        
        bindexNi = b[index*N + i];
        bindexNi1 = b[index*N + i+1];
        bindexNi2 = b[index*N + i+2];        
        bindexNi3 = b[index*N + i+3];
        
        alpha[i] = pii * bindexNi;
        alpha[i+1] = pii1 * bindexNi1;
        alpha[i+2] = pii2 * bindexNi2;
        alpha[i+3] = pii3 * bindexNi3;
        
        tmp += pii * bindexNi;
        tmp1 += pii1 * bindexNi1;
        tmp2 += pii2 * bindexNi2;
        tmp3 += pii3 * bindexNi3;
    }
    
    tmp = (tmp+tmp1)+(tmp2+tmp3);
    c_[0] = tmp;
    c[0] = 1 / tmp;

    tmp = c[0];
    for(int i = 0; i < N; i+=8){
        alpha[i] *= tmp;
        alpha[i+1] *= tmp;
        alpha[i+2] *= tmp;
        alpha[i+3] *= tmp;
        alpha[i+4] *= tmp;
        alpha[i+5] *= tmp;
        alpha[i+6] *= tmp;
        alpha[i+7] *= tmp;
    }
    
    // Step case
    double ctmp;
    double sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
    double alphatni0, alphatni1;
    double bindexNj0, bindexNj1, bindexNj2, bindexNj3;
    double bindexNj4, bindexNj5, bindexNj6, bindexNj7;
    
    for(int t = 0; t < T - 1; t++){
        index = o[t+1];
        ctmp = c[t+1];
        
        for(int j = 0; j < N; j+=8){
            sum0 = 0; sum1 = 0; sum2 = 0; sum3 = 0;
            sum4 = 0; sum5 = 0; sum6 = 0; sum7 = 0;
            
            bindexNj0 = b[index*N + j];
            bindexNj1 = b[index*N + j+1];
            bindexNj2 = b[index*N + j+2];
            bindexNj3 = b[index*N + j+3];
            bindexNj4 = b[index*N + j+4];
            bindexNj5 = b[index*N + j+5];
            bindexNj6 = b[index*N + j+6];
            bindexNj7 = b[index*N + j+7];
            
            //Not yet optimal. Looses one cycle every iteration because sum0 is not yet calculated when accessed the second time. unrolling i+= 4 would yield to horizontal adds of sums, which is not ideal too. and j+=16 not possible.
            for(int i = 0; i < N; i+=2){
                alphatni0 = alpha[t*N + i];
                alphatni1 = alpha[t*N + i+1];
                
                sum0 += alphatni0 * a[j*N + i];
                sum1 += alphatni0 * a[(j+1)*N + i];
                sum2 += alphatni0 * a[(j+2)*N + i];
                sum3 += alphatni0 * a[(j+3)*N + i];
                sum4 += alphatni0 * a[(j+4)*N + i];
                sum5 += alphatni0 * a[(j+5)*N + i];
                sum6 += alphatni0 * a[(j+6)*N + i];
                sum7 += alphatni0 * a[(j+7)*N + i];
                
                sum0 += alphatni1 * a[j*N + i+1];
                sum1 += alphatni1 * a[(j+1)*N + i+1];
                sum2 += alphatni1 * a[(j+2)*N + i+1];
                sum3 += alphatni1 * a[(j+3)*N + i+1];
                sum4 += alphatni1 * a[(j+4)*N + i+1];
                sum5 += alphatni1 * a[(j+5)*N + i+1];
                sum6 += alphatni1 * a[(j+6)*N + i+1];
                sum7 += alphatni1 * a[(j+7)*N + i+1];
            }
            
            alpha[(t+1)*N + j] = sum0 * bindexNj0;
            alpha[(t+1)*N + j+1] = sum1 * bindexNj1;
            alpha[(t+1)*N + j+2] = sum2 * bindexNj2;
            alpha[(t+1)*N + j+3] = sum3 * bindexNj3;
            alpha[(t+1)*N + j+4] = sum4 * bindexNj4;
            alpha[(t+1)*N + j+5] = sum5 * bindexNj5;
            alpha[(t+1)*N + j+6] = sum6 * bindexNj6;
            alpha[(t+1)*N + j+7] = sum7 * bindexNj7;
            
            ctmp += sum0 * bindexNj0;
            ctmp += sum1 * bindexNj1;
            ctmp += sum2 * bindexNj2;
            ctmp += sum3 * bindexNj3;
            ctmp += sum4 * bindexNj4;
            ctmp += sum5 * bindexNj5;
            ctmp += sum6 * bindexNj6;
            ctmp += sum7 * bindexNj7;
        }
        
        c[t+1] = 1 / ctmp;
        c_[t+1] = ctmp;

        ctmp = c[t+1];
        //Using FMA --> 2 Mults each cycle, 5 cycles each
        for(int j = 0; j < N; j+=8){
            alpha[(t+1)*N + j] *= ctmp;
            alpha[(t+1)*N + j+1] *= ctmp;
            alpha[(t+1)*N + j+2] *= ctmp;
            alpha[(t+1)*N + j+3] *= ctmp;
            alpha[(t+1)*N + j+4] *= ctmp;
            alpha[(t+1)*N + j+5] *= ctmp;
            alpha[(t+1)*N + j+6] *= ctmp;
            alpha[(t+1)*N + j+7] *= ctmp;
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
    
    double bbetatmp0, bbetatmp1, bbetatmp2, bbetatmp3;
    double bbetactmp0, bbetactmp1, bbetactmp2, bbetactmp3;
    double beta0, beta1, beta2, beta3, beta4, beta5, beta6, beta7;
     
    for(int t = T - 2; t >= 0; t-=1){
        index = o[t+1];
        double ct = c[t];
        for(int j = 0; j < N; j+=4){
            bbetatmp0 = b[index*N + j] * beta[(t+1)*N +j];
            bbetatmp1 = b[index*N + j+1] * beta[(t+1)*N +j+1];
            bbetatmp2 = b[index*N + j+2] * beta[(t+1)*N +j+2];
            bbetatmp3 = b[index*N + j+3] * beta[(t+1)*N +j+3];
            
            bbetactmp0 = bbetatmp0 * ct;
            bbetactmp1 = bbetatmp1 * ct;
            bbetactmp2 = bbetatmp2 * ct;
            bbetactmp3 = bbetatmp3 * ct;
            
            b_beta[t*N + j] = bbetatmp0;
            b_beta[t*N + j+1] = bbetatmp1;
            b_beta[t*N + j+2] = bbetatmp2;
            b_beta[t*N + j+3] = bbetatmp3;
            
            for(int i = 0; i < N; i+=8){
                beta0 = beta[t*N + i];
                beta1 = beta[t*N + i+1];
                beta2 = beta[t*N + i+2];
                beta3 = beta[t*N + i+3];
                beta4 = beta[t*N + i+4];
                beta5 = beta[t*N + i+5];
                beta6 = beta[t*N + i+6];
                beta7 = beta[t*N + i+7];
            
                beta0 += bbetactmp0 * a[j*N  + i];
                beta1 += bbetactmp0 * a[j*N  + i+1];
                beta2 += bbetactmp0 * a[j*N  + i+2];
                beta3 += bbetactmp0 * a[j*N  + i+3];
                beta4 += bbetactmp0 * a[j*N  + i+4];
                beta5 += bbetactmp0 * a[j*N  + i+5];
                beta6 += bbetactmp0 * a[j*N  + i+6];
                beta7 += bbetactmp0 * a[j*N  + i+7];
                
                beta0 += bbetactmp1 * a[(j+1)*N  + i];
                beta1 += bbetactmp1 * a[(j+1)*N  + i+1];
                beta2 += bbetactmp1 * a[(j+1)*N  + i+2];
                beta3 += bbetactmp1 * a[(j+1)*N  + i+3];
                beta4 += bbetactmp1 * a[(j+1)*N  + i+4];
                beta5 += bbetactmp1 * a[(j+1)*N  + i+5];
                beta6 += bbetactmp1 * a[(j+1)*N  + i+6];
                beta7 += bbetactmp1 * a[(j+1)*N  + i+7];
                
                beta0 += bbetactmp2 * a[(j+2)*N  + i];
                beta1 += bbetactmp2 * a[(j+2)*N  + i+1];
                beta2 += bbetactmp2 * a[(j+2)*N  + i+2];
                beta3 += bbetactmp2 * a[(j+2)*N  + i+3];
                beta4 += bbetactmp2 * a[(j+2)*N  + i+4];
                beta5 += bbetactmp2 * a[(j+2)*N  + i+5];
                beta6 += bbetactmp2 * a[(j+2)*N  + i+6];
                beta7 += bbetactmp2 * a[(j+2)*N  + i+7];
                
                beta0 += bbetactmp3 * a[(j+3)*N  + i];
                beta1 += bbetactmp3 * a[(j+3)*N  + i+1];
                beta2 += bbetactmp3 * a[(j+3)*N  + i+2];
                beta3 += bbetactmp3 * a[(j+3)*N  + i+3];
                beta4 += bbetactmp3 * a[(j+3)*N  + i+4];
                beta5 += bbetactmp3 * a[(j+3)*N  + i+5];
                beta6 += bbetactmp3 * a[(j+3)*N  + i+6];
                beta7 += bbetactmp3 * a[(j+3)*N  + i+7];
                
                beta[t*N + i] = beta0;
                beta[t*N + i+1] = beta1;
                beta[t*N + i+2] = beta2;
                beta[t*N + i+3] = beta3;
                beta[t*N + i+4] = beta4;
                beta[t*N + i+5] = beta5;
                beta[t*N + i+6] = beta6;
                beta[t*N + i+7] = beta7;
            }
        }
    }
    
    for(int t = 0; t < T; t+=4){
        for(int i = 0; i < N; i+=4){
            beta[t*N + i] *= alpha[t*N + i];
            beta[t*N + i+1] *= alpha[t*N + i+1];
            beta[t*N + i+2] *= alpha[t*N + i+2];
            beta[t*N + i+3] *= alpha[t*N + i+3];
            
            beta[(t+1)*N + i] *= alpha[(t+1)*N + i];
            beta[(t+1)*N + i+1] *= alpha[(t+1)*N + i+1];
            beta[(t+1)*N + i+2] *= alpha[(t+1)*N + i+2];
            beta[(t+1)*N + i+3] *= alpha[(t+1)*N + i+3];
            
            beta[(t+2)*N + i] *= alpha[(t+2)*N + i];
            beta[(t+2)*N + i+1] *= alpha[(t+2)*N + i+1];
            beta[(t+2)*N + i+2] *= alpha[(t+2)*N + i+2];
            beta[(t+2)*N + i+3] *= alpha[(t+2)*N + i+3];
            
            beta[(t+3)*N + i] *= alpha[(t+3)*N + i];
            beta[(t+3)*N + i+1] *= alpha[(t+3)*N + i+1];
            beta[(t+3)*N + i+2] *= alpha[(t+3)*N + i+2];
            beta[(t+3)*N + i+3] *= alpha[(t+3)*N + i+3];
        }
    }
    
    /*
     *    Update pi
     */
    double c_recipr = c_[0];
    for (int i = 0; i < N; i+=4) {
        pi_[i] = beta[i] * c_recipr;
        pi_[i+1] = beta[i+1] * c_recipr;
        pi_[i+2] = beta[i+2] * c_recipr;
        pi_[i+3] = beta[i+3] * c_recipr;
    }
    
    for(int j = 0; j < M; j++){
        for(int i = 0; i < N; i++){  
            b_[j*N + i] = 0;
        }
    }

    //TODO unroll outer with some condition?
    
    double betatNi0, betatNi1, betatNi2, betatNi3, betatNi4;
    for(int t = 0; t < T-1; t++){
        index = o[t];
        double ct = c_[t];
        for(int i = 0; i < N; i+=4){

            betatNi0 = beta[t*N + i];
            b_[index*N + i] += betatNi0 * ct;
            gamma[i] += betatNi0 * ct;

            betatNi1 = beta[t*N + i + 1];
            b_[index*N + i + 1] += betatNi1 * ct;
            gamma[i + 1] += betatNi1 * ct;

            betatNi2 = beta[t*N + i + 2];
            b_[index*N + i + 2] += betatNi2 * ct;
            gamma[i + 2] += betatNi2 * ct;

            betatNi3 = beta[t*N + i + 3];
            b_[index*N + i + 3] += betatNi3 * ct;
            gamma[i + 3] += betatNi3 * ct;
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
    
    double alphati0, alphati1, alphati2, alphati3;
    double temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
    for(int t = 0; t < T-1; t++){
        for(int j = 0; j < N; j+=4){
            temp0 = b_beta[t*N + j];
            temp1 = b_beta[t*N + j+1];
            temp2 = b_beta[t*N + j+2];
            temp3 = b_beta[t*N + j+3];
            for(int i = 0; i < N; i+=4){
                alphati0 = alpha[t*N + i];
                alphati1 = alpha[t*N + i + 1];
                alphati2 = alpha[t*N + i + 2];
                alphati3 = alpha[t*N + i + 3];

                a_[j*N + i] += alphati0 * temp0;
                a_[j*N + i + 1] += alphati1 * temp0;
                a_[j*N + i + 2] += alphati2 * temp0;
                a_[j*N + i + 3] += alphati3 * temp0;

                a_[(j+1)*N + i] += alphati0 * temp1;
                a_[(j+1)*N + i + 1] += alphati1 * temp1;
                a_[(j+1)*N + i + 2] += alphati2 * temp1;
                a_[(j+1)*N + i + 3] += alphati3 * temp1;

                a_[(j+2)*N + i] += alphati0 * temp2;
                a_[(j+2)*N + i + 1] += alphati1 * temp2;
                a_[(j+2)*N + i + 2] += alphati2 * temp2;
                a_[(j+2)*N + i + 3] += alphati3 * temp2;

                a_[(j+3)*N + i] += alphati0 * temp3;
                a_[(j+3)*N + i + 1] += alphati1 * temp3;
                a_[(j+3)*N + i + 2] += alphati2 * temp3;
                a_[(j+3)*N + i + 3] += alphati3 * temp3;
            }
        }
    }
    
    for(int i = 0; i < N; i+=8){
        gamma_[i] = 1/gamma[i];
        gamma_[i+1] = 1/gamma[i+1];
        gamma_[i+2] = 1/gamma[i+2];
        gamma_[i+3] = 1/gamma[i+3];
        gamma_[i+4] = 1/gamma[i+4];
        gamma_[i+5] = 1/gamma[i+5];
        gamma_[i+6] = 1/gamma[i+6];
        gamma_[i+7] = 1/gamma[i+7];
    }
    
    double tmpgammai0, tmpgammai1, tmpgammai2, tmpgammai3;
    for(int j = 0; j < N; j+=4){
        for(int i = 0; i < N; i+=4){
            tmpgammai0 = gamma_[i];
            tmpgammai1 = gamma_[i+1];
            tmpgammai2 = gamma_[i+2];
            tmpgammai3 = gamma_[i+3];
            a_[j*N + i] *= a[j*N + i] * tmpgammai0;
            a_[j*N + i+1] *= a[j*N + i+1] * tmpgammai1;
            a_[j*N + i+2] *= a[j*N + i+2] * tmpgammai2;
            a_[j*N + i+3] *= a[j*N + i+3] * tmpgammai3;

            a_[(j + 1)*N + i] *= a[(j + 1)*N + i] * tmpgammai0;
            a_[(j + 1)*N + i+1] *= a[(j + 1)*N + i+1] * tmpgammai1;
            a_[(j + 1)*N + i+2] *= a[(j + 1)*N + i+2] * tmpgammai2;
            a_[(j + 1)*N + i+3] *= a[(j + 1)*N + i+3] * tmpgammai3;

            a_[(j + 2)*N + i] *= a[(j + 2)*N + i] * tmpgammai0;
            a_[(j + 2)*N + i+1] *= a[(j + 2)*N + i+1] * tmpgammai1;
            a_[(j + 2)*N + i+2] *= a[(j + 2)*N + i+2] * tmpgammai2;
            a_[(j + 2)*N + i+3] *= a[(j + 2)*N + i+3] * tmpgammai3;

            a_[(j + 3)*N + i] *= a[(j + 3)*N + i] * tmpgammai0;
            a_[(j + 3)*N + i+1] *= a[(j + 3)*N + i+1] * tmpgammai1;
            a_[(j + 3)*N + i+2] *= a[(j + 3)*N + i+2] * tmpgammai2;
            a_[(j + 3)*N + i+3] *= a[(j + 3)*N + i+3] * tmpgammai3;
        }
    }
    
    /*
     *  Update B
     */
    index = o[T-1];
    double c_T = c_[T-1];
    double betaTNi0, betaTNi1, betaTNi2, betaTNi3;
    for(int i = 0; i < N; i+=4){
        betaTNi0 = beta[(T-1)*N + i];
        gamma[i] = 1/(gamma[i] + betaTNi0 * c_T);
        b_[index*N + i] += betaTNi0 * c_T;

        betaTNi1 = beta[(T-1)*N + i+1];
        gamma[i+1] = 1/(gamma[i+1] + betaTNi1 * c_T);
        b_[index*N + i+1] += betaTNi1 * c_T;

        betaTNi2 = beta[(T-1)*N + i+2];
        gamma[i+2] = 1/(gamma[i+2] + betaTNi2 * c_T);
        b_[index*N + i+2] += betaTNi2 * c_T;

        betaTNi3 = beta[(T-1)*N + i+3];
        gamma[i+3] = 1/(gamma[i+3] + betaTNi3 * c_T);
        b_[index*N + i+3] += betaTNi3 * c_T;

    }
    
    double tempGammaI0, tempGammaI1, tempGammaI2, tempGammaI3;
    for(int k = 0; k < M; k+=4){
        for(int i = 0; i < N; i+=4){
            tempGammaI0 = gamma[i];
            tempGammaI1 = gamma[i+1];
            tempGammaI2 = gamma[i+2];
            tempGammaI3 = gamma[i+3];
            b_[k*N + i] *= tempGammaI0;
            b_[k*N + i+1] *= tempGammaI1;
            b_[k*N + i+2] *= tempGammaI2;
            b_[k*N + i+3] *= tempGammaI3;

            b_[(k + 1)*N + i] *= tempGammaI0;
            b_[(k + 1)*N + i+1] *= tempGammaI1;
            b_[(k + 1)*N + i+2] *= tempGammaI2;
            b_[(k + 1)*N + i+3] *= tempGammaI3;

            b_[(k + 2)*N + i] *= tempGammaI0;
            b_[(k + 2)*N + i+1] *= tempGammaI1;
            b_[(k + 2)*N + i+2] *= tempGammaI2;
            b_[(k + 2)*N + i+3] *= tempGammaI3;

            b_[(k + 3)*N + i] *= tempGammaI0;
            b_[(k + 3)*N + i+1] *= tempGammaI1;
            b_[(k + 3)*N + i+2] *= tempGammaI2;
            b_[(k + 3)*N + i+3] *= tempGammaI3;
        }
    }

    free(c);
    c = NULL;
    
    free(c_);
    c_ = NULL;

    free(b_beta);
    b_beta = NULL;

    free(gamma);
    gamma = NULL;

    free(gamma_);
    gamma_ = NULL;
}

void baum_welch_simd(double *a, double* alpha, double *b, double *beta, double *pi, int *o, int N, int M, int T, double *a_, double *b_, double *pi_){
    
    // allocate resources
    double* c = static_cast<double *>(calloc(T, sizeof(double)));
    double* c_ = static_cast<double *>(calloc(T, sizeof(double)));
    double* b_beta = static_cast<double *>(aligned_alloc(32, (T-1) * N * sizeof(double)));
    double* gamma = static_cast<double *>(calloc(N, sizeof(double)));
    double* gamma_ = static_cast<double *>(aligned_alloc(32, N * sizeof(double)));


    __m256d zero_vec = _mm256_set1_pd(0);
    __m256d ones_vec = _mm256_set1_pd(1);

    /* 
     *  Forward procedure
     */
     
    // Base case
    int index = o[0];

    __m256d tmps0 = zero_vec;
    __m256d tmps1 = zero_vec;
    __m256d piis0, piis1;
    __m256d bindexNis0, bindexNis1;
    __m256d alphas0, alphas1;

    for(int i = 0; i < N; i+=8){
        //exececution of the computation for an 8 unrolling
        //8 because it was 4 and want to do horizontal sum on tmp

        piis0 = _mm256_load_pd(pi+i);
        bindexNis0 = _mm256_load_pd(b+index*N+i);
        alphas0 = _mm256_mul_pd(bindexNis0, piis0);
        _mm256_store_pd(alpha+i, alphas0);
        tmps0 = _mm256_fmadd_pd(piis0, bindexNis0, tmps0);

        piis1 = _mm256_load_pd(pi+i+4);
        bindexNis1 = _mm256_load_pd(b+index*N+i+4);
        alphas1 = _mm256_mul_pd(bindexNis1, piis1);
        _mm256_store_pd(alpha+i+4, alphas1);
        tmps1 = _mm256_fmadd_pd(piis1, bindexNis1, tmps1);
    }

    __m256d sum = _mm256_hadd_pd(tmps0, tmps1);
    __m128d sum_high = _mm256_extractf128_pd(sum, 1);
    __m128d result = _mm_add_pd(sum_high, _mm256_castpd256_pd128(sum));
    
    double tmp = result[0] + result[1];

    c_[0] = tmp;

    double tmp_r = 1 / tmp;
    c[0] = tmp_r;
    __m256d tmps = _mm256_set1_pd(tmp_r);

    for(int i = 0; i < N; i+=8){
        alphas0 = _mm256_load_pd(alpha+i);
        alphas1 = _mm256_load_pd(alpha+i+4);

        alphas0 = _mm256_mul_pd(alphas0,tmps);
        alphas1 = _mm256_mul_pd(alphas1,tmps); 

        _mm256_store_pd(alpha+i, alphas0);
        _mm256_store_pd(alpha+i+4, alphas1);
    }
    
    // Step case
    double ctmp;

    __m256d alphatnis00, alphatnis01;
    __m256d sums00, sums01, sums10, sums11;
    __m256d sums20, sums21, sums30, sums31;
    __m256d as00, as01, as10, as11;
    __m256d as20, as21, as30, as31;
    __m256d bindexNjs0;
    __m256d ctmps0;
    __m256d first, second, perm, blend, sumJs;

    for(int t = 0; t < T - 1; t++){
        index = o[t+1];
        ctmp = c[t+1];
        ctmps0 = _mm256_set1_pd(ctmp);
        
        for(int j = 0; j < N; j+=4){
            sums00 = zero_vec;
            sums01 = zero_vec;

            sums10 = zero_vec;
            sums11 = zero_vec;

            sums20 = zero_vec;
            sums21 = zero_vec;

            sums30 = zero_vec;
            sums31 = zero_vec;

            bindexNjs0 = _mm256_load_pd(b + index*N + j);
            
            for(int i = 0; i < N; i+=8){
                alphatnis00 = _mm256_load_pd(alpha+t*N+i);
                alphatnis01 = _mm256_load_pd(alpha+t*N+i+4);

                as00 = _mm256_load_pd(a + j*N + i);
                sums00 = _mm256_fmadd_pd(alphatnis00, as00, sums00);
                as01 = _mm256_load_pd(a + j*N + i+4);
                sums01 = _mm256_fmadd_pd(alphatnis01, as01, sums01);

                as10 = _mm256_load_pd(a + (j+1)*N + i);
                sums10 = _mm256_fmadd_pd(alphatnis00, as10, sums10);
                as11 = _mm256_load_pd(a + (j+1)*N + i+4);
                sums11 = _mm256_fmadd_pd(alphatnis01, as11, sums11);

                as20 = _mm256_load_pd(a + (j+2)*N + i);
                sums20 = _mm256_fmadd_pd(alphatnis00, as20, sums20);
                as21 = _mm256_load_pd(a + (j+2)*N + i+4);
                sums21 = _mm256_fmadd_pd(alphatnis01, as21, sums21);

                as30 = _mm256_load_pd(a + (j+3)*N + i);
                sums30 = _mm256_fmadd_pd(alphatnis00, as30, sums30);
                as31 = _mm256_load_pd(a + (j+3)*N + i+4);
                sums31 = _mm256_fmadd_pd(alphatnis01, as31, sums31);
            }

            sums00 = _mm256_add_pd(sums00, sums01);
            sums10 = _mm256_add_pd(sums10, sums11);
            sums20 = _mm256_add_pd(sums20, sums21);
            sums30 = _mm256_add_pd(sums30, sums31);
            
            first = _mm256_hadd_pd(sums00, sums10);
            second = _mm256_hadd_pd(sums20, sums30);
            blend = _mm256_blend_pd(first, second, 0b1100);
            perm = _mm256_permute2f128_pd(first, second, 0x21);
            sumJs =  _mm256_add_pd(perm, blend);

            _mm256_store_pd(alpha + (t+1)*N + j, _mm256_mul_pd(sumJs, bindexNjs0));

            ctmps0 = _mm256_fmadd_pd(sumJs, bindexNjs0, ctmps0);
        }
        
        ctmps0 = _mm256_hadd_pd(ctmps0,ctmps0);
        ctmp = ctmps0[0]+ctmps0[2];
        //ctmp = ctmps0[0]+ctmps0[1]+ctmps0[2]+ctmps0[3];
        c_[t+1] = ctmp;

        double ctmp_r = 1 / ctmp;
        c[t+1] = ctmp_r;
        ctmps0 = _mm256_set1_pd(ctmp_r);
        for(int j = 0; j < N; j+=8){
            alphatnis00 = _mm256_load_pd(alpha+(t+1)*N+j);
            alphatnis01 = _mm256_load_pd(alpha+(t+1)*N+j+4);

            alphatnis00 = _mm256_mul_pd(alphatnis00, ctmps0);
            alphatnis01 = _mm256_mul_pd(alphatnis01, ctmps0);

            _mm256_store_pd(alpha + (t+1)*N + j, alphatnis00);
            _mm256_store_pd(alpha + (t+1)*N + j+4, alphatnis01);
        }
    }

    /*
        Backward prodcedure
    */
    
    // Base case
    __m256d ct_vec = _mm256_broadcast_sd(c + T-1);
    for(int i = 0; i < N; i+=4) {
        _mm256_store_pd(beta + (T-1)*N + i, ct_vec);
    }

    // Step case
    for(int t = T - 2; t >= 0; t--){
        for(int i = 0; i < N; i++) {
            beta[t*N + i] = 0;
        }
    }
    
    __m256d a_vec01, a_vec02, a_vec11, a_vec12, a_vec21, a_vec22, a_vec31, a_vec32, alpha_vec1, alpha_vec2, alpha_vec3, alpha_vec4, 
            alpha_beta_vec1, alpha_beta_vec2, alpha_beta_vec3, alpha_beta_vec4, beta_vec, beta_vec1, beta_vec2, beta_vec3, beta_vec4, 
            b_vec, b_beta_vec, b_beta_ct_vec, b_beta_ct_vec1, b_beta_ct_vec2, b_beta_ct_vec3, b_beta_ct_vec4;
    for(int t = T - 2; t >= 0; t-=1){
        index = o[t+1];
        ct_vec = _mm256_broadcast_sd(c + t);
        for(int j = 0; j < N; j+=4){
            
            b_vec = _mm256_load_pd(b + index*N + j);
            beta_vec = _mm256_load_pd(beta + (t+1)*N + j);
            b_beta_vec = _mm256_mul_pd(b_vec, beta_vec);
            b_beta_ct_vec = _mm256_mul_pd(b_beta_vec, ct_vec);
            _mm256_store_pd(b_beta + t*N + j, b_beta_vec);

            b_beta_ct_vec1 = _mm256_permute4x64_pd(b_beta_ct_vec, 0);
            b_beta_ct_vec2 = _mm256_permute4x64_pd(b_beta_ct_vec, 85);
            b_beta_ct_vec3 = _mm256_permute4x64_pd(b_beta_ct_vec, 170);
            b_beta_ct_vec4 = _mm256_permute4x64_pd(b_beta_ct_vec, 255);

            for(int i = 0; i < N; i+=8){
                beta_vec1 = _mm256_load_pd(beta + t*N + i);
                beta_vec2 = _mm256_load_pd(beta + t*N + i+4);
                a_vec01 = _mm256_load_pd(a + j*N + i);
                a_vec02 = _mm256_load_pd(a + j*N + i+4);
                a_vec11 = _mm256_load_pd(a + (j+1)*N + i);
                a_vec12 = _mm256_load_pd(a + (j+1)*N + i+4);
                a_vec21 = _mm256_load_pd(a + (j+2)*N + i);
                a_vec22 = _mm256_load_pd(a + (j+2)*N + i+4);
                a_vec31 = _mm256_load_pd(a + (j+3)*N + i);
                a_vec32 = _mm256_load_pd(a + (j+3)*N + i+4);

                beta_vec1 = _mm256_fmadd_pd(b_beta_ct_vec1, a_vec01, beta_vec1);
                beta_vec2 = _mm256_fmadd_pd(b_beta_ct_vec1, a_vec02, beta_vec2);

                beta_vec1 = _mm256_fmadd_pd(b_beta_ct_vec2, a_vec11, beta_vec1);
                beta_vec2 = _mm256_fmadd_pd(b_beta_ct_vec2, a_vec12, beta_vec2);

                beta_vec1 = _mm256_fmadd_pd(b_beta_ct_vec3, a_vec21, beta_vec1);
                beta_vec2 = _mm256_fmadd_pd(b_beta_ct_vec3, a_vec22, beta_vec2);

                beta_vec1 = _mm256_fmadd_pd(b_beta_ct_vec4, a_vec31, beta_vec1);
                beta_vec2 = _mm256_fmadd_pd(b_beta_ct_vec4, a_vec32, beta_vec2);

                _mm256_store_pd(beta + t*N + i, beta_vec1);
                _mm256_store_pd(beta + t*N + i+4, beta_vec2);
            }
        }
    }
    
    for(int t = 0; t < T; t+=4){
        for(int i = 0; i < N; i+=4){
            alpha_vec1 = _mm256_load_pd(alpha + t*N + i);
            beta_vec1 = _mm256_load_pd(beta + t*N + i);

            alpha_vec2 = _mm256_load_pd(alpha + (t+1)*N + i);
            beta_vec2 = _mm256_load_pd(beta + (t+1)*N + i);

            alpha_vec3 = _mm256_load_pd(alpha + (t+2)*N + i);
            beta_vec3 = _mm256_load_pd(beta + (t+2)*N + i);

            alpha_vec4 = _mm256_load_pd(alpha + (t+3)*N + i);
            beta_vec4 = _mm256_load_pd(beta + (t+3)*N + i);

            beta_vec1 = _mm256_mul_pd(alpha_vec1, beta_vec1);
            beta_vec2 = _mm256_mul_pd(alpha_vec2, beta_vec2);
            beta_vec3 = _mm256_mul_pd(alpha_vec3, beta_vec3);
            beta_vec4 = _mm256_mul_pd(alpha_vec4, beta_vec4);

            _mm256_store_pd(beta + t*N + i, beta_vec1);
            _mm256_store_pd(beta + (t+1)*N + i, beta_vec2);
            _mm256_store_pd(beta + (t+2)*N + i, beta_vec3);
            _mm256_store_pd(beta + (t+3)*N + i, beta_vec4);
        }
    }
    
    /*
     *    Update pi
     */
    __m256d c_recipr_vec = _mm256_broadcast_sd(c_);
    __m256d pi_vec, tmp_vec;
    for (int i = 0; i < N; i+=4) {
        pi_vec   = _mm256_load_pd(pi + i);
        beta_vec = _mm256_load_pd(beta + i);
        tmp_vec  = _mm256_mul_pd(beta_vec, c_recipr_vec);
        _mm256_store_pd(pi_ + i, tmp_vec);
    }
    
    for(int j = 0; j < M; j+=4){
        for(int i = 0; i < N; i+=4){  
            _mm256_store_pd(b_+j*N+i, zero_vec);
            _mm256_store_pd(b_+(j+1)*N+i, zero_vec);
            _mm256_store_pd(b_+(j+2)*N+i, zero_vec);
            _mm256_store_pd(b_+(j+3)*N+i, zero_vec);
        }
    }

    //TODO unroll outer with some condition?
    
    for(int t = 0; t < T-1; t++){
        index = o[t];

        __m256d ct_vec = _mm256_broadcast_sd(c_+t);
        
        for(int i = 0; i < N; i+=4){

            __m256d betatNi_vec = _mm256_load_pd(beta+t*N + i);
            __m256d b_vec       = _mm256_load_pd(b_+index*N + i);
            __m256d gamma_vec   = _mm256_loadu_pd(gamma + i);
            
            __m256d tmp0_vec    = _mm256_fmadd_pd(betatNi_vec, ct_vec, b_vec);
            __m256d tmp1_vec    = _mm256_fmadd_pd(betatNi_vec, ct_vec, gamma_vec);

            _mm256_store_pd(b_+index*N + i, tmp0_vec);
            _mm256_storeu_pd(gamma + i, tmp1_vec);
        }
    }
    
    /*
     *  Update A
     */
    for(int i = 0; i < N; i+=4){
        for(int j = 0; j < N; j+=4){
            _mm256_store_pd(a_+j*N+i, zero_vec);
            _mm256_store_pd(a_+(j+1)*N+i, zero_vec);
            _mm256_store_pd(a_+(j+2)*N+i, zero_vec);
            _mm256_store_pd(a_+(j+3)*N+i, zero_vec);
        }
    }
    
    for(int t = 0; t < T-1; t++){
        for(int j = 0; j < N; j+=4){
            __m256d b_betatN0 = _mm256_broadcast_sd(b_beta+t*N+j);
            __m256d b_betatN1 = _mm256_broadcast_sd(b_beta+t*N+j+1);
            __m256d b_betatN2 = _mm256_broadcast_sd(b_beta+t*N+j+2);
            __m256d b_betatN3 = _mm256_broadcast_sd(b_beta+t*N+j+3);

            for(int i = 0; i < N; i+=4){
                __m256d alpha_vec = _mm256_load_pd(alpha+t*N+i);

                __m256d a_vec0 = _mm256_load_pd(a_+j*N+i);
                __m256d a_vec1 = _mm256_load_pd(a_+(j+1)*N+i);
                __m256d a_vec2 = _mm256_load_pd(a_+(j+2)*N+i);
                __m256d a_vec3 = _mm256_load_pd(a_+(j+3)*N+i);

                a_vec0 = _mm256_fmadd_pd(alpha_vec, b_betatN0, a_vec0);
                a_vec1 = _mm256_fmadd_pd(alpha_vec, b_betatN1, a_vec1);
                a_vec2 = _mm256_fmadd_pd(alpha_vec, b_betatN2, a_vec2);
                a_vec3 = _mm256_fmadd_pd(alpha_vec, b_betatN3, a_vec3);

                _mm256_store_pd(a_+j*N+i, a_vec0);
                _mm256_store_pd(a_+(j+1)*N+i, a_vec1);
                _mm256_store_pd(a_+(j+2)*N+i, a_vec2);
                _mm256_store_pd(a_+(j+3)*N+i, a_vec3);
            }
        }
    }
    
    
    for(int i = 0; i < N; i+=4){
        __m256d gamma_vec = _mm256_loadu_pd(gamma+i);
        
        gamma_vec = _mm256_div_pd(ones_vec, gamma_vec);
        
        _mm256_storeu_pd(gamma_+i, gamma_vec);
    }

    __m256d tmpgamma_vec;
    __m256d a_vec0, a_vec1, a_vec2, a_vec3;
    __m256d a__vec0, a__vec1, a__vec2, a__vec3;
    for(int j = 0; j < N; j+=4){

    __m256d tmp_vec0, tmp_vec1, tmp_vec2, tmp_vec3;
        for(int i = 0; i < N; i+=4){

            tmpgamma_vec = _mm256_load_pd(gamma_ + i);

            a_vec0   = _mm256_load_pd(a + j*N + i);
            a_vec1   = _mm256_load_pd(a + (j+1)*N + i);
            a_vec2   = _mm256_load_pd(a + (j+2)*N + i);
            a_vec3   = _mm256_load_pd(a + (j+3)*N + i);
            
            tmp_vec0 = _mm256_mul_pd(a_vec0, tmpgamma_vec);
            tmp_vec1 = _mm256_mul_pd(a_vec1, tmpgamma_vec);
            tmp_vec2 = _mm256_mul_pd(a_vec2, tmpgamma_vec);
            tmp_vec3 = _mm256_mul_pd(a_vec3, tmpgamma_vec);

            a__vec0  = _mm256_load_pd(a_ + j*N + i);
            a__vec1  = _mm256_load_pd(a_ + (j+1)*N + i);
            a__vec2  = _mm256_load_pd(a_ + (j+2)*N + i);
            a__vec3  = _mm256_load_pd(a_ + (j+3)*N + i);
            
            tmp_vec0 = _mm256_mul_pd(a__vec0, tmp_vec0);
            tmp_vec1 = _mm256_mul_pd(a__vec1, tmp_vec1);
            tmp_vec2 = _mm256_mul_pd(a__vec2, tmp_vec2);
            tmp_vec3 = _mm256_mul_pd(a__vec3, tmp_vec3);

            _mm256_store_pd(a_+j*N+i, tmp_vec0);
            _mm256_store_pd(a_+(j+1)*N+i, tmp_vec1);
            _mm256_store_pd(a_+(j+2)*N+i, tmp_vec2);
            _mm256_store_pd(a_+(j+3)*N+i, tmp_vec3);
        }
    }
    
    /*
     *  Update B
     */
    index = o[T-1];
    __m256d cT_vec    = _mm256_broadcast_sd(c_+T-1);
    for(int i = 0; i < N; i+=4){
        __m256d beta_vec  = _mm256_load_pd(beta+(T-1)*N+i);
        __m256d b_vec     = _mm256_load_pd(b_+index*N + i);
        __m256d gamma_vec = _mm256_loadu_pd(gamma + i);

        __m256d tmp0_vec = _mm256_fmadd_pd(beta_vec, cT_vec, gamma_vec);
        tmp0_vec = _mm256_div_pd(ones_vec, tmp0_vec);

        __m256d tmp1_vec = _mm256_fmadd_pd(beta_vec, cT_vec, b_vec);

        _mm256_storeu_pd(gamma+i, tmp0_vec);
        _mm256_store_pd(b_+index*N+i, tmp1_vec);
    }
    
    double tempGammaI0, tempGammaI1, tempGammaI2, tempGammaI3;
    for(int k = 0; k < M; k+=4){
        for(int i = 0; i < N; i+=4){
            __m256d b__vec0     = _mm256_load_pd(b_+k*N + i);
            __m256d b__vec1     = _mm256_load_pd(b_+(k+1)*N + i);
            __m256d b__vec2     = _mm256_load_pd(b_+(k+2)*N + i);
            __m256d b__vec3     = _mm256_load_pd(b_+(k+3)*N + i);

            __m256d gamma_vec   = _mm256_loadu_pd(gamma + i);

            b__vec0 = _mm256_mul_pd(b__vec0, gamma_vec);
            b__vec1 = _mm256_mul_pd(b__vec1, gamma_vec);
            b__vec2 = _mm256_mul_pd(b__vec2, gamma_vec);
            b__vec3 = _mm256_mul_pd(b__vec3, gamma_vec);

            _mm256_store_pd(b_+k*N+i, b__vec0);
            _mm256_store_pd(b_+(k+1)*N+i, b__vec1);
            _mm256_store_pd(b_+(k+2)*N+i, b__vec2);
            _mm256_store_pd(b_+(k+3)*N+i, b__vec3);
        }
    }

    free(c);
    c = NULL;
    
    free(c_);
    c_ = NULL;

    free(b_beta);
    b_beta = NULL;

    free(gamma);
    gamma = NULL;

    free(gamma_);
    gamma_ = NULL;
}


void baum_welch_simd_blocking(double *a, double* alpha, double *b, double *beta, double *pi, int *o, int N, int M, int T, double *a_, double *b_, double *pi_){
    
    // allocate resources
    double* c = static_cast<double *>(calloc(T, sizeof(double)));
    double* c_ = static_cast<double *>(calloc(T, sizeof(double)));
    double* b_beta = static_cast<double *>(aligned_alloc(32, (T-1) * N * sizeof(double)));
    double* gamma = static_cast<double *>(calloc(N, sizeof(double)));
    double* gamma_ = static_cast<double *>(aligned_alloc(32, N * sizeof(double)));


    __m256d zero_vec = _mm256_set1_pd(0);
    __m256d ones_vec = _mm256_set1_pd(1);

    /* 
     *  Forward procedure
     */
     
    // Base case
    int index = o[0];

    __m256d tmps0 = zero_vec;
    __m256d tmps1 = zero_vec;
    __m256d piis0, piis1;
    __m256d bindexNis0, bindexNis1;
    __m256d alphas0, alphas1;

    for(int i = 0; i < N; i+=8){
        //exececution of the computation for an 8 unrolling
        //8 because it was 4 and want to do horizontal sum on tmp

        piis0 = _mm256_load_pd(pi+i);
        bindexNis0 = _mm256_load_pd(b+index*N+i);
        alphas0 = _mm256_mul_pd(bindexNis0, piis0);
        _mm256_store_pd(alpha+i, alphas0);
        tmps0 = _mm256_fmadd_pd(piis0, bindexNis0, tmps0);

        piis1 = _mm256_load_pd(pi+i+4);
        bindexNis1 = _mm256_load_pd(b+index*N+i+4);
        alphas1 = _mm256_mul_pd(bindexNis1, piis1);
        _mm256_store_pd(alpha+i+4, alphas1);
        tmps1 = _mm256_fmadd_pd(piis1, bindexNis1, tmps1);
    }

    __m256d sum = _mm256_hadd_pd(tmps0, tmps1);
    __m128d sum_high = _mm256_extractf128_pd(sum, 1);
    __m128d result = _mm_add_pd(sum_high, _mm256_castpd256_pd128(sum));
    
    double tmp = result[0] + result[1];

    c_[0] = tmp;

    double tmp_r = 1 / tmp;
    c[0] = tmp_r;
    __m256d tmps = _mm256_set1_pd(tmp_r);
    for(int i = 0; i < N; i+=8){
        alphas0 = _mm256_load_pd(alpha+i);
        alphas1 = _mm256_load_pd(alpha+i+4);

        alphas0 = _mm256_mul_pd(alphas0,tmps);
        alphas1 = _mm256_mul_pd(alphas1,tmps); 

        _mm256_store_pd(alpha+i, alphas0);
        _mm256_store_pd(alpha+i+4, alphas1);
    }
    
    // Step case
    double ctmp;

    __m256d alphatnis00, alphatnis01;
    __m256d sums00, sums01, sums10, sums11;
    __m256d sums20, sums21, sums30, sums31;
    __m256d as00, as01, as10, as11;
    __m256d as20, as21, as30, as31;
    __m256d bindexNjs0;
    __m256d ctmps0;
    __m256d first, second, perm, blend, sumJs;

    for(int t = 0; t < T - 1; t++){
        index = o[t+1];
        ctmp = c[t+1];
        ctmps0 = _mm256_set1_pd(ctmp);
        
        for(int j = 0; j < N; j+=4){
            sums00 = zero_vec;
            sums01 = zero_vec;

            sums10 = zero_vec;
            sums11 = zero_vec;

            sums20 = zero_vec;
            sums21 = zero_vec;

            sums30 = zero_vec;
            sums31 = zero_vec;

            bindexNjs0 = _mm256_load_pd(b + index*N + j);
            
            for(int i = 0; i < N; i+=8){
                alphatnis00 = _mm256_load_pd(alpha+t*N+i);
                alphatnis01 = _mm256_load_pd(alpha+t*N+i+4);

                as00 = _mm256_load_pd(a + j*N + i);
                sums00 = _mm256_fmadd_pd(alphatnis00, as00, sums00);
                as01 = _mm256_load_pd(a + j*N + i+4);
                sums01 = _mm256_fmadd_pd(alphatnis01, as01, sums01);

                as10 = _mm256_load_pd(a + (j+1)*N + i);
                sums10 = _mm256_fmadd_pd(alphatnis00, as10, sums10);
                as11 = _mm256_load_pd(a + (j+1)*N + i+4);
                sums11 = _mm256_fmadd_pd(alphatnis01, as11, sums11);

                as20 = _mm256_load_pd(a + (j+2)*N + i);
                sums20 = _mm256_fmadd_pd(alphatnis00, as20, sums20);
                as21 = _mm256_load_pd(a + (j+2)*N + i+4);
                sums21 = _mm256_fmadd_pd(alphatnis01, as21, sums21);

                as30 = _mm256_load_pd(a + (j+3)*N + i);
                sums30 = _mm256_fmadd_pd(alphatnis00, as30, sums30);
                as31 = _mm256_load_pd(a + (j+3)*N + i+4);
                sums31 = _mm256_fmadd_pd(alphatnis01, as31, sums31);
            }

            sums00 = _mm256_add_pd(sums00, sums01);
            sums10 = _mm256_add_pd(sums10, sums11);
            sums20 = _mm256_add_pd(sums20, sums21);
            sums30 = _mm256_add_pd(sums30, sums31);
            
            first = _mm256_hadd_pd(sums00, sums10);
            second = _mm256_hadd_pd(sums20, sums30);
            blend = _mm256_blend_pd(first, second, 0b1100);
            perm = _mm256_permute2f128_pd(first, second, 0x21);
            sumJs =  _mm256_add_pd(perm, blend);

            _mm256_store_pd(alpha + (t+1)*N + j, _mm256_mul_pd(sumJs, bindexNjs0));

            ctmps0 = _mm256_fmadd_pd(sumJs, bindexNjs0, ctmps0);
        }
        
        ctmps0 = _mm256_hadd_pd(ctmps0,ctmps0);
        ctmp = ctmps0[0]+ctmps0[2];
        //ctmp = ctmps0[0]+ctmps0[1]+ctmps0[2]+ctmps0[3];
        c_[t+1] = ctmp;

        double ctmp_r = 1 / ctmp;
        c[t+1] = ctmp_r;
        ctmps0 = _mm256_set1_pd(ctmp_r);
        for(int j = 0; j < N; j+=8){
            alphatnis00 = _mm256_load_pd(alpha+(t+1)*N+j);
            alphatnis01 = _mm256_load_pd(alpha+(t+1)*N+j+4);

            alphatnis00 = _mm256_mul_pd(alphatnis00, ctmps0);
            alphatnis01 = _mm256_mul_pd(alphatnis01, ctmps0);

            _mm256_store_pd(alpha + (t+1)*N + j, alphatnis00);
            _mm256_store_pd(alpha + (t+1)*N + j+4, alphatnis01);
        }
    }

    /*
        Backward prodcedure
    */
    
    // Base case
    __m256d ct_vec = _mm256_broadcast_sd(c + T-1);
    for(int i = 0; i < N; i+=4) {
        _mm256_store_pd(beta + (T-1)*N + i, ct_vec);
    }

    // Step case
    for(int t = T - 2; t >= 0; t--){
        for(int i = 0; i < N; i++) {
            beta[t*N + i] = 0;
        }
    }
    
    __m256d a_vec01, a_vec02, a_vec11, a_vec12, a_vec21, a_vec22, a_vec31, a_vec32, alpha_vec1, alpha_vec2, alpha_vec3, alpha_vec4, 
            alpha_beta_vec1, alpha_beta_vec2, alpha_beta_vec3, alpha_beta_vec4, beta_vec, beta_vec1, beta_vec2, beta_vec3, beta_vec4, 
            b_vec, b_beta_vec, b_beta_ct_vec, b_beta_ct_vec1, b_beta_ct_vec2, b_beta_ct_vec3, b_beta_ct_vec4;
    for(int t = T - 2; t >= 0; t-=1){
        index = o[t+1];
        ct_vec = _mm256_broadcast_sd(c + t);
        for(int j = 0; j < N; j+=4){
            
            b_vec = _mm256_load_pd(b + index*N + j);
            beta_vec = _mm256_load_pd(beta + (t+1)*N + j);
            b_beta_vec = _mm256_mul_pd(b_vec, beta_vec);
            b_beta_ct_vec = _mm256_mul_pd(b_beta_vec, ct_vec);
            _mm256_store_pd(b_beta + t*N + j, b_beta_vec);

            b_beta_ct_vec1 = _mm256_permute4x64_pd(b_beta_ct_vec, 0);
            b_beta_ct_vec2 = _mm256_permute4x64_pd(b_beta_ct_vec, 85);
            b_beta_ct_vec3 = _mm256_permute4x64_pd(b_beta_ct_vec, 170);
            b_beta_ct_vec4 = _mm256_permute4x64_pd(b_beta_ct_vec, 255);

            for(int i = 0; i < N; i+=8){
                beta_vec1 = _mm256_load_pd(beta + t*N + i);
                beta_vec2 = _mm256_load_pd(beta + t*N + i+4);
                a_vec01 = _mm256_load_pd(a + j*N + i);
                a_vec02 = _mm256_load_pd(a + j*N + i+4);
                a_vec11 = _mm256_load_pd(a + (j+1)*N + i);
                a_vec12 = _mm256_load_pd(a + (j+1)*N + i+4);
                a_vec21 = _mm256_load_pd(a + (j+2)*N + i);
                a_vec22 = _mm256_load_pd(a + (j+2)*N + i+4);
                a_vec31 = _mm256_load_pd(a + (j+3)*N + i);
                a_vec32 = _mm256_load_pd(a + (j+3)*N + i+4);

                beta_vec1 = _mm256_fmadd_pd(b_beta_ct_vec1, a_vec01, beta_vec1);
                beta_vec2 = _mm256_fmadd_pd(b_beta_ct_vec1, a_vec02, beta_vec2);

                beta_vec1 = _mm256_fmadd_pd(b_beta_ct_vec2, a_vec11, beta_vec1);
                beta_vec2 = _mm256_fmadd_pd(b_beta_ct_vec2, a_vec12, beta_vec2);

                beta_vec1 = _mm256_fmadd_pd(b_beta_ct_vec3, a_vec21, beta_vec1);
                beta_vec2 = _mm256_fmadd_pd(b_beta_ct_vec3, a_vec22, beta_vec2);

                beta_vec1 = _mm256_fmadd_pd(b_beta_ct_vec4, a_vec31, beta_vec1);
                beta_vec2 = _mm256_fmadd_pd(b_beta_ct_vec4, a_vec32, beta_vec2);

                _mm256_store_pd(beta + t*N + i, beta_vec1);
                _mm256_store_pd(beta + t*N + i+4, beta_vec2);
            }
        }
    }
    
    /*
     *    Update pi
     */
    __m256d c_recipr_vec = _mm256_broadcast_sd(c_);
    __m256d pi_vec, tmp_vec, alpha_vec;
    for (int i = 0; i < N; i+=4) {
        pi_vec   = _mm256_load_pd(pi + i);
        beta_vec = _mm256_load_pd(beta + i);
        alpha_vec = _mm256_load_pd(alpha + i);
        beta_vec = _mm256_mul_pd(beta_vec, alpha_vec);
        tmp_vec  = _mm256_mul_pd(beta_vec, c_recipr_vec);
        _mm256_store_pd(pi_ + i, tmp_vec);
    }
    
    for(int j = 0; j < M; j+=4){
        for(int i = 0; i < N; i+=4){  
            _mm256_store_pd(b_+j*N+i, zero_vec);
            _mm256_store_pd(b_+(j+1)*N+i, zero_vec);
            _mm256_store_pd(b_+(j+2)*N+i, zero_vec);
            _mm256_store_pd(b_+(j+3)*N+i, zero_vec);
        }
    }

    int blocksize = 4;
    for(int t = 0; t < T-blocksize; t+=blocksize){
        for(int i = 0; i < N; i+=8){

            __m256d gamma_vec0   = _mm256_loadu_pd(gamma + i);
            __m256d gamma_vec1   = _mm256_loadu_pd(gamma + i + 4);
            
            // this inner loop can not be unrolled
            // because o[tt] might be equal to o[tt+1]
            // which would not allow reordering of statements
            for(int tt = t; tt < t+blocksize; tt++){
                index = o[tt];
                __m256d ct_vec       = _mm256_broadcast_sd(c_+tt);
                
                __m256d betatNi_vec0 = _mm256_load_pd(beta+tt*N + i);
                __m256d betatNi_vec1 = _mm256_load_pd(beta+tt*N + i + 4);
                __m256d alphatNi_vec0 = _mm256_load_pd(alpha+tt*N + i);
                __m256d alphatNi_vec1 = _mm256_load_pd(alpha+tt*N + i + 4);
                betatNi_vec0 = _mm256_mul_pd(alphatNi_vec0, betatNi_vec0);
                betatNi_vec1 = _mm256_mul_pd(alphatNi_vec1, betatNi_vec1);
                
                __m256d b_vec0       = _mm256_load_pd(b_+index*N + i);
                __m256d b_vec1       = _mm256_load_pd(b_+index*N + i + 4);
                
                __m256d tmp0_vec0    = _mm256_fmadd_pd(betatNi_vec0, ct_vec, b_vec0);
                __m256d tmp0_vec1    = _mm256_fmadd_pd(betatNi_vec1, ct_vec, b_vec1);

                gamma_vec0    = _mm256_fmadd_pd(betatNi_vec0, ct_vec, gamma_vec0);
                gamma_vec1    = _mm256_fmadd_pd(betatNi_vec1, ct_vec, gamma_vec1);

                _mm256_store_pd(b_+index*N + i, tmp0_vec0);
                _mm256_store_pd(b_+index*N + i + 4, tmp0_vec1);
            }
            _mm256_storeu_pd(gamma + i, gamma_vec0);
            _mm256_storeu_pd(gamma + i + 4, gamma_vec1);
        }
    }
    // run remaining iterations
    for(int t = T-blocksize; t < T-1; t++){
        index = o[t];

        __m256d ct_vec = _mm256_broadcast_sd(c_+t);
        
        for(int i = 0; i < N; i+=4){
            __m256d betatNi_vec = _mm256_load_pd(beta+t*N + i);
            __m256d alphatNi_vec = _mm256_load_pd(alpha+t*N + i);
           betatNi_vec = _mm256_mul_pd(betatNi_vec, alphatNi_vec);
            __m256d b_vec       = _mm256_load_pd(b_+index*N + i);
            __m256d gamma_vec   = _mm256_loadu_pd(gamma + i);
            
            __m256d tmp0_vec    = _mm256_fmadd_pd(betatNi_vec, ct_vec, b_vec);
            __m256d tmp1_vec    = _mm256_fmadd_pd(betatNi_vec, ct_vec, gamma_vec);

            _mm256_store_pd(b_+index*N + i, tmp0_vec);
            _mm256_storeu_pd(gamma + i, tmp1_vec);
        }
    }
    
    /*
     *  Update A
     */
    for(int i = 0; i < N; i+=4){
        for(int j = 0; j < N; j+=4){
            _mm256_store_pd(a_+j*N+i, zero_vec);
            _mm256_store_pd(a_+(j+1)*N+i, zero_vec);
            _mm256_store_pd(a_+(j+2)*N+i, zero_vec);
            _mm256_store_pd(a_+(j+3)*N+i, zero_vec);
        }
    }
    
    int t;
    blocksize = 8;
    for(t = 0; t < T-blocksize; t+=blocksize){
        for(int j = 0; j < N; j+=8){
            for(int i = 0; i < N; i+=4){
                __m256d a_vec0 = _mm256_load_pd(a_+j*N+i);
                __m256d a_vec1 = _mm256_load_pd(a_+(j+1)*N+i);
                __m256d a_vec2 = _mm256_load_pd(a_+(j+2)*N+i);
                __m256d a_vec3 = _mm256_load_pd(a_+(j+3)*N+i);
                __m256d a_vec4 = _mm256_load_pd(a_+(j+4)*N+i);
                __m256d a_vec5 = _mm256_load_pd(a_+(j+5)*N+i);
                __m256d a_vec6 = _mm256_load_pd(a_+(j+6)*N+i);
                __m256d a_vec7 = _mm256_load_pd(a_+(j+7)*N+i);
                    
                for(int tt = t; tt < t+blocksize; tt++){
                    __m256d b_betatN0 = _mm256_broadcast_sd(b_beta+tt*N+j);
                    __m256d b_betatN1 = _mm256_broadcast_sd(b_beta+tt*N+j+1);
                    __m256d b_betatN2 = _mm256_broadcast_sd(b_beta+tt*N+j+2);
                    __m256d b_betatN3 = _mm256_broadcast_sd(b_beta+tt*N+j+3);
                    __m256d b_betatN4 = _mm256_broadcast_sd(b_beta+tt*N+j+4);
                    __m256d b_betatN5 = _mm256_broadcast_sd(b_beta+tt*N+j+5);
                    __m256d b_betatN6 = _mm256_broadcast_sd(b_beta+tt*N+j+6);
                    __m256d b_betatN7 = _mm256_broadcast_sd(b_beta+tt*N+j+7);
                
                    __m256d alpha_vec = _mm256_load_pd(alpha+tt*N+i);

                    a_vec0 = _mm256_fmadd_pd(alpha_vec, b_betatN0, a_vec0);
                    a_vec1 = _mm256_fmadd_pd(alpha_vec, b_betatN1, a_vec1);
                    a_vec2 = _mm256_fmadd_pd(alpha_vec, b_betatN2, a_vec2);
                    a_vec3 = _mm256_fmadd_pd(alpha_vec, b_betatN3, a_vec3);
                    a_vec4 = _mm256_fmadd_pd(alpha_vec, b_betatN4, a_vec4);
                    a_vec5 = _mm256_fmadd_pd(alpha_vec, b_betatN5, a_vec5);
                    a_vec6 = _mm256_fmadd_pd(alpha_vec, b_betatN6, a_vec6);
                    a_vec7 = _mm256_fmadd_pd(alpha_vec, b_betatN7, a_vec7);
                }
                
                _mm256_store_pd(a_+j*N+i, a_vec0);
                _mm256_store_pd(a_+(j+1)*N+i, a_vec1);
                _mm256_store_pd(a_+(j+2)*N+i, a_vec2);
                _mm256_store_pd(a_+(j+3)*N+i, a_vec3);
                _mm256_store_pd(a_+(j+4)*N+i, a_vec4);
                _mm256_store_pd(a_+(j+5)*N+i, a_vec5);
                _mm256_store_pd(a_+(j+6)*N+i, a_vec6);
                _mm256_store_pd(a_+(j+7)*N+i, a_vec7);
            }
        }
    }


    for(t; t < T-1; t++){
        for(int j = 0; j < N; j+=4){
            __m256d b_betatN0 = _mm256_broadcast_sd(b_beta+t*N+j);
            __m256d b_betatN1 = _mm256_broadcast_sd(b_beta+t*N+j+1);
            __m256d b_betatN2 = _mm256_broadcast_sd(b_beta+t*N+j+2);
            __m256d b_betatN3 = _mm256_broadcast_sd(b_beta+t*N+j+3);

            for(int i = 0; i < N; i+=4){
                __m256d alpha_vec = _mm256_load_pd(alpha+t*N+i);

                __m256d a_vec0 = _mm256_load_pd(a_+j*N+i);
                __m256d a_vec1 = _mm256_load_pd(a_+(j+1)*N+i);
                __m256d a_vec2 = _mm256_load_pd(a_+(j+2)*N+i);
                __m256d a_vec3 = _mm256_load_pd(a_+(j+3)*N+i);

                a_vec0 = _mm256_fmadd_pd(alpha_vec, b_betatN0, a_vec0);
                a_vec1 = _mm256_fmadd_pd(alpha_vec, b_betatN1, a_vec1);
                a_vec2 = _mm256_fmadd_pd(alpha_vec, b_betatN2, a_vec2);
                a_vec3 = _mm256_fmadd_pd(alpha_vec, b_betatN3, a_vec3);

                _mm256_store_pd(a_+j*N+i, a_vec0);
                _mm256_store_pd(a_+(j+1)*N+i, a_vec1);
                _mm256_store_pd(a_+(j+2)*N+i, a_vec2);
                _mm256_store_pd(a_+(j+3)*N+i, a_vec3);
            }
        }
    }
    
    for(int i = 0; i < N; i+=4){
        __m256d gamma_vec = _mm256_loadu_pd(gamma+i);
        
        gamma_vec = _mm256_div_pd(ones_vec, gamma_vec);
        
        _mm256_storeu_pd(gamma_+i, gamma_vec);
    }

    __m256d tmpgamma_vec;
    __m256d a_vec0, a_vec1, a_vec2, a_vec3;
    __m256d a__vec0, a__vec1, a__vec2, a__vec3;
    for(int i = 0; i < N; i+=blocksize){

    __m256d tmp_vec0, tmp_vec1, tmp_vec2, tmp_vec3;
        for(int j = 0; j < N; j+=blocksize){
            for (int ii = i; ii<i+blocksize; ii+=4) {
                for (int jj = j; jj<j+blocksize; jj+=4) {
                    tmpgamma_vec = _mm256_load_pd(gamma_ + ii);

                    a_vec0   = _mm256_load_pd(a + jj*N + ii);
                    a_vec1   = _mm256_load_pd(a + (jj+1)*N + ii);
                    a_vec2   = _mm256_load_pd(a + (jj+2)*N + ii);
                    a_vec3   = _mm256_load_pd(a + (jj+3)*N + ii);
                    
                    tmp_vec0 = _mm256_mul_pd(a_vec0, tmpgamma_vec);
                    tmp_vec1 = _mm256_mul_pd(a_vec1, tmpgamma_vec);
                    tmp_vec2 = _mm256_mul_pd(a_vec2, tmpgamma_vec);
                    tmp_vec3 = _mm256_mul_pd(a_vec3, tmpgamma_vec);

                    a__vec0  = _mm256_load_pd(a_ + jj*N + ii);
                    a__vec1  = _mm256_load_pd(a_ + (jj+1)*N + ii);
                    a__vec2  = _mm256_load_pd(a_ + (jj+2)*N + ii);
                    a__vec3  = _mm256_load_pd(a_ + (jj+3)*N + ii);
                    
                    tmp_vec0 = _mm256_mul_pd(a__vec0, tmp_vec0);
                    tmp_vec1 = _mm256_mul_pd(a__vec1, tmp_vec1);
                    tmp_vec2 = _mm256_mul_pd(a__vec2, tmp_vec2);
                    tmp_vec3 = _mm256_mul_pd(a__vec3, tmp_vec3);

                    _mm256_store_pd(a_+jj*N+ii, tmp_vec0);
                    _mm256_store_pd(a_+(jj+1)*N+ii, tmp_vec1);
                    _mm256_store_pd(a_+(jj+2)*N+ii, tmp_vec2);
                    _mm256_store_pd(a_+(jj+3)*N+ii, tmp_vec3);
                }
            }
        }
    }
    
    /*
     *  Update B
     */
    index = o[T-1];
    __m256d cT_vec    = _mm256_broadcast_sd(c_+T-1);
    for(int i = 0; i < N; i+=4){
        __m256d beta_vec  = _mm256_load_pd(beta+(T-1)*N+i);
        __m256d alpha_vec = _mm256_load_pd(alpha+(T-1)*N+i);
        beta_vec = _mm256_mul_pd(alpha_vec, beta_vec);
        
        __m256d b_vec     = _mm256_load_pd(b_+index*N + i);
        __m256d gamma_vec = _mm256_loadu_pd(gamma + i);

        __m256d tmp0_vec = _mm256_fmadd_pd(beta_vec, cT_vec, gamma_vec);
        tmp0_vec = _mm256_div_pd(ones_vec, tmp0_vec);

        __m256d tmp1_vec = _mm256_fmadd_pd(beta_vec, cT_vec, b_vec);

        _mm256_storeu_pd(gamma+i, tmp0_vec);
        _mm256_store_pd(b_+index*N+i, tmp1_vec);
    }
    
    __m256d b__vec0, b__vec1, b__vec2, b__vec3, gamma_vec;
    for(int i = 0; i < N; i+=blocksize){
        for(int k = 0; k < M; k+=blocksize){
            for (int ii = i; ii<i+blocksize; ii+=4) {
                for (int kk = k; kk<k+blocksize; kk+=4) {
                    b__vec0     = _mm256_load_pd(b_+kk*N + ii);
                    b__vec1     = _mm256_load_pd(b_+(kk+1)*N + ii);
                    b__vec2     = _mm256_load_pd(b_+(kk+2)*N + ii);
                    b__vec3     = _mm256_load_pd(b_+(kk+3)*N + ii);

                    gamma_vec   = _mm256_loadu_pd(gamma + ii);

                    b__vec0 = _mm256_mul_pd(b__vec0, gamma_vec);
                    b__vec1 = _mm256_mul_pd(b__vec1, gamma_vec);
                    b__vec2 = _mm256_mul_pd(b__vec2, gamma_vec);
                    b__vec3 = _mm256_mul_pd(b__vec3, gamma_vec);

                    _mm256_store_pd(b_+kk*N+ii, b__vec0);
                    _mm256_store_pd(b_+(kk+1)*N+ii, b__vec1);
                    _mm256_store_pd(b_+(kk+2)*N+ii, b__vec2);
                    _mm256_store_pd(b_+(kk+3)*N+ii, b__vec3);
                }
            }
            
        }
    }

    free(c);
    c = NULL;
    
    free(c_);
    c_ = NULL;

    free(b_beta);
    b_beta = NULL;

    free(gamma);
    gamma = NULL;

    free(gamma_);
    gamma_ = NULL;
}


void register_functions() {
  add_function(&baum_welch_basic, "base");
  add_function(&baum_welch_reduced_flops, "reduced flops");
  add_function(&baum_welch_all_row_access_column_major_no_aliasing, "column major no aliasing");
  //add_function(&baum_welch_loop_unrolling, "loop unrolling");
  add_function(&baum_welch_simd, "simd");
  add_function(&baum_welch_simd_blocking, "blocking");
}
