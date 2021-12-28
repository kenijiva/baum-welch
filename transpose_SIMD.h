#define _MM_TRANSPOSE4_PD(row0,row1,row2,row3)                                 \
                {                                                                \
                    __m256d tmp3, tmp2, tmp1, tmp0;                              \
                                                                                 \
                    tmp0 = _mm256_shuffle_pd((row0),(row1), 0x0);                    \
                    tmp2 = _mm256_shuffle_pd((row0),(row1), 0xF);                \
                    tmp1 = _mm256_shuffle_pd((row2),(row3), 0x0);                    \
                    tmp3 = _mm256_shuffle_pd((row2),(row3), 0xF);                \
                                                                                 \
                    (row0) = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);   \
                    (row1) = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);   \
                    (row2) = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);   \
                    (row3) = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);   \
                }

inline void transpose4x4_SSE(double *A, double *B, const int lda, const int ldb) {
    __m256d row1 = _mm256_loadu_pd(&A[0*lda]);
    __m256d row2 = _mm256_loadu_pd(&A[1*lda]);
    __m256d row3 = _mm256_loadu_pd(&A[2*lda]);
    __m256d row4 = _mm256_loadu_pd(&A[3*lda]);
     _MM_TRANSPOSE4_PD(row1, row2, row3, row4);
     _mm256_storeu_pd(&B[0*ldb], row1);
     _mm256_storeu_pd(&B[1*ldb], row2);
     _mm256_storeu_pd(&B[2*ldb], row3);
     _mm256_storeu_pd(&B[3*ldb], row4);
}


inline void transpose_block_SSE4x4(double *A, double *B, const int n, const int m, const int block_size) {
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            int max_i2 = i+block_size < n ? i + block_size : n;
            int max_j2 = j+block_size < m ? j + block_size : m;
            for(int i2=i; i2<max_i2; i2+=4) {
                for(int j2=j; j2<max_j2; j2+=4) {
                    transpose4x4_SSE(&A[i2*m +j2], &B[j2*n + i2], m, n);
                }
            }
        }
    }
}

inline void transpose_simd(double *v, int N, int M){
    double *transpose = static_cast<double*>(malloc(N* M * sizeof(double)));

    transpose_block_SSE4x4(v, transpose, M, N, N);
    
    memcpy(v, transpose, N * M * sizeof(double));
    free(transpose);
}
