/**
*      _________   _____________________  ____  ______
*     / ____/   | / ___/_  __/ ____/ __ \/ __ \/ ____/
*    / /_  / /| | \__ \ / / / /   / / / / / / / __/
*   / __/ / ___ |___/ // / / /___/ /_/ / /_/ / /___
*  /_/   /_/  |_/____//_/  \____/\____/_____/_____/
*
*  http://www.acl.inf.ethz.ch/teaching/fastcode
*  How to Write Fast Numerical Code 263-2300 - ETH Zurich
*  Copyright (C) 2019 
*                   Tyler Smith        (smitht@inf.ethz.ch) 
*                   Alen Stojanov      (astojanov@inf.ethz.ch)
*                   Gagandeep Singh    (gsingh@inf.ethz.ch)
*                   Markus Pueschel    (pueschel@inf.ethz.ch)
*
*  This program is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program. If not, see http://www.gnu.org/licenses/.
*/
//#include "stdafx.h"

#include <list>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <random>
#include <cstdio>
#include <time.h>
#include <chrono>


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <sstream>
#include <fstream>
#include "tsc_x86.h"
#include "baumwelch.c"

using namespace std;
using namespace std::chrono;

#define CYCLES_REQUIRED 1e8
#define REP 50
#define EPS (1e-16)
#define VALIDATE false

/* prototype of the function you need to optimize */
typedef void(*comp_func)(double *, double *, double*, double*, double*, int*, int, int, int, double*, double*, double*);

//headers
void   register_functions();
double get_perf_score(comp_func f);
double perf_test(comp_func f);


void add_function(comp_func f, string name);

/* Global vars, used to keep track of student functions */
vector<comp_func> userFuncs;
vector<string> funcNames;
int numFuncs = 0;

/*
 * Generate a random m * n matrix where each row is normalized
 */
void gen_rand_mat(double *a, int rows, int cols) {
    for(int i = 0; i < rows; i++){
        double sum = 0;
        for(int j = 0; j < cols; j++){
            double random = (double) rand();
            a[i*cols + j] = random;
            sum += random;
        }

        for(int j = 0; j < cols; j++){
            a[i*cols + j] /= sum;
        }
    }
}

void fill_double_matrix(double **a, int m, int n) {
    *a = static_cast<double *>(aligned_alloc(32, m * n * sizeof(double)));
    for (int i = 0; i < m*n; i++) {
        cin >> *(*a + i);
    }
}

void fill_int_matrix(int **a, int m, int n) {
    *a = static_cast<int *>(aligned_alloc(32, m * n * sizeof(int)));
    for (int i = 0; i < m*n; i++) {
        cin >> *(*a + i);
    }
}

void destroy(double * m){
    free(m);
    m = NULL; //Gives null-pointer exception upon next use
}

/*
* Called by the driver to register your functions
* Use add_function(func, description) to add your own functions
*/
void register_functions();


double nrm_sqr_diff(double *x, double *y, int n){
    double nrm_sqr = 0.0;
    for(int i = 0; i < n; i++) {
        nrm_sqr += (x[i] - y[i]) * (x[i] - y[i]);
    }
    
    if (isnan(nrm_sqr)) {
      nrm_sqr = INFINITY;
    }
    
    return nrm_sqr;
}

/*
* Registers a user function to be tested by the driver program. Registers a
* string description of the function as well
*/
void add_function(comp_func f, string name){
    userFuncs.push_back(f);
    funcNames.emplace_back(name);

    numFuncs++;
}

void validate_base(){
    int num_tests; cin >> num_tests;
    for (int i = 0; i < num_tests; i++) {
        //Load dimensions
        int n, m, t;
        cin >> n >> m >> t;

        // Load inputs
        double *A, *B, *pi;
        int *o;

        fill_double_matrix(&A, n, n);
        fill_double_matrix(&B, n, m);
        fill_double_matrix(&pi, 1, n);
        fill_int_matrix(&o, 1, t);

        // Load solutions
        double *A_, *B_ , *pi_;

        fill_double_matrix(&A_, n, n);
        fill_double_matrix(&B_, n, m);
        fill_double_matrix(&pi_, 1, n);

        double *alpha = static_cast<double *>(aligned_alloc(32, t* n * sizeof(double)));
        double *beta = static_cast<double *>(aligned_alloc(32, t * n * sizeof(double)));

        comp_func f = userFuncs[0];
        f(A, alpha, B, beta, pi, o, n, m, t, A, B, pi);
        double error_a = nrm_sqr_diff(A, A_, n*n);
        double error_b = nrm_sqr_diff(B, B_, m*n);

        if (error_a > EPS || error_b > EPS) {
            cout << "ERROR_A: " << error_a << endl;
            cout << "ERROR_B: " << error_b << endl;
            cout << "ERROR!!!!  the results for the " << i+1 << "th testset are WRONG" << endl;
        } else {
            cout << "Test " << i << " OK" << endl;
        }

        destroy(A_);
        destroy(B_);
        destroy(pi_);

        destroy(A);
        destroy(alpha);
        destroy(B);
        destroy(beta);
        destroy(pi);
        free(o);

    }
}

int get_num_flops(int n, int m, int t) {
    return 6*n*n*t + 9*n*t + 2*t - 4*n*n - 2*n + m*n;
}

/*
* Returns the number of cycles required per iteration
*/
pair<double,double> perf_test(comp_func f, double *A, double *alpha, double *B, double *beta, double *pi, int *o, int n, int m, int t, double *A_, double *B_, double *pi_){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start_cycles, end_cycles;
    double time;

    // Warm-up phase: we determine a number of executions that allows
    // the code to be executed for at least CYCLES_REQUIRED cycles.
    // This helps excluding timing overhead when measuring small runtimes.
    do {
        num_runs = num_runs * multiplier;
        start_cycles = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            f(A, alpha, B, beta, pi, o, n, m, t, A_, B_, pi_);
        }
        end_cycles = stop_tsc(start_cycles);

        cycles = (double)end_cycles;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);

    list<double> cyclesList;
    list<double> timeList;

    // Actual performance measurements repeated REP times.
    // We simply store all results and compute medians during post-processing.
    for (size_t j = 0; j < REP; j++) {

        auto start_time = high_resolution_clock::now();
        start_cycles = start_tsc();
        for (size_t i = 0; i < num_runs; ++i) {
            f(A, alpha, B, beta, pi, o, n, m, t, A_, B_, pi_);
        }
        end_cycles = stop_tsc(start_cycles);
        auto end_time = high_resolution_clock::now();

        cycles = ((double)end_cycles) / num_runs;
        time = duration_cast<microseconds>(end_time-start_time).count();
        time = time / num_runs / 1000;

        cyclesList.push_back(cycles);
        timeList.push_back(time);
    }

    cyclesList.sort();
    timeList.sort();
    return make_pair(cyclesList.front(),timeList.front());
}

int main(int argc, char **argv){
    if(argc > 1)
            std::cout.setstate(std::ios_base::failbit);

    cout << "Starting program...\n";
    register_functions();

    if (numFuncs == 0){
        cout << endl;
        cout << "No functions registered - nothing for driver to do" << endl;
        cout << "Register functions by calling register_func(f, name)" << endl;
        cout << "in register_funcs()" << endl;

        return 0;
    }

    if(VALIDATE) validate_base();

    int n, m, t;
    if (argc > 1) {
        n = atoi(argv[1]);
        m = atoi(argv[2]);
        t = atoi(argv[3]);
    } else {
        n = 32;
        m = 32;
        t = 128;
    }

    if (n > INT_MAX / n || n > INT_MAX / m || n > INT_MAX / t){
        // we would have overflow while computing the array sizes to allocate
        cout << "input arguments are too big, choose smaller values" << endl;
        return 0;
    }

    if (n > INT_MAX / n || n > INT_MAX / m || n > INT_MAX / t){
        // we would have overflow while computing the array sizes to allocate
        cout << "input arguments are too big, choose smaller values" << endl;
        return 0;
    }

    // Generate input matrices
    double *A = static_cast<double *>(aligned_alloc(32, n * n * sizeof(double)));
    double *B = static_cast<double *>(aligned_alloc(32, m * n * sizeof(double)));
    double *pi = static_cast<double *>(aligned_alloc(32, n * sizeof(double)));

    gen_rand_mat(A, n, n);
    gen_rand_mat(B, m, n);
    gen_rand_mat(pi, 1, n);

    int *o = static_cast<int *>(aligned_alloc(32, t * sizeof(int)));
    random_device rd;
    mt19937 gen{rd()};
    uniform_int_distribution<int> dist(0, m-1);
    for (int i = 0; i < t; i++)
        o[i] = dist(gen);

    // Generate output matrices
    double *A_ = static_cast<double *>(aligned_alloc(32, n* n * sizeof(double)));
    double *B_ = static_cast<double *>(aligned_alloc(32, m* n * sizeof(double)));
    double *pi_ = static_cast<double *>(aligned_alloc(32, n * sizeof(double)));

    // Generate matrices for base algorithm
    double *A_base = static_cast<double *>(aligned_alloc(32, n* n * sizeof(double)));
    double *B_base = static_cast<double *>(aligned_alloc(32, m* n * sizeof(double)));
    double *pi_base = static_cast<double *>(aligned_alloc(32, n * sizeof(double)));

    // Helper matrices
    double *alpha = static_cast<double *>(aligned_alloc(32, t* n * sizeof(double)));
    double *beta = static_cast<double *>(aligned_alloc(32, t * n * sizeof(double)));

    // Generate base results
    comp_func f_base = userFuncs[0];
    f_base(A, alpha, B, beta, pi, o, n, m, t, A_base, B_base, pi_base);

    for (int i = 1; i < numFuncs; i++) {

        if (funcNames[i-1] == "reduced flops"){
            transpose_matrix(A, n, n);
            transpose_matrix(B, m, n);
            transpose_matrix(A_base, n, n);
            transpose_matrix(B_base, m, n);
        }

        comp_func f = userFuncs[i];
        f(A, alpha, B, beta, pi, o, n, m, t, A_, B_, pi_);
        double error_a = nrm_sqr_diff(A_, A_base, n*n);
        double error_b = nrm_sqr_diff(B_, B_base, m*n);
        double error_pi = nrm_sqr_diff(pi_, pi_base, n);

        if (error_a > EPS || error_b > EPS || error_pi > EPS) {
            cout << "ERROR_A: " << error_a << endl;
            cout << "ERROR_B: " << error_b << endl;
            cout << "ERROR_pi: " << error_pi << endl;
            cout << "ERROR!!!!  the results of the function " << funcNames[i] << " are WRONG" << endl;
            return 0;
        }
    }

    // Destroy useless matrices
    destroy(A_base);
    destroy(B_base);
    destroy(pi_base);

    for (int i = 0; i < numFuncs; i++){
        cout << endl << "Running: " << funcNames[i] << endl;
        pair<double,double> result = perf_test(userFuncs[i], A, alpha, B, beta, pi, o, n, m, t, A_, B_, pi_);
        double cycles = result.first;
        double time_ms = result.second;
        double perf = get_num_flops(n, m, t) / cycles;
        cout << setprecision(2) << perf << " flops / cycles" << endl;
        if(argc > 1) {
            printf("%s,%i,%i,%i,%.2f,%.2f \n", funcNames[i].c_str(), n, m, t, perf, time_ms);
        }
    }

    // Destroy all matrices
    destroy(A);
    destroy(B);
    destroy(pi);
    destroy(A_);
    destroy(B_);
    destroy(pi_);
    destroy(alpha);
    destroy(beta);
    free(o);

    return 0;
}
