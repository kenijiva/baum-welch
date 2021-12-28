#include "tsc_x86.h"
#include <list>
#include <chrono>
#include <limits>

#include <iostream>
#include <iomanip>
#include "baum_welch.cpp"

/*
 * BAUM WELCH HOOK
 */

using namespace std;
using namespace std::chrono;

#define CYCLES_REQUIRED 1e8
#define REP 50


void baum_welch_hook(int *data, int nseq, int iterations, int length, int nstates, int nobvs, float *prior, float *trans, float *transT, float *obvs)
{
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start_cycles, end_cycles;
    double time;
    double time_original = numeric_limits<double>::max();

    // Warm-up phase: we determine a number of executions that allows
    // the code to be executed for at least CYCLES_REQUIRED cycles.
    // This helps excluding timing overhead when measuring small runtimes.
    do {
        num_runs = num_runs * multiplier;
        start_cycles = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            // call original function
            baum_welch(data, nseq, iterations, length, nstates, nobvs, prior, trans, transT, obvs);
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
            // call original function
            baum_welch(data, nseq, iterations, length, nstates, nobvs, prior, trans, transT, obvs);
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

    printf("Measured cycles %f cycles\n", cyclesList.front());
    printf("Measured time %f ms\n", timeList.front());
}
