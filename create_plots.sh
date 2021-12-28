cp fig/base_best/results_n_32_32.csv results_n.csv
cp fig/base_best/results_32_32_t.csv results_t.csv
cp fig/base_best/results_32_m_32.csv results_m.csv
python3 plot.py --variable n --fixed_vars "m,t=32"
python3 plot.py --variable t --fixed_vars "m,n=32"
python3 plot.py --variable m --fixed_vars "n,t=32"
mv fig/performance_n.svg fig/base_best/performance_n_32_32.svg
mv fig/performance_t.svg fig/base_best/performance_32_32_t.svg
mv fig/performance_m.svg fig/base_best/performance_32_m_32.svg
python3 plot.py --variable n --fixed_vars "m,t=32" --benchmark
python3 plot.py --variable t --fixed_vars "m,n=32" --benchmark
python3 plot.py --variable m --fixed_vars "n,t=32" --benchmark
mv fig/benchmark_n.svg fig/base_best/benchmark_n_32_32.svg
mv fig/benchmark_t.svg fig/base_best/benchmark_32_32_t.svg
mv fig/benchmark_m.svg fig/base_best/benchmark_32_m_32.svg

cp fig/base_best/results_n_64_64.csv results_n.csv
cp fig/base_best/results_64_64_t.csv results_t.csv
cp fig/base_best/results_64_m_64.csv results_m.csv
python3 plot.py --variable n --fixed_vars "m,t=64"
python3 plot.py --variable t --fixed_vars "m,n=64"
python3 plot.py --variable m --fixed_vars "n,t=64"
mv fig/performance_n.svg fig/base_best/performance_n_32_32.svg
mv fig/performance_t.svg fig/base_best/performance_32_32_t.svg
mv fig/performance_m.svg fig/base_best/performance_32_m_32.svg
python3 plot.py --variable n --fixed_vars "m,t=64" --benchmark
python3 plot.py --variable t --fixed_vars "m,n=64" --benchmark
python3 plot.py --variable m --fixed_vars "n,t=64" --benchmark
mv fig/benchmark_n.svg fig/base_best/benchmark_n_64_64.svg
mv fig/benchmark_t.svg fig/base_best/benchmark_64_64_t.svg
mv fig/benchmark_m.svg fig/base_best/benchmark_64_m_64.svg

cp fig/base_best/results_n_m_t.csv results_nmt.csv
python3 plot.py --variable "nmt" 
mv fig/performance_nmt.svg fig/base_best/
cp fig/base_best/results_n_m_t.csv results_nmt.csv
python3 plot.py --variable "nmt" --benchmark
mv fig/benchmark_nmt.svg fig/base_best/

#Presentation plots
cp fig/base_red_bench/results_n.csv .
cp fig/base_red_bench/results_t.csv .
python3 plot.py --variable n --fixed_vars "m,t=64"
python3 plot.py --variable t --fixed_vars "m,n=64"
mv fig/performance_n.svg fig/base_red_bench/
mv fig/performance_t.svg fig/base_red_bench/

cp fig/base_red_bench/results_n.csv .
cp fig/base_red_bench/results_t.csv .
python3 plot.py --variable n --fixed_vars "m,t=64" --benchmark
python3 plot.py --variable t --fixed_vars "m,n=64" --benchmark
mv fig/benchmark_n.svg fig/base_red_bench/
mv fig/benchmark_t.svg fig/base_red_bench/

cp fig/colmaj_unrolled_loops/results_n.csv .
cp fig/colmaj_unrolled_loops/results_t.csv .
python3 plot.py --variable n --fixed_vars "m,t=64"
python3 plot.py --variable t --fixed_vars "m,n=64"
mv fig/performance_n.svg fig/colmaj_unrolled_loops/
mv fig/performance_t.svg fig/colmaj_unrolled_loops/

cp fig/colmaj_unrolled_loops/results_n.csv .
cp fig/colmaj_unrolled_loops/results_t.csv .
python3 plot.py --variable n --fixed_vars "m,t=64" --benchmark
python3 plot.py --variable t --fixed_vars "m,n=64" --benchmark
mv fig/benchmark_n.svg fig/colmaj_unrolled_loops/
mv fig/benchmark_t.svg fig/colmaj_unrolled_loops/

cp fig/loopun_simd_simdblocking/results_n.csv .
cp fig/loopun_simd_simdblocking/results_t.csv .
python3 plot.py --variable n --fixed_vars "m,t=64"
python3 plot.py --variable t --fixed_vars "m,n=64"
mv fig/performance_n.svg fig/loopun_simd_simdblocking/
mv fig/performance_t.svg fig/loopun_simd_simdblocking/

cp fig/loopun_simd_simdblocking/results_n.csv .
cp fig/loopun_simd_simdblocking/results_t.csv .
python3 plot.py --variable n --fixed_vars "m,t=64" --benchmark
python3 plot.py --variable t --fixed_vars "m,n=64" --benchmark
mv fig/benchmark_n.svg fig/loopun_simd_simdblocking/
mv fig/benchmark_t.svg fig/loopun_simd_simdblocking/

cp fig/red_columnmaj_bench/results_n.csv .
cp fig/red_columnmaj_bench/results_t.csv .
python3 plot.py --variable n --fixed_vars "m,t=64"
python3 plot.py --variable t --fixed_vars "m,n=64"
mv fig/performance_n.svg fig/red_columnmaj_bench/
mv fig/performance_t.svg fig/red_columnmaj_bench/

cp fig/red_columnmaj_bench/results_n.csv .
cp fig/red_columnmaj_bench/results_t.csv .
python3 plot.py --variable n --fixed_vars "m,t=64" --benchmark
python3 plot.py --variable t --fixed_vars "m,n=64" --benchmark
mv fig/benchmark_n.svg fig/red_columnmaj_bench/
mv fig/benchmark_t.svg fig/red_columnmaj_bench/

#ALL PLOTS/appendix
cp data/results_32_32_t.csv results_t.csv
cp data/results_n_32_32.csv results_n.csv
cp data/results_32_m_32.csv results_m.csv

python3 plot.py --variable n --fixed_vars "m,t=32"
python3 plot.py --variable m --fixed_vars "n,t=32"
python3 plot.py --variable t --fixed_vars "m,n=32"
mv fig/performance_n.svg fig/performance_n_32_32.svg
mv fig/performance_m.svg fig/performance_32_m_32.svg
mv fig/performance_t.svg fig/performance_32_32_t.svg

python3 plot.py --variable n --fixed_vars "m,t=32" --benchmark
python3 plot.py --variable m --fixed_vars "n,t=32" --benchmark
python3 plot.py --variable t --fixed_vars "m,n=32" --benchmark
mv fig/benchmark_n.svg fig/benchmark_n_32_32.svg
mv fig/benchmark_m.svg fig/benchmark_32_m_32.svg
mv fig/benchmark_t.svg fig/benchmark_32_32_t.svg

cp data/results_64_64_t.csv results_t.csv
cp data/results_n_64_64.csv results_n.csv
cp data/results_64_m_64.csv results_m.csv

python3 plot.py --variable n --fixed_vars "m,t=64"
python3 plot.py --variable m --fixed_vars "n,t=64"
python3 plot.py --variable t --fixed_vars "m,n=64"
mv fig/performance_n.svg fig/performance_n_64_64.svg
mv fig/performance_m.svg fig/performance_64_m_64.svg
mv fig/performance_t.svg fig/performance_64_64_t.svg

python3 plot.py --variable n --fixed_vars "m,t=64" --benchmark
python3 plot.py --variable m --fixed_vars "n,t=64" --benchmark
python3 plot.py --variable t --fixed_vars "m,n=64" --benchmark
mv fig/benchmark_n.svg fig/benchmark_n_64_64.svg
mv fig/benchmark_m.svg fig/benchmark_64_m_64.svg
mv fig/benchmark_t.svg fig/benchmark_64_64_t.svg

cp data/results_n_m_t.csv results_nmt.csv
python3 plot.py --variable "nmt"
mv fig/performance_nmt.svg fig/performance_n_m_t.svg

cp data/results_n_m_t.csv results_nmt.csv
python3 plot.py --variable "nmt" --benchmark
mv fig/benchmark_nmt.svg fig/benchmark_n_m_t.svg

mv fig/bench* fig/perf* fig/all
rm results_t.csv results_n.csv results_m.csv results_nmt.csv results_m.csv
