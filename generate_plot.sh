#!/bin/bash

benchmark=false

out_path="./out"
benchmark_path="./benchmark_parahmm/source"

# Output file format:       FLAG;FUNCTION;VERSION;n;m;t;cycles
#
#   FLAG = novec / vec / fastmath / march
#   FUNCTION = Name at addfunction();
#   n, m, t
#   cylces = ./VERSION m

#function plotting for some variable: first arg
plot_var() {
	# Prepare outputfile
	out_file="results_$1.csv"

	mkdir -p old_results
	[ -f $out_file ] && cp $out_file old_results/results_$1_$(date +"%s").csv
	[ -f $out_file ] && rm $out_file
	touch $out_file
	# fix variables
	n=$N
	m=$M
	t=$T

	if [ "$1" = "n" ]; then
		FIXED1="m"
		FIXED2="t"
	elif [ "$1" = "m" ]; then
		FIXED1="n"
		FIXED2="t"
	elif [ "$1" = "t" ]; then
		FIXED1="n"
		FIXED2="m"
	else
		echo "Variable $1 is unknown as an argument"
	fi

	echo -e "\nMeasure performance for\nvariable $1\nfixed $FIXED1=${!FIXED1}\nfixed $FIXED2=${!FIXED2}\n"
	
	echo flag,function,n,m,t,performance,time_ms >> $out_file
	for (( $1 = $((2**3)); $1 <=  $((2**10)); $1 *= 2 )) do
		
		echo -ne "\rProgress: $1=${!1} $FIXED1=${!FIXED1} $FIXED2=${!FIXED2}\n"
		(${out_path}/novec.out $n $m $t | sed 's/^/novec,/') >> $out_file
		(${out_path}/vec.out $n $m $t | sed 's/^/vec,/') >> $out_file
		(${out_path}/fastmath.out $n $m $t | sed 's/^/fastmath,/') >> $out_file
		(${out_path}/march.out $n $m $t | sed 's/^/march,/') >> $out_file
		
		if [ "$benchmark" = true ]; then
			
			# generate random data
			python3 ${benchmark_path}/TestGenerator.py $n $m 1 $t
		
			# if there is a segmentation fault for some input values parsing of this output will fail
			output=$(${benchmark_path}/hmm -c ./test.txt -t1 -p3 -n1)
			
			# available for debugging
			# cycles=$(echo "${output}" | grep -o -P '(?<=Measured cycles ).*(?= cycles)')
			time_ms=$(echo "${output}" | grep -o -P '(?<=Measured time ).*(?= ms)')

			# clean up
			rm ./test.txt
			
			echo benchmark,benchmark,$n,$m,$t,0,$time_ms >> $out_file
		fi
		
	done

	echo ""

	FIXED_VARS="fixed $FIXED1=${!FIXED1}, $FIXED2=${!FIXED2}"
	if [ "$benchmark" = true ]; then
		python3 plot.py --variable "$1" --fixed_vars "$FIXED_VARS" --benchmark
	else
		python3 plot.py --variable "$1" --fixed_vars "$FIXED_VARS"
	fi

}

# Function to display commands
exec() { echo "\$ $@" ; "$@" ; }





#DEFINITION OF COMPILER FLAGS: COMMENT OUT FLAG DECLARATION OR ADD NEW FLAG BUT ADD VARIABLE NAME TO FLAGS ARRAY
novec=(" -o ${out_path}/novec.out -O3 -mavx2 -fno-tree-vectorize")
vec=(" -o ${out_path}/vec.out -O3 -mavx2")
fastmath=(" -o ${out_path}/fastmath.out -O3 -ffast-math -mavx2")
march=(" -o ${out_path}/march.out -O3 -ffast-math -march=native")
FLAGS=(novec vec fastmath march)
for flag in ${FLAGS[@]}
do
	if [ "${!flag}" != "" ]; then
		exec g++ main.cpp tsc_x86.h common.h ${!flag}
	fi
done

# compile benchmark
cd $benchmark_path
exec g++ -std=c++11 -Wno-narrowing -fopenmp -mavx2 -mavx -g -lm -O3 hmm.cpp -o hmm
cd ../..

#THIS ARE THE VALUES IF A VARIABLE IS FIXED: CHANGE IF NEEDED
N=32
M=32
T=32

#GENERATE PLOT: COMMENT OUT IF NEEDED
plot_var n
plot_var m
plot_var t
