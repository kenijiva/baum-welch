import os
import sys

import pandas as pd
import numpy as np
import math

#
# Computes the maximum speedup factor between two implementations
# input argument: path of the csv file with the timings
#
# Change compare_1 and compare_2 (line 35/36) to the functions you want to compare
#
def main():

    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python3 get_max_speedup.py <result_file>")
        return

    base       = "Base"
    less_flops = "Less Flops"
    col_major  = "Column Major"
    unrolled   = "Unrolled Loops"
    simd       = "SIMD"
    blocking   = "SIMD+Blocking"

    
    # set correct input file
    input_file = args[0]
    flag = "march"
    
    # compare_1: worse performance
    # compare_2: better performance
    compare_1 = col_maj
    compare_2 = unrol

    
    # read in data
    full_data = pd.read_csv(input_file, delimiter = ',')

    # find functions
    data_1 = full_data[full_data["function"] == compare_1]
    table_1 = data_1[data_1["flag"]==flag]["performance"]
    table_1 = list(table_1)

    data_2 = full_data[full_data["function"] == compare_2]
    table_2 = data_2[data_2["flag"]==flag]["performance"]
    table_2 = list(table_2)

    speedup = []
    for index in range(len(table_1)):
        speedup.append(float(table_2[index]) / table_1[index])

    index_max = speedup.index(max(speedup))

    print(f"speedup for function \"{compare_2}\" against \"{compare_1}\":")
    print(speedup)
    print("-----------")
    print(f"highest factor: {max(speedup)} at index {index_max}.")
    print(f"index {index_max} is 2^{index_max+3} on x axis.")

if __name__ == '__main__':
    main()
