import click
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from labellines import labelLine, labelLines
from labellines import *

plt.rcParams['svg.fonttype'] = 'none'


#https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.lines.Line2D.html
#https://matplotlib.org/gallery/color/named_colors.html


PEAK_PERFORMANCE = 4    # flops/cycle
out_folder = "./fig"

@click.command()
@click.option('--variable',
              help='which variable is on x-axis? "m","n","t","nmt"')
@click.option('--fixed_vars',
              help='string containing the fixed variables')
@click.option('--benchmark/--no_benchmark', default=False,
              help='create benchmark plot if True')

def main(
    variable,
    fixed_vars,
    benchmark,
    ):

    color = {
        "ParaHMM": "#976ed7",
        "Base": "#f39a27",
        "SIMD": "#03c03c",
        "Unrolled Loops": "#f39a27",
        "Column Major": "#976ed7",
        "Less Flops": "#c47a53",
        "SIMD+Blocking": "#eada52"
    }

    rename = {
        "ParaHMM": "ParaHMM",
        "Base": "Base",
        "SIMD": "Vectorization",
        "Unrolled Loops": "Loop Unrolling",
        "Column Major": "Column Major",
        "Less Flops": "Flop Reduction",
        "SIMD+Blocking": "Blocking"
    }

    # helper function
    def plot(x, y, label, colour, marker):
        plt.plot(x, y, label=rename[label], color=color[label], marker=marker,
                linewidth=4, markersize=10)
    

    input_file = f"data/results_n_m_t.csv"
    #input_file = f"data/results_n_64_64.csv"
    #input_file = f"data/results_64_m_64.csv"
    #input_file = f"data/results_64_64_t.csv"
    plot_type = ""

    timing = ""
    title = ""
    if benchmark:
        plot_type = "benchmark"
        timing = "time_ms"
        title = f"Benchmark plot for variable {variable.capitalize()} and {fixed_vars}"
    else:
        plot_type = "performance"
        timing = "performance"
        if variable != "nmt":
            title = f"Performance plot for variable {variable.capitalize()} and {fixed_vars}"
        else:
            title = f"Performance plot for variable N=M=T"

    # Graph setup
    plt.rcParams['axes.facecolor'] = '#A9A9A9'
    plt.xscale("log", basex=2)
    plt.title(title, loc="left", fontsize=28)
    if variable == "nmt":
        plt.xlabel("N,M,T", fontsize=22)
    else:
        plt.xlabel(variable.capitalize(), fontsize=22)
    plt.ylabel('flops/cycle', fontsize=22)
    plt.grid(axis='y', color='white')

    # Data Format:
    # flag, function name, n, m , t, performance, time_ms
    #   flag and function name to filter out
    #   n, m, t, performance, time_ms should not be used to filter out
    full_data = pd.read_csv(input_file, delimiter = ',')

    # find functions
    functions = list(set(full_data["function"]))
    functions = [f for f in functions if f != "benchmark"]

    functions = ["Unrolled Loops", "SIMD", "Column Major", "SIMD+Blocking", "Less Flops"]
    
    if benchmark: functions = ["Base","Less Flops","ParaHMM", "SIMD+Blocking"]

    print(f"Found timings for {len(functions)} different functions")

    colours = ["#03c03c","#f39a27","#976ed7","#c47a53","#579abe", "#eada52"]
    markers = ["o", "^", "D"]

    min_x = float("inf")
    max_x = 0

    for index, function in enumerate(functions):

        #if function == "Base":
        #    continue

        marker = markers[index%3]
        data = full_data[full_data["function"] == function]

        flags = list(set(data["flag"]))
        for idx_flag, flag in enumerate(flags):
            table = data[data["flag"]==flag]
            which_index = index
            if (len(flags) > 1):
                which_index = idx_flag
            colour = colours[which_index]

            if variable=="nmt":
                x = table["n"]
            else:
                x = table[variable]
            y = table[timing]

            min_x = min(min_x,list(x)[0])
            max_x = max(max_x,list(x)[-1])

            label = function #+ " -" + flag
            plot(x,y,label, colour, marker)


    if benchmark:
        plt.ylabel('time [ms]', fontsize=22)
        plt.yscale("log", basey=10)

        data = full_data[full_data["function"] == "benchmark"]
        benchmark = data[data["flag"]=="benchmark"]

        if variable=="nmt":
            x = table["n"]
        else:
            x = table[variable]
        y = benchmark[timing]

        # annotate parahmm benchmark comparison
        plot(x,y, "ParaHMM", "#c23b23", ",")
        # maybe inline legend for parahmm timing
        # plt.text(list(x)[-1]/2, list(y)[-2]/2, 'parahmm', color="red")
    else:
        # annotate peak performance
        plt.axhline(y=PEAK_PERFORMANCE, color="#c23b23")
        #plt.axvline(x=L1/L2/L3, color="red")

        # variable n: /6
        # variable m,t: /28
        # variable nmt: /3.8
        plt.text(max_x/3.8, PEAK_PERFORMANCE+0.1, 'Peak Performance w/o vec', color="#c23b23", size=20)
    
    # plt.legend(loc='upper left',bbox_to_anchor=(1.,0.6), facecolor='white')
    fig = plt.gcf()
    fig.set_size_inches(14.0, 7)
    ticks = 2 ** np.arange(21)
    if variable == "n":
        plt.xticks(ticks[3:15],fontsize=22)
    elif variable == "nmt":
        plt.xticks(ticks[3:11],fontsize=22)
    else:
        plt.xticks(ticks[3:],fontsize=22)

    plt.yticks(fontsize=22)

    out_file = f"{out_folder}/{plot_type}_{variable}.pdf"

    # for variable nmt: ,xvals=[20,48,32,180,32]
    # for variable n: ,xvals=[24,96,256,2050,2048]
    # for variable m: ,xvals=[32,32,32,24,32]
    # for variable t: ,xvals=[48,512,4096,32000,2048]
    # for benchmark t: ,xvals=[196,1024,196,32]
    # for benchmark n: ,xvals=[60,32,256,128]
    # for benchmark m: ,xvals=[4096,1024,196,32]
    labelLines(plt.gca().get_lines(),fontsize=20,color='k',xvals=[20,96,32,180,32])

    plt.tight_layout()
    plt.savefig(out_file, format='pdf')
    #plt.show()
    print(f"Saved figure: {out_file}")


if __name__ == '__main__':
    main()
