import matplotlib.pyplot as plt
import math
import numpy as np

eps=0.02
eps2=0.05

x = np.array([0, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64, 128])
y = np.array([0, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64, 128]) * 4.375;

ceiling1_x = [0, 64]
ceiling1_y = [4, 4]

ceiling2_x = [0, 64]
ceiling2_y = [16, 16]

line = plt.loglog(x,y, ceiling1_x, ceiling1_y, 'k', ceiling2_x, ceiling2_y, 'k')

plt.axis([0, 64, 0, 64*4.375])



plt.axvline(x=4/4.375, color='k', linestyle='--', ymax=5/11-0.005)
plt.axvline(x=16/4.375, color='k', linestyle='--', ymax=7/11-0.005)

plt.annotate('Peak \u03C0 without SIMD (4.0 flops/cycle)', xy=(1/24, 4), xytext=(1/28, 4.5), color='black')
plt.annotate('Peak \u03C0 with SIMD (16.0 flops/cycle)', xy=(1/24, 16), xytext=(1/28, 17.5), color='black')
plt.annotate('Read/Write bandwidth \u03B2 = 4.375 bytes/cycle', xy=(1, 7.5), xytext=(1.25,7.25), color='blue', rotation=37, fontsize=9)

#plt.scatter(9.62, 6.9)
#plt.scatter(4.83, 6.2)
#plt.scatter(2.43, 3.8)
#plt.scatter(1.23, 1.4)

opt_x = np.array([5.70, 11.52, 23.19, 39.87, 43.53, 45.53, 46.57, 47.10, 47.37])
opt_y = np.array([2.07, 4.04, 5.89, 3.90, 4.78, 5.55, 6.36, 6.60, 6.75])
plt.loglog(opt_x, opt_y, marker='o', color='red')
plt.annotate("AVX2 + Blocking", (8.0, 8.0), color='red')

#plt.scatter(9.62, 6.1)
#plt.annotate("SIMD w/o blocking", (9.62, 6.1))

#plt.scatter(9.62, 3.3)
#plt.annotate("Locality + aliasing", (9.62, 3.3))

#plt.scatter(9.62, 0.32)
#plt.scatter(4.83, 0.32)
#plt.scatter(2.43, 0.3)
#plt.scatter(1.23, 0.27)

unrolled_x = np.array([5.70, 11.52, 23.19, 39.87, 43.53, 45.53, 46.57, 47.10, 47.37])
unrolled_y = np.array([1.92, 2.88, 3.75, 3.13, 3.54, 3.82, 4.05, 4.11, 4.09])
plt.loglog(unrolled_x, unrolled_y, marker='o', color='green')
plt.annotate("Loop Unrolling", (10.5, 1.25), color='green')
plt.annotate("(w/o AVX2, w/o Blocking)", (6.5, 0.8), color='green')

plt.xticks(2**np.arange(-5.0, 6.0, 1.0), ["1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4", "8", "16", "32", "64"])
plt.yticks(2**np.arange(-3.0, 8.0, 1.0), 2**np.arange(-3.0, 8.0, 1.0))

plt.grid(True)
plt.xlabel('Operational Intensity [Flops/Byte]')
plt.ylabel('Performance [Flops/Cycle]')

#plt.show()
#plt.title("Roofline model for n,m,t = 64")
plt.savefig('roofline.svg')