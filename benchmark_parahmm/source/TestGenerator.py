import sys
import random
import numpy as np, numpy.random
def prefix_sum(srcList):
	arr = []
	prefix_sum = 0
	for j in range(len(srcList)):
		prefix_sum += srcList[j]
		arr.append(prefix_sum)
	arr[len(srcList)-1] = 1.0
	return arr

def random_index(srcList):
	seed = random.random()
	index = 0
	while seed >= srcList[index]:
		index+=1	
	return index

if __name__ == "__main__":
	if len(sys.argv) < 5:
		print("Not enough arguments!")
		sys.exit(-1)
	
	hstate = int(sys.argv[1])
	ostate = int(sys.argv[2])
	seq_number = int(sys.argv[3]) 
	seq_length = int(sys.argv[4])

	fd = open("test.txt", "w")
	fd.write("# a HMM model configuration for testing purpose\n\n")
	fd.write("# number of states\n%d\n\n" % hstate)
	fd.write("# number of output\n%d\n\n" % ostate)

	fd.write("# initial state probability\n")
	prior = np.random.dirichlet(np.ones(hstate),size=1)[0].tolist()
	fd.write(' '.join(str(a) for a in prior)+"\n")
	
	prefix_prior = prefix_sum(prior)
	
	fd.write("\n# state transition probability\n")
	transit_sum = []
	for i in range(hstate):
		transit = np.random.dirichlet(np.ones(hstate),size=1)[0].tolist()
		fd.write(' '.join(str(a) for a in transit)+"\n")
		transit_sum.append(prefix_sum(transit))

	fd.write("\n# state output probability\n")
	emit_sum = []
	for i in range(hstate):
		emit = np.random.dirichlet(np.ones(ostate),size=1)[0].tolist()
		fd.write(' '.join(str(a) for a in emit)+"\n")
		emit_sum.append(prefix_sum(emit))
	fd.write("\n# data size\n")
	fd.write("%d %d\n" % (seq_number, seq_length))
	fd.write("\n# data\n")
	for j in range(seq_number):
		parent = random_index(prefix_prior)
		for i in range(seq_length):
			transit_state = random_index(transit_sum[parent])
			emit_state = random_index(emit_sum[transit_state])
			fd.write("%d " % emit_state)
			parent = transit_state
		fd.write("\n")
	fd.flush()
	fd.close()





