from args import get_args

from random import random
import time

def randomMatrix(m, n):
    matrix = []
    for i in range(m):
        sum = 0
        row = []
        for j in range(n):
            rand = random()
            row.append(rand)
            sum += rand
            
        for j in range(n):
            row[j] /= sum

        matrix.append(row)
        
    return matrix


def get_probability_intervals_matrix(m, n, matrix):
    interval_matrix = []
    for i in range(m):
        row = matrix[i]
        interval_row = []
        for j in range(n):
            interval_row.append(row[j]) if j == 0 else interval_row.append(interval_row[j-1] + row[j])
        interval_matrix.append(interval_row)
        
    return interval_matrix
    

def generate_obeservation_sequence(A, B, pi, M, N, T):
    A_intervals = get_probability_intervals_matrix(N, N, A)
    B_intervals = get_probability_intervals_matrix(N, M, B)
    pi_intervals = get_probability_intervals_matrix(1, N, pi)

    observed_sequence = []
    states = []

    # First state
    rand = random()
    for i in range(N):
        if rand <= pi_intervals[0][i]:
            states.append(i)
            break


    # Subsequent states
    for t in range(1, T):
        last_state = states[t-1]
        rand = random()
        for j in range(N):
            if rand <= A_intervals[last_state][j]:
                states.append(j)
                break


    # Generate observed sequence
    for t in range(T):
        state = states[t]
        rand = random()
        for j in range(M):
            if rand <= B_intervals[state][j]:
                observed_sequence.append(j)
                break

    return observed_sequence

            
def main():
    args = get_args()
    N = args.num_states
    M = args.num_observation_symbols
    T = args.num_observed_symbols

    A = randomMatrix(N, N)
    B = randomMatrix(N, M)
    pi = randomMatrix(1, N)

    timestr = time.strftime("%Y:%m:%d-%H:%M:%S")

    observations = generate_obeservation_sequence(A,B,pi,M,N,T)
    with open("test_data_input/" + timestr + ".txt", "w") as file:
    	file.write(str(N) + "\n" + str(M) + "\n" + str(T) + "\n")
    	for i in range(N):
    		for j in range(N):
    			file.write(str(A[i][j]))
    			file.write(" ") if not (i == N-1 and j == N-1) else file.write("\n")
    	for i in range(N):
    		for j in range(M):
    			file.write(str(B[i][j]))
    			file.write(" ") if not (i == N-1 and j == M-1) else file.write("\n")
    	for i in range(N):
    		file.write(str(pi[0][i]))
    		file.write(" ") if not i == N-1 else file.write("\n")
    	for i in range(T):
    		file.write(str(observations[i]))
    		if not i == T-1:
    			file.write(" ")

if __name__ == "__main__":
    main()
