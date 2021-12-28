library('HMM')

options(digits = 22);

generate_vec <- function(length) {
	vec = runif(length, min=0, max=1)
	val = sum(vec)
	res = vec / val
	return(res)
}

generate_matrix <- function(N, M) {
	A = c()
	for (j in 1:N) {
		A = c(A, generate_vec(M))
	}
	A = matrix(A, nrow=N, ncol=M, byrow=TRUE)
	return(A)
}

generate_list <- function(size) {
	res = c()
	for (i in 0:(size-1)) {
		res = c(res, paste(i))
	}
	return(res)
}

print_matrix <- function(A, N, M) {
	to_print = ""
	for(j in 1:N) {
		for(k in 1:M) {
			cat(A[j,k], "\n")
		}
	}
	return(to_print)
}

print_vector <- function(A, N) {
	to_print = ""
	for(j in 1:N) {
		cat(A[j], "\n")
	}
	return(to_print)
}

TESTS = 1000

sink('testcases.txt', append=TRUE)
cat(TESTS, "\n")
for (i in 1:TESTS) {
	N = floor(runif(1, min=4, max=256))
	M = floor(runif(1, min=4, max=256))
	T_ = floor(runif(1, min=4, max=256))
	cat(N, "\n")
	cat(M, "\n")
	cat(T_, "\n")

	# Initial HMM
	hmm_init = initHMM(generate_list(N), generate_list(M),
		startProbs = generate_vec(N),
		transProbs = generate_matrix(N,N),
		emissionProbs = generate_matrix(N,M))
	print_matrix(hmm_init$transProbs, N, N)
	print_matrix(hmm_init$emissionProbs, N, M)
	print_vector(hmm_init$startProbs, N)

	observations = paste(floor(runif(T_, min=0, max=(M-0.1))),sep = "\n")
	print_vector(observations, T_)

	hmm_trained = baumWelch(hmm_init,observations,1)$hmm
	print_matrix(hmm_trained$transProbs, N, N)
	print_matrix(hmm_trained$emissionProbs, N, M)
	print_vector(hmm_trained$startProbs, N)
}

#STARTING EXAMPLE from PDF
#hmm = initHMM(c("0", "1"), c("0", "1", "2"),
#	startProbs = c(1.0, 0.0),
#	transProbs=matrix(c(c(.7,.3),  c(.5, .5)), nrow=N, ncol=N, byrow=TRUE),
#	emissionProbs=matrix(c(c(.6,.1,.3),  c(.1,.7,.2)), nrow=N, ncol=M, byrow=TRUE))
#
#print(hmm)
#
## Sequence of observation
#a = c("2", "1", "0")
##a = sample(c(rep("0",100),rep("1",300) ))
#
##observation = c(a) #use this for multiple sequences
#observation = a
#
## Baum-Welch
#bw = baumWelch(hmm,observation,1)
#
#print(bw$hmm)
