#bprop.py
#Author: Nicholas Smith
import numpy as np

#Array of layer sizes
ls = np.array([2, 4, 4, 1])
n = len(ls)

#List of weight matrices (each a numpy array)
W = []
#Initialize weights to small random values
for i in range(n - 1):
	W.append(np.random.randn(ls[i], ls[i + 1]) * 0.1)

#List of bias vectors initialized to small random values
B = []
for i in range(1, n):
	B.append(np.random.randn(ls[i]) * 0.1)
	
#List of output vectors
O = []
for i in range(n):
	O.append(np.zeros([ls[i]]))
	
#List of Delta vectors
D = []
for i in range(1, n):
	D.append(np.zeros(ls[i]))

#Input vectors (1 row per each)
A = np.matrix([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
#Target Vectors (1 row per each)
y = np.matrix([[-0.5], [0.5], [0.5], [-0.5]])

#Activation function (tanh) for each layer
#Linear activation for final layer
actF = []
dF = []
for i in range(n - 1):
	actF.append(lambda (x) : np.tanh(x))
	#Derivative of activation function in terms of itself
	dF.append(lambda (y) : 1 - np.square(y))
	
#Linear activation for final layer
actF.append(lambda (x): x)
dF.append(lambda (x) : np.ones(x.shape))

#Learning rate
a = 0.5
#Number of iterations
numIter = 250

#Loop for each iteration
for c in range(numIter):
	#loop over all input vectors
	for i in range(len(A)):
		print(str(i))
		#Target vector
		t = y[i, :]
		#Feed-forward step
		O[0] = A[i, :]
		for j in range(n - 1):
			O[j + 1] = actF[j](np.dot(O[j], W[j]) + B[j])
		print('Out:' + str(O[-1]))
		#Compute output node delta values
		D[-1] = np.multiply((t - O[-1]), dF[-1](O[-1]))
		#Compute hidden node deltas
		for j in range(n - 2, 0, -1):
			D[j - 1] = np.multiply(np.dot(D[j], W[j].T), dF[j](O[j]))
		#Perform weight and bias updates
		for j in range(n - 1):
			W[j] = W[j] + a * np.dot(O[j].T, D[j])
			B[j] = B[j] + a * D[j]		

print('\nFinal weights:')
#Display final weights
for i in range(n - 1):
	print('Layer ' + str(i + 1) + ':\n' + str(W[i]) + '\n')
