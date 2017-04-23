#TheanoANN.py
import numpy as np
import theano
import theano.tensor as T

#Create an MLP: A sequence of fully-connected layers with an activation
#function AF applied at all layers except the last.
#X:		The input tensor
#W:		A list of weight tensors for layers of the MLP
#B:		A list of bias tensors for the layers of the MLP
#AF: 	The activation function to be used at hidden layers
#Ret:	The network output 
def CreateMLP(X, W, B, AF):
	n = len(W)
	for i in range(n - 1):
		X = AF(X.dot(W[i]) + B[i])
	return X.dot(W[n - 1]) + B[n - 1]

#Creates weight and bias matrices for an MLP network
#given a list of the layer sizes.
#L:		A list of the layer sizes
#Ret:	The lists of weight and bias matrices (W, B)
def CreateMLPWeights(L):
	W, B = [], []
	n = len(L)
	for i in range(n - 1):
		#Use Xavier initialization for weights
		xv = np.sqrt(6. / (L[i] + L[i + 1]))
		W.append(theano.shared(np.random.uniform(-xv, xv, [L[i], L[i + 1]])))
		#Initialize bias to 0
		B.append(theano.shared(np.zeros([L[i + 1]])))
	return (W, B)  

#Given a string of the activation function name, return the
#corresponding Theano function.
#Ret:	The Theano activation function handle
def GetActivationFunction(name):
	if name == 'tanh':
		return T.tanh
	elif name  == 'sig':
		return T.nnet.sigmoid
	elif name == 'fsig':
		return T.nnet.ultra_fast_sigmoid
	elif name == 'relu':
		return T.nnet.relu
	elif name == 'softmax':
		return T.nnet.softmax

class TheanoMLPR:

	def __init__(self, layers, actfn = 'tanh', batchSize = None, learnRate = 1e-3, maxIter = 1000, tol = 5e-2, verbose = True):
		self.AF = GetActivationFunction(actfn)
		#Batch size
		self.bs = batchSize
		self.L = layers
		self.lr = learnRate
		#Error tolerance for early stopping criteron
		self.tol = tol
		#Toggles verbose output
		self.verbose = verbose
		#Maximum number of iterations to run
		self.nIter = maxIter
		#List of weight matrices
		self.W = []
		#List of bias matrices
		self.B = []
		#Input matrix
		self.X = T.matrix()
		#Output matrix
		self.Y = T.matrix()
		#Weight and bias matrices
		self.W, self.B = CreateMLPWeights(layers)
		#The result of a forward pass of the network
		self.YH = CreateMLP(self.X, self.W, self.B, self.AF)
		#Use L2 loss for network
		self.loss = ((self.YH - self.Y) ** 2).mean()
		#Function for performing a forward pass
		self.ffp = theano.function([self.X], self.YH)
		#For computing the loss
		self.fcl = theano.function([self.X, self.Y], self.loss)
		#Gradients for weight matrices
		self.DW = [T.grad(self.loss, Wi) for Wi in self.W]
		#Gradients for bias
		self.DB = [T.grad(self.loss, Bi) for Bi in self.B]
		#Weight update terms
		WU = [(self.W[i], self.W[i] - self.lr * self.DW[i]) for i in range(len(self.DW))]
		BU = [(self.B[i], self.B[i] - self.lr * self.DB[i]) for i in range(len(self.DB))]
		#Gradient step
		self.fgs = theano.function([self.X, self.Y], updates = tuple(WU + BU))

	#Initializes the weight and bias matrices of the network
	def Initialize(self):
		n = len(self.L)
		for i in range(n - 1):
			#Use Xavier initialization for weights
			xv = np.sqrt(6. / (self.L[i] + self.L[i + 1]))
			self.W[i].set_value(np.random.uniform(-xv, xv, [self.L[i], self.L[i + 1]]))
			#Initialize bias to 0
			self.B[i].set_value(np.zeros([self.L[i + 1]]))		
			
	#Fit the MLP to the data
	#A: 	numpy matrix where each row is a sample
	#Y: 	numpy matrix of target values
	def fit(self, A, Y):
		self.Initialize()
		m = len(A)
		for i in range(self.nIter):
			if self.bs is None: #Use all samples
				self.fgs(A, Y)			#Perform the gradient step
			else: 	#Train m samples using random batches of size self.bs
				for _ in range(0, m, self.bs):
					#Choose a random batch of samples
					bi = np.random.randint(m, size = self.bs)
					self.fgs(A[bi], Y[bi])	#Perform the gradient step on the batch
			if i % 10 == 9:
				loss = self.score(A, Y)
				if self.verbose:
					print('Iter {:7d}: {:8f}'.format(1 + i, loss))
				if loss < self.tol:
					break

	#Predict the output given the input (only run after calling fit)
	#A: 	The input values for which to predict outputs
	#Ret: 	The predicted output values (one row per input sample)
	def predict(self, A):
		return self.ffp(A)

	#Predicts the ouputs for input A and then computes the loss term
	#between the predicted and actual outputs
	#A: 	The input values for which to predict outputs
	#Y: 	The actual target values
	#Ret: 	The network loss term
	def score(self, A, Y):
		return np.float64(self.fcl(A, Y))