#TFANN.py
#This file contains classes implementing ANN using numpy 
#and tensorflow. Presently, multi-layer perceptron (MLP) 
#and convolutional neural networks (CNN) are supported.
import tensorflow as tf
import numpy as np

#Return the classification accuracy
#given a vector of target labels and
#predicted labels
#y: The target labels
#yHat: The predicted labels
#return: The percentage correct
def _Accuracy(Y, YH):
	n = float(len(Y))
	return np.sum(Y == YH) / n

#Create the MLP variables for TF graph
#_X: The input matrix
#_W: The weight matrices
#_B: The bias vectors
#_AF: The activation function
def _CreateMLP(_X, _W, _B, _AF):
	n = len(_W)
	for i in range(n - 1):
		_X = _AF(tf.matmul(_X, _W[i]) + _B[i])
	return tf.matmul(_X, _W[n - 1]) + _B[n - 1]

#Add L2 regularizers for the weight and bias matrices
#_W: The weight matrices
#_B: The bias matrices
#return: tensorflow variable representing l2 regularization cost
def _CreateL2Reg(_W, _B):
	n = len(_W)
	regularizers = tf.nn.l2_loss(_W[0]) + tf.nn.l2_loss(_B[0])
	for i in range(1, n):
		regularizers += tf.nn.l2_loss(_W[i]) + tf.nn.l2_loss(_B[i])
	return regularizers

#Create weight and bias vectors for an MLP
#layers: The number of neurons in each layer (including input and output)
#return: A tuple of lists of the weight and bias matrices respectively
def _CreateVars(layers):
	weight = []
	bias = []
	n = len(layers)
	for i in range(n - 1):
		#Fan-in for layer; used as standard dev
		lyrstd = np.sqrt(1.0 / layers[i])
		curW = tf.Variable(tf.random_normal([layers[i], layers[i + 1]], stddev = lyrstd))
		weight.append(curW)
		curB = tf.Variable(tf.random_normal([layers[i + 1]], stddev = lyrstd))
		bias.append(curB)
	return (weight, bias)

#Helper function for selecting an activation function
#name: The name of the activation function
#return: A handle for the tensorflow activation function
def _GetActvFn(name):
	if name == 'tanh':
		return tf.tanh
	elif name == 'sig':
		return tf.sigmoid
	elif name == 'relu':
		return tf.nn.relu
	elif name == 'relu6':
		return tf.nn.relu6
	elif name == 'elu':
		return tf.nn.elu
	elif name == 'softplus':
		return tf.nn.softplus
	elif name == 'softsign':
		return tf.nn.softsign
	return None

#Helper function for getting a tensorflow optimizer
#name:	The name of the optimizer to use
#lr:	  The learning rate if applicable
#return;  A the tensorflow optimization object
def _GetOptimizer(name, lr):
	if(name == 'adam'):
		return tf.train.AdamOptimizer(learning_rate = lr)
	elif(name == 'grad'):
		return tf.train.GradientDescentOptimizer(learning_rate = lr)
	elif(name == 'adagrad'):
		return tf.train.AdagradOptimizer(learning_rate = lr)
	elif(name == 'ftrl'):
		return tf.train.FtrlOptimizer(learning_rate = lr)
	return None

#TensorFlow Artificial Neural Network (Base Class)
class ANN:
	#Sets up the graph for a convolutional neural network
	#from a list of specifications of the form:
	#[('C', [5, 5, 3, 64], [1, 1, 1, 1]), ('P', [1, 3, 3, 1], [1, 2, 2, 1]),('F', 10),]
	#Where 'C' denotes a covolution layer, 'P' denotes a pooling layer, and 'F' denotes
	#a fully-connected layer.
	def _CreateCNN(self, ws):
		self.W = []
		self.B = []
		YH = self.X
		for i, wsi in enumerate(ws):
			if wsi[0] == 'C':	   #Convolutional layer
				self.W.append(tf.Variable(tf.truncated_normal(wsi[1], stddev = 5e-2)))
				self.B.append(tf.Variable(tf.constant(0.0, shape = [wsi[1][-1]])))
				YH = tf.nn.conv2d(YH, self.W[-1], wsi[2], padding = self.pad)
				YH = tf.nn.bias_add(YH, self.B[-1])
				YH = self.AF(YH)	#Apply the activation function to the output
			elif wsi[0] == 'P':	 #Pooling layer
				YH = tf.nn.max_pool(YH, ksize = wsi[1], strides = wsi[2], padding = self.pad)
				YH = tf.nn.lrn(YH, 4, bias=1.0, alpha = 0.001 / 9.0, beta = 0.75)
			elif wsi[0] == 'F':	 #Fully-connected layer
				#Flatten volume of previous layer for fully-connected layer
				yhs = YH.get_shape()
				lls = 1
				for i in yhs[1:]:
					lls *= i.value
				YH = tf.reshape(YH, [-1, lls])
				self.W.append(tf.Variable(tf.truncated_normal([lls, wsi[1]], stddev = 0.04)))
				self.B.append(tf.Variable(tf.constant(0.1, shape = [wsi[1]])))
				YH = tf.matmul(YH, self.W[-1]) + self.B[-1]
				if i + 1 != len(ws):	#Last layer shouldn't apply activation function
					YH = self.AF(YH)
		return YH

	#Clean-up resources
	def __del__(self):
		self.sess.close()

	#Common arguments for all artificial neural network regression models
	#actvFn: The activation function to use: 'tanh', 'sig', or 'relu'
	#batchSize: Size of training batches to use (use all if None)
	#learnRate: The learning rate parameter for the optimizer
	#maxIter: Maximum number of training iterations
	#optmzr: The optimizer method to use ('adam', 'grad', 'adagrad', or 'ftrl')
	#reg: Weight of the regularization term (None for no regularization)
	#tol: Training ends if error falls below this tolerance
	#verbose: Print training information
	def __init__(self, actvFn = 'relu', batchSize = None, learnRate = 1e-4, maxIter = 1000,
					 optmzr = 'adam', reg = None, tol = 1e-1, verbose = False):
		#Activation function to use
		self.AF = _GetActvFn(actvFn)
		#Batch size
		self.batSz = batchSize
		#Learning rate
		self.lr = learnRate
		#Maximum number of iterations
		self.mIter = maxIter
		#Optimizer method
		self.opt = optmzr
		#Regularization strength
		self.reg = reg
		#Flag for stopping training early
		self.stopIter = False
		#Error tolerance
		self.tol = tol
		#Verbose output
		self.vrbse = verbose
		self.saver = None
		#Data members to be populated in a subclass
		self.loss = None
		self.optmzr = None
		self.sess = None
		self.X = None
		self.Y = None
		self.YH = None
		
	#Fit the MLP to the data
	#A: numpy matrix where each row is a sample
	#Y: numpy matrix of target values
	def fit(self, A, Y):
		m = len(A)
		#Begin training
		for i in range(self.mIter):
			#Batch mode or all at once
			if self.batSz is None:	#Compute loss and optimize simultaneously
				err, _ = self.sess.run([self.loss, self.optmzr], feed_dict={self.X:A, self.Y:Y})
			else:	#Train m samples using random batches of size self.bs
				err = 0.0
				for j in range(0, m, self.batSz): #Compute loss and optimize simultaneously
					bi = np.random.randint(m, size = self.batSz)	#Randomly chosen batch indices
					l, _ = self.sess.run([self.loss, self.optmzr], feed_dict = {self.X:A[bi], self.Y:Y[bi]})
					err += l	#Accumulate loss over all batches
			if self.vrbse:
				print("Iter {:5d}\t{:.8f}".format(i + 1, err))
			if err < self.tol or self.stopIter:
				break	#Stop if tolerance was reached or flag was set
				
	#Predict the output given the input (only run after calling fit)
	#A: The input values for which to predict outputs
	#return: The predicted output values (one row per input sample)
	def predict(self, A):
		if self.sess is None:
			raise Exception("Error: MLP has not yet been fitted.")
		return self.sess.run(self.YH, feed_dict={self.X:A})

	#Restores a model that was previously created with SaveModel
	def RestoreModel(self, p, name):
		try:
			sav = tf.train.import_meta_graph(p + name + '.meta')
			sav.restore(self.sess, tf.train.latest_checkpoint(p))
		except OSError:
			return False
		return True

	#Start the TensorFlow session
	def RunSession(self):
		self.saver = tf.train.Saver()
		#Initialize all variables on the TF session
		self.sess = tf.Session()
		init = tf.global_variables_initializer()
		self.sess.run(init)

	#Save model to file; the format should be like "path/prefixname"
	#If using the current directory "./prefixname" may need to be used instead
	#of simply "prefixname"
	def SaveModel(self, name):
		self.saver.save(self.sess, name)

	#Update the maximum number of iterations
	def SetMaxIter(self, mi):
		self.mIter = mi
		
	#Set the flag for stopping iteration early
	def SetStopIter(self, si):
		self.stopIter = si

#Artificial Neural Network for Regression (Base Class)
class ANNR(ANN):
	#Common arguments for all artificial neural network regression models
	def __init__(self, actvFn = 'relu', batchSize = None, learnRate = 1e-4, maxIter = 1000,
				 optmzr = 'adam', reg = None, tol = 1e-1, verbose = False):
		#Initialize fields from base class
		super().__init__(actvFn, batchSize, learnRate, maxIter, optmzr, reg, tol, verbose)

	#Predicts the ouputs for input A and then computes the RMSE between
	#The predicted values and the actualy values
	#A: The input values for which to predict outputs
	#Y: The actual target values
	#return: The RMSE
	def score(self, A, Y):
		return np.sqrt(self.sess.run(self.loss, feed_dict = {self.X:A, self.Y:Y}) * 2.0) / len(A)

#Convolotional Neural Network for Regression
class CNNR(ANNR):
	#imageSize:	 Size of the images used (Height, Width, Depth)
	#ws:			Weight matrix sizes
	def __init__(self, imageSize, ws, actvFn = 'relu', batchSize = None, learnRate = 1e-4, maxIter = 1000,
				 optmzr = 'adam', pad = 'SAME', tol = 1e-1, reg = None, verbose = False):
		#Initialize fields from base class
		super().__init__(actvFn, batchSize, learnRate, maxIter, optmzr, reg, tol, verbose)
		#Input placeholder
		self.imgSize = list(imageSize)
		self.X = tf.placeholder("float", [None] + self.imgSize)
		#Padding method to use
		self.pad = pad
		#Target vector placeholder; final layer should be a fully connected layer
		self.Y = tf.placeholder("float", [None, ws[-1][1]])
		#Create neural network graph and keep track of output variable
		self.YH = self._CreateCNN(ws)
		#l2_loss of t is sum(t**2)/2
		self.loss = tf.reduce_sum(tf.nn.l2_loss(self.YH - self.Y))
		#Use regularization to prevent over-fitting
		self.reg = reg
		if reg is not None:
			self.loss += _CreateL2Reg(self.W, self.B) * reg
		self.optmzr = _GetOptimizer(optmzr, learnRate).minimize(self.loss)
		#Begin the TensorFlow Session
		self.RunSession()

#Multi-Layer Perceptron for Regression
class MLPR(ANNR):
	#layers: A list of layer sizes
	def __init__(self, layers, actvFn = 'tanh', batchSize = None, learnRate = 1e-3,
				 maxIter = 2000, optmzr = 'adam',  reg = 1e-3, tol = 1e-2, verbose = False):
		super().__init__(actvFn, batchSize, learnRate, maxIter, optmzr, reg, tol, verbose)
		#Input size
		self.X = tf.placeholder("float", [None, layers[0]])
		#Output size
		self.Y = tf.placeholder("float", [None, layers[-1]])
		#Setup the weight and bias variables
		weight, bias = _CreateVars(layers)
		#Create the tensorflow MLP model
		self.YH = _CreateMLP(self.X, weight, bias, self.AF)
		#l2_loss of t is sum(t**2)/2
		self.loss = tf.reduce_sum(tf.nn.l2_loss(self.YH - self.Y))
		#Use regularization to prevent over-fitting
		if reg is not None:
			self.loss += _CreateL2Reg(weight, bias) * reg
		#Use ADAM method to minimize the loss function
		self.optmzr = _GetOptimizer(optmzr, learnRate).minimize(self.loss)
		#Initialize all variables on the TF session
		self.RunSession()

#Multi-Layer Perceptron for Binary Data
class MLPB(MLPR):
	#layers: A list of layer sizes
	def __init__(self, layers, actvFn = 'tanh', batchSize = None, learnRate = 0.001,
				 maxIter = 2000, optmzr = 'adam', reg = 0.001, tol = 1e-2, verbose = False):
		super().__init__(layers, actvFn, batchSize, learnRate, maxIter, optmzr, reg, tol, verbose)

	#Fit the MLP to the data
	#A: numpy matrix where each row is a sample
	#y: numpy matrix of target values
	def fit(self, A, Y):
		#Transform data for better performance
		A = self.Tx(A)
		Y = self.Tx(Y)
		super().fit(A, Y)

	#Predict the output given the input (only run after calling fit)
	#A: The input values for which to predict outputs
	#return: The predicted output values (one row per input sample)
	def predict(self, A):
		if self.sess == None:
			print("Error: MLPC has not yet been fitted.")
			return None
		A = self.Tx(A)
		YH = super().predict(A)
		#Transform back to un-scaled data
		return self.YHatF(self.TInvX(YH))

	#Convert from prediction (YH) to original data type (integer 0 or 1)
	def YHatF(self, Y):
		return Y.clip(0.0, 1.0).round().astype(np.int)

	#Map from {0, 1} -> {-0.9, .9} (for tanh activation)
	def Tx(self, Y):
		return (2 * Y - 1) * 0.9
	
	#Map from {-0.9, .9} -> {0, 1}
	def TInvX(self, Y):
		return (Y / 0.9 + 1) / 2.0

#Artificial Neural Network for Classification (Base Class)
class ANNC(ANN):
	#Fit the MLP to the data
	#A: numpy matrix where each row is a sample
	#y: numpy matrix of target values
	def fit(self, A, Y):
		Y = self.To1Hot(Y)
		super().fit(A, Y)

	def __init__(self, actvFn = 'tanh', batchSize = None, learnRate = 1e-3, maxIter = 2000,
				 optmzr = 'adam', reg = 1e-3, tol = 1e-2, verbose = False):
		super().__init__(actvFn, batchSize, learnRate, maxIter, optmzr, reg, tol, verbose)
		#Lookup table for class labels
		self._classes = None

	#Predict the output given the input (only run after calling fit)
	#A: The input values for which to predict outputs
	#return: The predicted output values (one row per input sample)
	def predict(self, A):
		#Get the predicted indices
		res = np.argmax(super().predict(A), 1)
		res.shape = [-1]
		#Return prediction using the original labels
		return np.array([self._classes[i] for i in res])

	#Restores the classes lookup table
	def RestoreClasses(self, Y):
		self._classes = sorted(list(set(list(Y))))

	#Predicts the ouputs for input A and then computes the classification error
	#The predicted values and the actualy values
	#A: The input values for which to predict outputs
	#y: The actual target values
	#return: The percent of outputs predicted correctly
	def score(self, A, Y):
		YH = self.predict(A)
		return _Accuracy(Y, YH)

	#Creates an array of 1-hot vectors based on a vector of class labels
	#y: The vector of class labels
	#return: The 1-Hot encoding of y
	def To1Hot(self, Y):
		self._classes = sorted(list(set(list(Y))))		#Reform class labels
		lblDic = {}
		for i, ci in enumerate(self._classes):
			lblDic[ci] = i
		b = np.zeros([len(Y), len(self._classes)])
		for i in range(len(Y)):
			b[i, lblDic[Y[i]]] = 1
		return b

#Convolutional Neural Network for Classification
class CNNC(ANNC):
	#imageSize:	 Size of the images used (Height, Width, Depth)
	#ws:			Weight matrix sizes
	def __init__(self, imageSize, ws, actvFn = 'relu', batchSize = None, learnRate = 1e-4, maxIter = 1000,
				 optmzr = 'adam', pad = 'SAME', tol = 1e-1, reg = None, verbose = False):
		#Initialize fields from base class
		super().__init__(actvFn, batchSize, learnRate, maxIter, optmzr, reg, tol, verbose)
		#Input placeholder
		self.imgSize = list(imageSize)
		self.X = tf.placeholder("float", [None] + self.imgSize)
		#Padding method to use
		self.pad = pad
		#Target vector placeholder; final layer should be a fully connected layer
		self.Y = tf.placeholder("float", [None, ws[-1][1]])
		#Create neural network graph and keep track of output variable
		self.YH = self._CreateCNN(ws)
		#Loss term
		self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = self.YH, labels = self.Y))
		#Use regularization to prevent over-fitting
		self.reg = reg
		if reg is not None:
			self.loss += _CreateL2Reg(self.W, self.B) * reg
		self.optmzr = _GetOptimizer(optmzr, learnRate).minimize(self.loss)
		#Begin the TensorFlow Session
		self.RunSession()

#Multi-Layer Perceptron for Classification
class MLPC(ANNC):
	#layers: A list of layer sizes
	def __init__(self, layers, actvFn = 'tanh', batchSize = None, learnRate = 1e-3, maxIter = 2000,
				 optmzr = 'adam', reg = 1e-3, tol = 1e-2, verbose = False):
		super().__init__(actvFn, batchSize, learnRate, maxIter, optmzr, reg, tol, verbose)
		#Input size
		self.X = tf.placeholder("float", [None, layers[0]])
		#Output size
		self.Y = tf.placeholder("float", [None, layers[-1]])
		#Setup the weight and bias variables
		weight, bias = _CreateVars(layers)
		#Create the tensorflow model
		self.YH = _CreateMLP(self.X, weight, bias, self.AF)
		#Cross entropy loss function
		self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = self.YH, labels = self.Y))
		#Use regularization to prevent over-fitting
		if reg is not None:
			self.loss += _CreateL2Reg(weight, bias) * reg
		self.optmzr = _GetOptimizer(optmzr, learnRate).minimize(self.loss)
		#Initialize all variables on the TF session
		self.RunSession()