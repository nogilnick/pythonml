#TFANN.py
#This file contains classes implementing ANN using numpy 
#and tensorflow. Presently, multi-layer perceptron (MLP) 
#and convolutional neural networks (CNN) are supported.
import tensorflow as tf
import numpy as np

def _Accuracy(Y, YH):
	'''
	Return the classification accuracy
	given a vector of target labels and
	predicted labels
	y: The target labels
	yHat: The predicted labels
	return: The percentage correct
	'''
	n = float(len(Y))
	return np.sum(Y == YH) / n

def _CreateMLP(_X, _W, _B, _AF):
	'''
	Create the MLP variables for TF graph
	_X: The input matrix
	_W: The weight matrices
	_B: The bias vectors
	_AF: The activation function
	'''
	n = len(_W)
	for i in range(n - 1):
		_X = _AF(tf.matmul(_X, _W[i]) + _B[i])
	return tf.matmul(_X, _W[n - 1]) + _B[n - 1]

def _CreateL2Reg(_W, _B):
	'''
	Add L2 regularizers for the weight and bias matrices
	_W: The weight matrices
	_B: The bias matrices
	return: tensorflow variable representing l2 regularization cost
	'''
	n = len(_W)
	regularizers = tf.nn.l2_loss(_W[0]) + tf.nn.l2_loss(_B[0])
	for i in range(1, n):
		regularizers += tf.nn.l2_loss(_W[i]) + tf.nn.l2_loss(_B[i])
	return regularizers

def _CreateVars(layers):
	'''
	Create weight and bias vectors for an MLP
	layers: The number of neurons in each layer (including input and output)
	return: A tuple of lists of the weight and bias matrices respectively
	'''
	weight = []
	bias = []
	n = len(layers)
	for i in range(n - 1):
		lyrstd = np.sqrt(1.0 / layers[i])			#Fan-in for layer; used as standard dev
		curW = tf.Variable(tf.random_normal([layers[i], layers[i + 1]], stddev = lyrstd))
		weight.append(curW)
		curB = tf.Variable(tf.random_normal([layers[i + 1]], stddev = lyrstd))
		bias.append(curB)
	return (weight, bias)

def _GetActvFn(name):
	'''
	Helper function for selecting an activation function
	name: The name of the activation function
	return: A handle for the tensorflow activation function
	'''
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
	
def _GetLossFn(name):
	'''
	Helper function for selecting loss function
	name: 	The name of the loss function
	return: 	A handle for a loss function LF(YH, Y)
	'''
	if name == 'l2':
		return lambda YH, Y : tf.squared_difference(Y, YH)
	elif name == 'l1':
		return lambda YH, Y : tf.losses.absolute_difference(Y, YH)
	elif name == 'smce':
		return lambda YH, Y : tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = YH)
	elif name == 'sgce':
		return lambda YH, Y : tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = YH)
	elif name == 'cos':
		return lambda YH, Y : tf.losses.cosine_distance(Y, YH)
	elif name == 'log':
		return lambda YH, Y : tf.losses.log_loss(Y, YH)
	elif name == 'hinge':
		return lambda YH, Y : tf.losses.hinge_loss(Y, YH)
	return None
		
	

def _GetOptimizer(name, lr):
	'''
	Helper function for getting a tensorflow optimizer
	name:	The name of the optimizer to use
	lr:	  The learning rate if applicable
	return;  A the tensorflow optimization object
	'''
	if name == 'adam':
		return tf.train.AdamOptimizer(learning_rate = lr)
	elif name == 'grad':
		return tf.train.GradientDescentOptimizer(learning_rate = lr)
	elif name == 'adagrad':
		return tf.train.AdagradOptimizer(learning_rate = lr)
	elif name == 'ftrl':
		return tf.train.FtrlOptimizer(learning_rate = lr)
	return None

class ANN:
	'''
	TensorFlow Artificial Neural Network (Base Class)
	'''

	def _CreateCNN(self, ws):
		'''
		Sets up the graph for a convolutional neural network from a list of specifications 
		of the form:
		[('C', [5, 5, 3, 64], [1, 1, 1, 1]), ('P', [1, 3, 3, 1], [1, 2, 2, 1]),('F', 10),]
		Where 'C' denotes a covolution layer, 'P' denotes a pooling layer, and 'F' denotes
		a fully-connected layer.'''
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

	def __init__(self, actvFn = 'relu', batchSize = None, learnRate = 1e-4, loss = 'l2', maxIter = 1000,
					 optmzr = 'adam', reg = None, tol = 1e-1, verbose = False):
		'''
		Common arguments for all artificial neural network regression models
		actvFn: The activation function to use: 'tanh', 'sig', or 'relu'
		batchSize: Size of training batches to use (use all if None)
		learnRate: The learning rate parameter for the optimizer
		loss: 	The name of the loss function (l2, l1, smce, sgme, cos, log, hinge)	
		maxIter: Maximum number of training iterations
		optmzr: The optimizer method to use ('adam', 'grad', 'adagrad', or 'ftrl')
		reg: Weight of the regularization term (None for no regularization)
		tol: Training ends if error falls below this tolerance
		verbose: Print training information
		'''
		self.AF = _GetActvFn(actvFn)					#Activation function to use
		self.LF = _GetLossFn(loss) 					#Handle to loss function to use
		self.batSz = batchSize						#Batch size
		self.lr = learnRate							#Learning rate
		self.mIter = maxIter						#Maximum number of iterations
		self.opt = optmzr							#Optimizer method
		self.reg = reg								#Regularization strength
		self.stopIter = False						#Flag for stopping training early
		self.tol = tol								#Error tolerance
		self.vrbse = verbose						#Verbose output
		self.saver = None
		#Data members to be populated in a subclass
		self.loss = None
		self.optmzr = None
		self.sess = None
		self.X = None
		self.Y = None
		self.YH = None
		
	def fit(self, A, Y):
		'''
		Fit the ANN to the data
		A: numpy matrix where each row is a sample
		Y: numpy matrix of target values
		'''
		m = len(A)
		for i in range(self.mIter):							#Loop up to mIter times
			if self.batSz is None:							#Compute loss and optimize simultaneously for all samples
				err, _ = self.sess.run([self.loss, self.optmzr], feed_dict={self.X:A, self.Y:Y})
			else:											#Train m samples using random batches of size self.bs
				err = 0.0
				for j in range(0, m, self.batSz): 			#Compute loss and optimize simultaneously for batch
					bi = np.random.randint(m, size = self.batSz)	#Randomly chosen batch indices
					l, _ = self.sess.run([self.loss, self.optmzr], feed_dict = {self.X:A[bi], self.Y:Y[bi]})
					err += l								#Accumulate loss over all batches
				err /= len(range(0, m, self.batSz))			#Average over all batches
			if self.vrbse:
				print("Iter {:5d}\t{:.8f}".format(i + 1, err))
			if err < self.tol or self.stopIter:
				break	#Stop if tolerance was reached or flag was set
				
	def predict(self, A):
		'''
		Predict the output given the input (only run after calling fit)
		A: The input values for which to predict outputs
		return: The predicted output values (one row per input sample)
		'''
		if self.sess is None:
			raise Exception("Error: MLP has not yet been fitted.")
		return self.sess.run(self.YH, feed_dict = {self.X:A})

	def RestoreModel(self, p, name):
		'''
		Restores a model that was previously created with SaveModel
		p:		Path to model containing files
		name:	Name of model to restore
		'''
		try:
			sav = tf.train.import_meta_graph(p + name + '.meta')
			sav.restore(self.sess, tf.train.latest_checkpoint(p))
		except OSError:
			return False
		return True

	def RunSession(self):
		'''
		Start the TensorFlow session
		'''
		self.saver = tf.train.Saver()					#For saving a model for later restoration
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True			#Tell TensorFlow to use GPU memory as needed
		self.sess = tf.Session(config = config)			#instead of allocating all up-front
		init = tf.global_variables_initializer()		
		self.sess.run(init)							#Initialize all variables on the TF session

	def SaveModel(self, name):
		'''
		Save model to file; the format should be like "path/prefixname"
		If using the current directory "./prefixname" may need to be used instead
		of simply "prefixname"
		'''
		self.saver.save(self.sess, name)

	def SetMaxIter(self, mi):
		'''
		Update the maximum number of iterations to run for fit
		'''
		self.mIter = mi
		
	def SetStopIter(self, si):
		'''
		Set the flag for stopping iteration early
		'''
		self.stopIter = si

class ANNR(ANN):
	'''
	Artificial Neural Network for Regression (Base Class)
	'''
	
	def __init__(self, actvFn = 'relu', batchSize = None, learnRate = 1e-4, loss = 'l2', 
			  maxIter = 1000, optmzr = 'adam', reg = None, tol = 1e-1, verbose = False):
		'''
		Constructor with common argument for all artificial neural network regression models
		'''
		#Initialize fields from base class
		super().__init__(actvFn, batchSize, learnRate, loss, maxIter, optmzr, reg, tol, verbose)

	def score(self, A, Y):
		'''
		Predicts the ouputs for input A and then computes the RMSE between
		The predicted values and the actualy values
		A: The input values for which to predict outputs
		Y: The actual target values
		return: The RMSE
		'''
		return np.sqrt(self.sess.run(self.loss, feed_dict = {self.X:A, self.Y:Y}) * 2.0) / len(A)

class CNNR(ANNR):
	'''
	Convolutional Neural Network for Regression
	'''
	
	def __init__(self, imageSize, ws, actvFn = 'relu', batchSize = None, learnRate = 1e-4, loss = 'l2',
			   maxIter = 1000, optmzr = 'adam', pad = 'SAME', tol = 1e-1, reg = None, verbose = False):
		'''
		imageSize:	Size of the images used (Height, Width, Depth)
		ws:			Weight matrix sizes
		'''
		#Initialize fields from base class
		super().__init__(actvFn, batchSize, learnRate, loss, maxIter, optmzr, reg, tol, verbose)
		self.imgSize = list(imageSize)
		self.X = tf.placeholder("float", [None] + self.imgSize)			#Input data matrices (batch of RGB images)
		self.pad = pad													#Padding method to use
		self.Y = tf.placeholder("float", [None, ws[-1][1]])				#Target value placeholder
		self.YH = self._CreateCNN(ws)									#Create graph; YH is output from feedforward
		self.loss = tf.reduce_mean(self.LF(self.YH, self.Y))		#l2_loss of t is sum(t**2)/2
		self.reg = reg													#Regularization can prevent over-fitting
		if reg is not None:
			self.loss += _CreateL2Reg(self.W, self.B) * reg
		self.optmzr = _GetOptimizer(optmzr, learnRate).minimize(self.loss)
		self.RunSession()												#Begin the TensorFlow Session

class MLPR(ANNR):
	'''
	Multi-Layer Perceptron for Regression
	'''
	
	def __init__(self, layers, actvFn = 'tanh', batchSize = None, learnRate = 1e-3, loss = 'l2',
				 maxIter = 2000, optmzr = 'adam',  reg = 1e-3, tol = 1e-2, verbose = False):
		'''
		layers: A list of layer sizes
		'''
		super().__init__(actvFn, batchSize, learnRate, loss, maxIter, optmzr, reg, tol, verbose)
		self.X = tf.placeholder("float", [None, layers[0]])				#Input data matrix
		self.Y = tf.placeholder("float", [None, layers[-1]])				#Target value matrix
		weight, bias = _CreateVars(layers)								#Setup the weight and bias variables
		self.YH = _CreateMLP(self.X, weight, bias, self.AF)				#Create the tensorflow MLP model; YH is graph output
		self.loss = tf.reduce_mean(self.LF(self.YH, self.Y))				#l2_loss of t is sum(t**2)/2
		if reg is not None:											#Regularization can prevent over-fitting
			self.loss += _CreateL2Reg(weight, bias) * reg
		self.optmzr = _GetOptimizer(optmzr, learnRate).minimize(self.loss)	#Get optimizer method to minimize the loss function
		self.RunSession()											#Start TF session				

class MLPB(MLPR):
	'''
	Multi-Layer Perceptron for Binary Data
	'''
	
	def __init__(self, layers, actvFn = 'tanh', batchSize = None, learnRate = 0.001, loss = 'l2',
				 maxIter = 2000, optmzr = 'adam', reg = 0.001, tol = 1e-2, verbose = False):
		'''
		layers: A list of layer sizes
		'''
		super().__init__(layers, actvFn, batchSize, learnRate, loss, maxIter, optmzr, reg, tol, verbose)

	def fit(self, A, Y):
		'''
		Fit the MLP to the data
		A: numpy matrix where each row is a sample
		y: numpy matrix of target values
		'''
		A = self.Tx(A)			#Transform data for better performance
		Y = self.Tx(Y)			#with tanh activation
		super().fit(A, Y)
		
	def predict(self, A):
		'''
		Predict the output given the input (only run after calling fit)
		A: The input values for which to predict outputs
		return: The predicted output values (one row per input sample)
		'''
		if self.sess == None:
			print("Error: MLPC has not yet been fitted.")
			return None
		A = self.Tx(A)
		YH = super().predict(A)
		#Transform back to un-scaled data
		return self.YHatF(self.TInvX(YH))

	def YHatF(self, Y):
		'''
		Convert from prediction (YH) to original data type (integer 0 or 1)
		'''
		return Y.clip(0.0, 1.0).round().astype(np.int)

	def Tx(self, Y):
		'''
		Map from {0, 1} -> {-0.9, .9} (for tanh activation)
		'''
		return (2 * Y - 1) * 0.9
	
	def TInvX(self, Y):
		'''
		Map from {-0.9, .9} -> {0, 1}
		'''
		return (Y / 0.9 + 1) / 2.0

class ANNC(ANN):
	'''
	Artificial Neural Network for Classification (Base Class)
	'''

	def fit(self, A, Y):
		'''
		Fit the MLP to the data
		A: numpy matrix where each row is a sample
		y: numpy matrix of target values
		'''
		Y = self.To1Hot(Y)
		super().fit(A, Y)

	def __init__(self, actvFn = 'tanh', batchSize = None, learnRate = 1e-3, loss = 'smce', 
			   maxIter = 2000, optmzr = 'adam', reg = 1e-3, tol = 1e-2, verbose = False):
		super().__init__(actvFn, batchSize, learnRate, loss, maxIter, optmzr, reg, tol, verbose)
		#Lookup table for class labels
		self._classes = None

	def predict(self, A):
		'''
		Predict the output given the input (only run after calling fit)
		A: The input values for which to predict outputs
		return: The predicted output values (one row per input sample)
		'''
		res = np.argmax(super().predict(A), 1)			#Get the predicted indices
		res.shape = [-1]
		#Return prediction using the original labels
		return np.array([self._classes[i] for i in res])

	def RestoreClasses(self, Y):
		'''
		#Restores the classes lookup table
		'''
		self._classes = sorted(list(set(list(Y))))

	def score(self, A, Y):
		'''
		Predicts the ouputs for input A and then computes the classification error
		The predicted values and the actualy values
		A: The input values for which to predict outputs
		y: The actual target values
		return: The percent of outputs predicted correctly
		'''
		YH = self.predict(A)
		return _Accuracy(Y, YH)

	def To1Hot(self, Y):
		'''
		Creates an array of 1-hot vectors based on a vector of class labels
		Y: The vector of class labels
		#return: The 1-Hot encoding of Y
		'''
		self._classes = sorted(list(set(list(Y))))		#Reform class labels
		lblDic = {}
		for i, ci in enumerate(self._classes):
			lblDic[ci] = i
		b = np.zeros([len(Y), len(self._classes)])
		for i in range(len(Y)):
			b[i, lblDic[Y[i]]] = 1
		return b

class CNNC(ANNC):
	'''
	Convolutional Neural Network for Classification
	'''

	def __init__(self, imageSize, ws, actvFn = 'relu', batchSize = None, learnRate = 1e-4, loss = 'smce',
			   maxIter = 1000, optmzr = 'adam', pad = 'SAME', tol = 1e-1, reg = None, verbose = False):
		'''
		imageSize:	Size of the images used (Height, Width, Depth)
		ws:			Weight matrix sizes
		'''
		#Initialize fields from base class
		super().__init__(actvFn, batchSize, learnRate, loss, maxIter, optmzr, reg, tol, verbose)
		self.imgSize = list(imageSize)
		self.X = tf.placeholder("float", [None] + self.imgSize)				#Input data matrix of samples
		self.pad = pad													#Padding method to use
		self.Y = tf.placeholder("float", [None, ws[-1][1]])					#Target matrix
		self.YH = self._CreateCNN(ws)										#Create graph; YH is output matrix
		#Loss term
		self.loss = tf.reduce_mean(self.LF(self.YH, self.Y))
		self.reg = reg
		if reg is not None:												#Regularization can prevent over-fitting
			self.loss += _CreateL2Reg(self.W, self.B) * reg
		self.optmzr = _GetOptimizer(optmzr, learnRate).minimize(self.loss)
		self.RunSession()												#Begin the TensorFlow Session

class MLPC(ANNC):
	'''
	Multi-Layer Perceptron for Classification
	'''
	
	def __init__(self, layers, actvFn = 'tanh', batchSize = None, learnRate = 1e-3, loss = 'smce',
			   maxIter = 2000, optmzr = 'adam', reg = 1e-3, tol = 1e-2, verbose = False):
		'''
		layers: A list of layer sizes
		'''
		super().__init__(actvFn, batchSize, learnRate, loss, maxIter, optmzr, reg, tol, verbose)
		self.X = tf.placeholder("float", [None, layers[0]])					#Input data matrix
		self.Y = tf.placeholder("float", [None, layers[-1]])					#Target matrix
		weight, bias = _CreateVars(layers)									#Setup the weight and bias variables
		self.YH = _CreateMLP(self.X, weight, bias, self.AF)					#Create the tensorflow model; YH is output matrix
		#Cross entropy loss function
		self.loss = tf.reduce_mean(self.LF(self.YH, self.Y))
		if reg is not None:												#Regularization can prevent over-fitting
			self.loss += _CreateL2Reg(weight, bias) * reg
		self.optmzr = _GetOptimizer(optmzr, learnRate).minimize(self.loss)
		self.RunSession()												#Start the TF session