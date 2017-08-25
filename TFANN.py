'''
TFANN.py
This file contains classes implementing ANN using numpy 
and tensorflow. Presently, multi-layer perceptron (MLP) 
and convolutional neural networks (CNN) are supported.
'''
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
    return np.mean(Y == YH)

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

def _CreateRFSMLP(_X, _W, _B, _AF):
    '''
    Create the RFSMLP variables for TF graph
    _X: The input matrix
    _W: The weight matrices
    _B: The bias vectors
    _AF: The activation function
    '''
    X = [_X]
    n = len(_W)
    for i in range(n - 1):
        X.append(_AF(tf.matmul(X[-1], _W[i]) + _B[i]))
    X.append(tf.matmul(X[-1], _W[n - 1]) + _B[n - 1])
    return tf.stack(X, axis = 1)

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
        lyrstd = np.sqrt(1.0 / layers[i])           #Fan-in for layer; used as standard dev
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
    name:   The name of the loss function
    return:     A handle for a loss function LF(YH, Y)
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
    name:   The name of the optimizer to use
    lr:   The learning rate if applicable
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
    sess = None     #Class variable shared among all instances
    sessc = 0       #Number of active sessions
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
            if wsi[0] == 'C':           #Convolutional layer
                self.W.append(tf.Variable(tf.truncated_normal(wsi[1], stddev = 5e-2)))
                self.B.append(tf.Variable(tf.constant(0.0, shape = [wsi[1][-1]])))
                YH = tf.nn.conv2d(YH, self.W[-1], wsi[2], padding = self.pad)
                YH = tf.nn.bias_add(YH, self.B[-1])
                YH = self.AF(YH)        #Apply the activation function to the output
            elif wsi[0] == 'P':         #Pooling layer
                YH = tf.nn.max_pool(YH, ksize = wsi[1], strides = wsi[2], padding = self.pad)
                YH = tf.nn.lrn(YH, 4, bias=1.0, alpha = 0.001 / 9.0, beta = 0.75)
            elif wsi[0] == 'F':         #Fully-connected layer
                yhs = YH.get_shape()    #Flatten volume of previous layer for fully-connected layer
                lls = 1
                for i in yhs[1:]:
                    lls *= i.value
                YH = tf.reshape(YH, [-1, lls])
                self.W.append(tf.Variable(tf.truncated_normal([lls, wsi[1]], stddev = 0.04)))
                self.B.append(tf.Variable(tf.constant(0.1, shape = [wsi[1]])))
                YH = tf.matmul(YH, self.W[-1]) + self.B[-1]
                if i + 1 != len(ws):    #Last layer shouldn't apply activation function
                    YH = self.AF(YH)
        return YH

    #Clean-up resources
    def __del__(self):
        ANN.sessc -= 1          #Decrement session counter
        if ANN.sessc == 0:      #Close session only if not in use
            self.GetSes().close()
            ANN.sess = None

    def __init__(self, actvFn = 'relu', batchSize = None, learnRate = 1e-4, loss = 'l2', maxIter = 1024,
                 name = 'tfann', optmzrn = 'adam', reg = None, tol = 1e-1, verbose = False):
        '''
        Common arguments for all artificial neural network regression models
        actvFn: The activation function to use: 'tanh', 'sig', or 'relu'
        batchSize: Size of training batches to use (use all if None)
        learnRate: The learning rate parameter for the optimizer
        loss:   The name of the loss function (l2, l1, smce, sgme, cos, log, hinge) 
        name:   The name of the model for variable scope, saving, and restoring
        maxIter: Maximum number of training iterations
        optmzrn: The optimizer method to use ('adam', 'grad', 'adagrad', or 'ftrl')
        reg: Weight of the regularization term (None for no regularization)
        tol: Training ends if error falls below this tolerance
        verbose: Print training information
        '''
        self.AF = _GetActvFn(actvFn)                #Activation function to use
        self.LF = _GetLossFn(loss)                  #Handle to loss function to use
        self.batSz = batchSize                      #Batch size
        self.lr = learnRate                         #Learning rate
        self.mIter = maxIter                        #Maximum number of iterations
        self.name = name                            #Name of model for variable scope
        self.optmzrn = optmzrn                      #Optimizer method name
        self.reg = reg                              #Regularization strength
        self.stopIter = False                       #Flag for stopping training early
        self.tol = tol                              #Error tolerance
        self.vrbse = verbose                        #Verbose output
        self.saver = None
        #Data members to be populated in a subclass
        self.loss = None
        self.optmzr = None
        self.X = None
        self.Y = None
        self.YH = None
        self.TFVar = None
        with tf.variable_scope(self.name):
            self.TFVar = set(tf.global_variables())                 #All variables before model
            self.CreateModel()                                      #Populate TensorFlow graph
            self.TFVar = set(tf.global_variables()) - self.TFVar    #Only variables in this model
            self.Initialize()                                       #Start TF session and initialize vars 
            
    def CreateModel(self):
        '''
        Creates the acutal model (implemented in subclasses)
        '''
        pass
        
    def fit(self, A, Y):
        '''
        Fit the ANN to the data
        A: numpy matrix where each row is a sample
        Y: numpy matrix of target values
        '''
        m = len(A)
        for i in range(self.mIter):                         #Loop up to mIter times
            if self.batSz is None:                          #Compute loss and optimize simultaneously for all samples
                err, _ = self.GetSes().run([self.loss, self.optmzr], feed_dict={self.X:A, self.Y:Y})
            else:                                           #Train m samples using random batches of size self.bs
                err = 0.0
                for j in range(0, m, self.batSz):                   #Compute loss and optimize simultaneously for batch
                    bi = np.random.randint(m, size = self.batSz)    #Randomly chosen batch indices
                    l, _ = self.GetSes().run([self.loss, self.optmzr], feed_dict = {self.X:A[bi], self.Y:Y[bi]})
                    err += l                                        #Accumulate loss over all batches
                err /= len(range(0, m, self.batSz))                 #Average over all batches
            if self.vrbse:
                print("Iter {:5d}\t{:.8f}".format(i + 1, err))
            if err < self.tol or self.stopIter:
                break   #Stop if tolerance was reached or flag was set
                
    def GetBias(self, i):
        '''
        Gets the i-th bias vector from the model
        '''
        return self.B[i].eval(session = self.GetSes())
                
    def GetSes(self):
        if ANN.sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True      #Only use GPU memory as needed
            ANN.sess = tf.Session(config = config)      #instead of allocating all up-front
        return ANN.sess
                
    def GetWeightMatrix(self, i):
        '''
        Gets the i-th weight matrix from the model
        '''
        return self.W[i].eval(session = self.GetSes())

    def Initialize(self):
        '''
        Initialize variables and start the TensorFlow session
        '''
        self.saver = tf.train.Saver()               #For saving a model for later restoration
        ANN.sessc += 1                              #Increment session counter
        sess = self.GetSes()
        init = tf.variables_initializer(self.TFVar)
        sess.run(init)                              #Initialize all variables for this model
        
    def predict(self, A):
        '''
        Predict the output given the input (only run after calling fit)
        A: The input values for which to predict outputs
        return: The predicted output values (one row per input sample)
        '''
        return self.GetSes().run(self.YH, feed_dict = {self.X:A})

    def Reinitialize(self):
        '''
        Reinitialize variables and start the TensorFlow session
        '''
        sess = self.GetSes()
        init = tf.variables_initializer(self.TFVar)
        sess.run(init)                              #Initialize all variables for this model
        
    def Reset():
        '''
        Performs a hard reset of the tensorflow graph and session
        '''
        ANN.sessc = 0
        if ANN.sess is not None:
            ANN.sess.close()
        ANN.sess = None
        tf.reset_default_graph()
        
    def RestoreModel(self, p, n):
        '''
        Restores a model that was previously created with SaveModel
        p:      Path to model containing files like "path/"
        '''
        try:
            sav = tf.train.import_meta_graph(p + n + '.meta')
            sav.restore(self.GetSes(), tf.train.latest_checkpoint(p))
        except OSError as ose:
            print('Error Restoring: ' + str(ose))
            return False
        return True

    def SaveModel(self, p):
        '''
        Save model to file
        p:  Path to save file like "path/"
        '''
        self.saver.save(self.GetSes(), p)

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
    
    def YHatF(self, Y):
        '''
        Convert from prediction (YH) to original data type (integer 0 or 1)
        '''
        return Y.clip(0.0, 1.0).round().astype(np.int)

class ANNR(ANN):
    '''
    Artificial Neural Network for Regression (Base Class)
    '''
   
    def score(self, A, Y):
        '''
        Predicts the ouputs for input A and then computes the RMSE between
        The predicted values and the actualy values
        A: The input values for which to predict outputs
        Y: The actual target values
        return: The RMSE
        '''
        return np.sqrt(self.GetSes().run(self.loss, feed_dict = {self.X:A, self.Y:Y}) * 2.0) / len(A)

class CNNR(ANNR):
    '''
    Convolutional Neural Network for Regression
    '''
    
    def __init__(self, imageSize, ws, actvFn = 'relu', batchSize = None, learnRate = 1e-4, loss = 'l2', maxIter = 1024, 
                 name = 'tfann', optmzrn = 'adam', pad = 'SAME', tol = 1e-1, reg = None, verbose = False):
        '''
        imageSize:  Size of the images used (Height, Width, Depth)
        ws:         Weight matrix sizes
        '''
        self.imgSize = list(imageSize)      #Image size for CNN
        self.ws = ws                        #Add data member so CreateModel can access
        self.pad = pad                                                  #Padding method to use
        super().__init__(actvFn, batchSize, learnRate, loss, maxIter, name, optmzrn, reg, tol, verbose)
        
    def CreateModel(self):
        self.X = tf.placeholder("float", [None] + self.imgSize)         #Input data matrices (batch of RGB images)
        self.Y = tf.placeholder("float", [None, self.ws[-1][1]])        #Target value placeholder
        self.YH = self._CreateCNN(self.ws)                              #Create graph; YH is output from feedforward
        self.loss = tf.reduce_mean(self.LF(self.YH, self.Y))            
        if self.reg is not None:                                        #Regularization can prevent over-fitting
            self.loss += _CreateL2Reg(self.W, self.B) * self.reg
        self.optmzr = _GetOptimizer(self.optmzrn, self.lr).minimize(self.loss)

class MLPR(ANNR):
    '''
    Multi-Layer Perceptron for Regression
    '''
    
    def __init__(self, layers, actvFn = 'tanh', batchSize = None, learnRate = 1e-3, loss = 'l2', maxIter = 1024, 
                 name = 'tfann', optmzrn = 'adam',  reg = 1e-3, tol = 1e-2, verbose = False):
        '''
        layers: A list of layer sizes
        '''
        self.layers = layers
        super().__init__(actvFn, batchSize, learnRate, loss, maxIter, name, optmzrn, reg, tol, verbose)

    def CreateModel(self):
        self.X = tf.placeholder("float", [None, self.layers[0]])            #Input data matrix
        self.Y = tf.placeholder("float", [None, self.layers[-1]])           #Target value matrix
        self.W, self.B = _CreateVars(self.layers)                           #Setup the weight and bias variables
        self.YH = _CreateMLP(self.X, self.W, self.B, self.AF)               #Create the tensorflow MLP model; YH is graph output
        self.loss = tf.reduce_mean(self.LF(self.YH, self.Y))                
        if self.reg is not None:                                            #Regularization can prevent over-fitting
            self.loss += _CreateL2Reg(self.W, self.B) * self.reg
        self.optmzr = _GetOptimizer(self.optmzrn, self.lr).minimize(self.loss)  #Get optimizer method to minimize the loss function    
   
class MLPB(MLPR):
    '''
    Multi-Layer Perceptron for Binary Data
    '''
    
    def fit(self, A, Y):
        '''
        Fit the MLP to the data
        A: numpy matrix where each row is a sample
        y: numpy matrix of target values
        '''
        A = self.Tx(A)          #Transform data for better performance
        Y = self.Tx(Y)          #with tanh activation
        super().fit(A, Y)
        
    def predict(self, A):
        '''
        Predict the output given the input (only run after calling fit)
        A: The input values for which to predict outputs
        return: The predicted output values (one row per input sample)
        '''
        A = self.Tx(A)
        YH = super().predict(A)
        #Transform back to un-scaled data
        return self.YHatF(self.TInvX(YH))
    
class RFSMLPB(ANNR):
    '''
    Rectangular Fully-Supervised Multi-Layer Perceptron for Binary Data (experimental)
    '''
    
    def __init__(self, layers, actvFn = 'tanh', batchSize = None, learnRate = 1e-3, loss = 'l2', maxIter = 1024, 
                 name = 'tfann', optmzrn = 'adam',  reg = 1e-3, tol = 1e-2, verbose = False):
        '''
        layers: A list of layer sizes
        '''
        self.layers = layers
        super().__init__(actvFn, batchSize, learnRate, loss, maxIter, name, optmzrn, reg, tol, verbose)

    def CreateModel(self):
        ls = self.layers[0]
        nl = len(self.layers)
        self.X = tf.placeholder("float", [None, ls])                        #Input data matrix
        self.Y = tf.placeholder("float", [None, nl, ls])                    #Target value matrix
        self.W, self.B = _CreateVars(self.layers)                           #Setup the weight and bias variables
        self.YH = _CreateRFSMLP(self.X, self.W, self.B, self.AF)            #Create the tensorflow MLP model; YH is graph output
        self.loss = tf.reduce_mean(self.LF(self.YH, self.Y))                
        if self.reg is not None:                                            #Regularization can prevent over-fitting
            self.loss += _CreateL2Reg(self.W, self.B) * self.reg
        self.optmzr = _GetOptimizer(self.optmzrn, self.lr).minimize(self.loss)  #Get optimizer method to minimize the loss function
         
    def fit(self, A, Y):
        '''
        Fit the MLP to the data
        A: numpy matrix where each row is a sample
        y: numpy matrix of target values
        '''
        A = self.Tx(A)          #Transform data for better performance
        Y = self.Tx(Y)          #with tanh activation
        super().fit(A, Y)
        
    def predict(self, A):
        '''
        Predict the output given the input (only run after calling fit)
        A: The input values for which to predict outputs
        return: The predicted output values (one row per input sample)
        '''
        A = self.Tx(A)
        YH = super().predict(A)
        #Last layer is actual output; hidden layers are used for training
        return self.YHatF(self.TInvX(YH[:, -1]))
    
    def score(self, A, Y):
        '''
        Compute a loss score given a matrix of samples and a target matrix Y
        '''
        YH = self.predict(A)
        return np.linalg.norm(Y[:, -1] - YH)

class ANNC(ANN):
    '''
    Artificial Neural Network for Classification (Base Class)
    '''
    
    def ClearClasses(self):
        '''
        Clears the class lookup for encoding 1-hot vectors so it will
        be rebuilt on the next call to RestoreClasses
        '''
        self._classes = None

    def fit(self, A, Y):
        '''
        Fit the MLP to the data
        A: numpy matrix where each row is a sample
        y: numpy matrix of target values
        '''
        Y = self.To1Hot(Y)
        super().fit(A, Y)

    def __init__(self, actvFn = 'tanh', batchSize = None, learnRate = 1e-3, loss = 'smce', maxIter = 1024, 
                 name = 'tfann', optmzrn = 'adam', reg = 1e-3, tol = 1e-2, verbose = False):
        #Lookup table for class labels
        self._classes = None
        super().__init__(actvFn, batchSize, learnRate, loss, maxIter, name, optmzrn, reg, tol, verbose)

    def predict(self, A):
        '''
        Predict the output given the input (only run after calling fit)
        A: The input values for which to predict outputs
        return: The predicted output values (one row per input sample)
        '''
        #Return prediction labels recovered from 1-hot encoding
        return self._classes[np.argmax(super().predict(A), 1).reshape(-1)]

    def RestoreClasses(self, Y):
        '''
        Restores the classes lookup table
        Y:  An array that contains all the class labels
        '''
        if self._classes is None:
            self._classes = np.unique(Y)

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
        self._classes, inv = np.unique(Y, return_inverse = True)
        b = np.zeros([len(Y), len(self._classes)])
        b[np.arange(b.shape[0]), inv] = 1
        return b

class CNNC(ANNC):
    '''
    Convolutional Neural Network for Classification
    '''

    def __init__(self, imageSize, ws, actvFn = 'relu', batchSize = None, learnRate = 1e-4, loss = 'smce', name = 'tfann',
                 maxIter = 1024, optmzrn = 'adam', pad = 'SAME', tol = 1e-1, reg = None, verbose = False):
        '''
        imageSize:  Size of the images used (Height, Width, Depth)
        ws:         Weight matrix sizes
        '''
        #Initialize fields from base class
        self.imgSize = list(imageSize)
        self.ws = ws
        self.pad = pad
        super().__init__(actvFn, batchSize, learnRate, loss, maxIter, name, optmzrn, reg, tol, verbose)
        
    def CreateModel(self):
        self.X = tf.placeholder("float", [None] + self.imgSize)         #Input data matrix of samples                                                 #Padding method to use
        self.Y = tf.placeholder("float", [None, self.ws[-1][1]])        #Target matrix
        self.YH = self._CreateCNN(self.ws)                              #Create graph; YH is output matrix
        self.loss = tf.reduce_mean(self.LF(self.YH, self.Y))            #Loss term
        if self.reg is not None:                                        #Regularization can prevent over-fitting
            self.loss += _CreateL2Reg(self.W, self.B) * self.reg
        self.optmzr = _GetOptimizer(self.optmzrn, self.lr).minimize(self.loss)
        
class MLPC(ANNC):
    '''
    Multi-Layer Perceptron for Classification
    '''
    
    def __init__(self, layers, actvFn = 'tanh', batchSize = None, learnRate = 1e-3, loss = 'smce', maxIter = 1024,
                 name = 'tfann', optmzrn = 'adam', reg = 1e-3, tol = 1e-2, verbose = False):
        '''
        layers: A list of layer sizes
        '''
        self.layers = layers
        super().__init__(actvFn, batchSize, learnRate, loss, maxIter, name, optmzrn, reg, tol, verbose)
        
    def CreateModel(self):
        self.X = tf.placeholder("float", [None, self.layers[0]])            #Input data matrix
        self.Y = tf.placeholder("float", [None, self.layers[-1]])           #Target matrix
        self.W, self.B = _CreateVars(self.layers)                           #Setup the weight and bias variables
        self.YH = _CreateMLP(self.X, self.W, self.B, self.AF)               #Create the tensorflow model; YH is output matrix
        #Cross entropy loss function
        self.loss = tf.reduce_mean(self.LF(self.YH, self.Y))
        if self.reg is not None:                                            #Regularization can prevent over-fitting
            self.loss += _CreateL2Reg(self.W, self.B) * self.reg
        self.optmzr = _GetOptimizer(self.optmzrn, self.lr).minimize(self.loss)
        
class MLPMC(ANNC):
    '''
    Multi-Layer Perceptron for Multi-Classification
    Uses n classifiers to solve a multi-classification problem with n output columns.
    '''
    
    def __init__(self, layers, cl, actvFn = 'tanh', batchSize = None, learnRate = 1e-3, loss = 'smce', 
                 maxIter = 1024, name = 'tfann', optmzrn = 'adam', reg = None, tol = 1e-1, verbose = False):
        '''
        layers:     Only specify the input sizes and hidden layers sizes. Output layers are
                    automatically inferred from the cl parameter
        cl:         The lists of possible classes for each column (from left to right)    
        '''
        self.cl = cl                #List of list of possible classes
        self.lo = len(cl)           #Total length of output vectors
        self.layers = layers
        super().__init__(actvFn, batchSize, learnRate, loss, maxIter, name, optmzrn, reg, tol, verbose)

    def CreateModel(self):
        self.X = tf.placeholder("float", [None, self.layers[0]])                #Input data matrix
        self.optmzrs, self.losses, self.Ws, self.Bs, self.Ys, self.YHs = [], [], [], [], [], []
        for i in range(self.lo):
            Y = tf.placeholder("float", [None, len(self.cl[i])])                #Target matrix is 1-hot encoding
            self.Ys.append(Y)                                                   #of classes for column i
            W, B = _CreateVars(self.layers + [len(self.cl[i])])                 #Add last layer for 1-hot encoding prediction
            self.Ws.append(W)
            self.Bs.append(B)
            YH = _CreateMLP(self.X, W, B, self.AF)                              #Create the tensorflow model; YH is output matrix
            self.YHs.append(YH)
            loss = tf.reduce_mean(self.LF(YH, Y))                               #Reduce mean of the loss function
            if self.reg is not None:                                            #Regularization can prevent over-fitting
                loss += _CreateL2Reg(W, B) * self.reg
            self.losses.append(loss)
            self.optmzrs.append(_GetOptimizer(self.optmzrn, self.lr).minimize(loss))
 
    def fit(self, A, Y):
        for i in range(self.lo):                #Loop over all columns in output
            self.loss = self.losses[i]          #Choose loss for model predicting i-th feature
            self.optmzr = self.optmzrs[i]
            self.Y = self.Ys[i]                 #Choose appropriate Y placeholder
            self.ClearClasses()                 #Setup lookup to encoding 1-hot vectors
            self.RestoreClasses(self.cl[i])     #using classes for column i
            super().fit(A, Y[:, i])             #Fit model for i-th column
            
    def predict(self, A):
        P = []
        for i in range(self.lo):                #Loop over each column in output
            self.YH = self.YHs[i]               #Choose feedforward variable for this column
            P.append(super().predict(A))        #Predict classes for i-th column
        return np.column_stack(P)               #Stack columns together
    
    def score(self, A, Y):
        '''
        The score is # correct / total across all rows and column
        '''
        YH = self.predict(A)
        return (YH == Y).sum() / np.prod(Y.shape)