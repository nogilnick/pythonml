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
    
def _CreateANN(PM, NA, X):
    '''
    Sets up the graph for a convolutional neural network from 
    a list of operation specifications like:
    
    [('C', [5, 5, 3, 64], [1, 1, 1, 1]), ('AF', 'tanh'), 
     ('P', [1, 3, 3, 1], [1, 2, 2, 1]), ('F', 10)]
    
    Operation Types:
    
        AF:     ('AF', <name>)  Activation Function 'relu', 'tanh', etc
        C:      ('C', [Filter Shape], [Stride])     2d Convolution
        CT:     2d Convolution Transpose
        C1d:    ('C1d', [Filter Shape], Stride)     1d Convolution
        C3d:    ('C3d', [Filter Shape], [Stride])   3d Convolution
        D:      ('D', Probability)                  Dropout Layer
        F:      ('F', Output Neurons)               Fully-connected
        LRN:    ('LRN')                             
        M:      ('M', Dims)                         Average over Dims
        P:      ('P', [Filter Shape], [Stride])     Max Pool
        P1d:    ('P1d', [Filter Shape], Stride)     1d Max Pooling
        R:      ('R', shape)                        Reshape
        S:      ('S', Dims)                         Sum over Dims
    
    [Filter Shape]: (Height, Width, In Channels, Out Channels)
    [Stride]:       (Batch, Height, Width, Channel)
    Stride:         1-D Stride (single value)
    
    PM:     The padding method
    NA:     The network architecture
    X:      Input tensor
    '''
    WI = tf.contrib.layers.xavier_initializer() #Weight initializer
    W, B, O = [], [], [X]
    for i, NAi in enumerate(NA):
        if NAi[0] == 'AF':      #Apply an activation function ('AF', 'name')
            AF = _GetActvFn(NAi[1])
            X = AF(X)
        elif NAi[0] == 'C':     #2-D Convolutional layer
            W.append(tf.Variable(WI(NAi[1])))
            B.append(tf.Variable(tf.constant(0.0, shape = [NAi[1][-1]])))
            X = tf.nn.conv2d(X, W[-1], NAi[2], padding = PM)
            X = tf.nn.bias_add(X, B[-1])
        elif NAi[0] == 'CT':    #2-D Convolutional Transpose layer
            W.append(tf.Variable(WI(NAi[1])))
            B.append(tf.Variable(tf.constant(0.0, shape = [NAi[1][-2]])))
            X = tf.nn.conv2d_transpose(X, W[-1], NAi[2], NAi[3], padding = PM)
            X = tf.nn.bias_add(X, B[-1])
        elif NAi[0] == 'C1d':   #1-D Convolutional Layer
            W.append(tf.Variable(WI(NAi[1])))
            B.append(tf.Variable(tf.constant(0.0, shape = [NAi[1][-1]])))
            X = tf.nn.conv1d(X, W[-1], NAi[2], padding = PM)
            X = tf.nn.bias_add(X, B[-1])
        elif NAi[0] == 'C3d':
            W.append(tf.Variable(WI(NAi[1])))
            B.append(tf.Variable(tf.constant(0.0, shape = [NAi[1][-1]])))
            X = tf.nn.conv1d(X, W[-1], NAi[2], padding = PM)
            X = tf.nn.bias_add(X, B[-1])
        elif NAi[0] == 'D':     #Dropout layer ('D', <probability>)
            X = tf.nn.dropout(X, NAi[1])
        elif NAi[0] == 'F':     #Fully-connected layer
            if tf.rank(X) != 2:
                X = tf.contrib.layers.flatten(X)
            W.append(tf.Variable(WI([X.shape[1].value, NAi[1]])))
            B.append(tf.Variable(WI([NAi[1]])))
            X = tf.matmul(X, W[-1]) + B[-1]
        elif NAi[0] == 'LRN':
            X = tf.nn.lrn(X, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
        elif NAi[0] == 'M':
            X = tf.reduce_mean(X, NAi[1])
        elif NAi[0] == 'P':     #Pooling layer
            X = tf.nn.max_pool(X, ksize = NAi[1], strides = NAi[2], padding = PM)
        elif NAi[0] == 'P1d':   #1-D Pooling layer
            X = tf.nn.pool(X, [NAi[1]], 'MAX', PM, strides = [NAi[2]])
        elif NAi[0] == 'R':     #Reshape layer
            X = tf.reshape(X, NAi[1])
        elif NAi[0] == 'S':
            X = tf.reduce_sum(X, NAi[1])
        O.append(X)
    return O, W, B

def _CreateL2Reg(_W, _B):
    '''
    Add L2 regularizers for the weight and bias matrices
    _W: The weight matrices
    _B: The bias matrices
    return: tensorflow variable representing l2 regularization cost
    '''
    return sum(tf.nn.l2_loss(_Wi) + tf.nn.l2_loss(_Bi) for _Wi, _Bi in zip(_W, _B))

def _GetActvFn(name):
    '''
    Helper function for selecting an activation function
    name: The name of the activation function
    return: A handle for the tensorflow activation function
    '''
    return {'atanh': tf.atanh,          'elu': tf.nn.elu,
            'ident': tf.identity,
            'sig': tf.sigmoid,          'softplus': tf.nn.softplus, 
            'softsign': tf.nn.softsign, 'relu': tf.nn.relu,
            'relu6': tf.nn.relu6,       'tanh': tf.tanh}.get(name)
    
def _GetBatchRange(bs, mIter):
    '''
    Gives a range from which to choose the batch size
    '''
    try:
        return np.linspace(*bs[0:2], num = mIter).round().astype(np.int)
    except TypeError:
        pass
    return np.repeat(bs, mIter)
    
def _GetLossFn(name):
    '''
    Helper function for selecting loss function
    name:   The name of the loss function
    return:     A handle for a loss function LF(YH, Y)
    '''
    return {'cos': lambda YH, Y : tf.losses.cosine_distance(Y, YH), 'hinge': lambda YH, Y : tf.losses.hinge_loss(Y, YH),
            'l1': lambda YH, Y : tf.losses.absolute_difference(Y, YH), 'l2': lambda YH, Y : tf.squared_difference(Y, YH),
            'log': lambda YH, Y : tf.losses.log_loss(Y, YH), 
            'sgce': lambda YH, Y : tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = YH), 
            'smce': lambda YH, Y : tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = YH)}.get(name)

def _GetOptimizer(name, lr):
    '''
    Helper function for getting a tensorflow optimizer
    name:   The name of the optimizer to use
    lr:   The learning rate if applicable
    return;  A the tensorflow optimization object
    '''
    return {'adam': tf.train.AdamOptimizer(learning_rate = lr),
            'adagrad': tf.train.AdagradOptimizer(learning_rate = lr),
            'ftrl': tf.train.FtrlOptimizer(learning_rate = lr),
            'grad': tf.train.GradientDescentOptimizer(learning_rate = lr)}.get(name)

class ANN:
    sess = None     #Class variable shared among all instances
    sessc = 0       #Number of active sessions
    '''
    TensorFlow Artificial Neural Network (Base Class)
    '''

    #Clean-up resources
    def __del__(self):
        ANN.sessc -= 1          #Decrement session counter
        if ANN.sessc == 0:      #Close session only if not in use
            self.GetSes().close()
            ANN.sess = None

    def __init__(self, _IS, _NA, batchSize = None, learnRate = 1e-4, loss = 'l2', maxIter = 1024,
                 name = 'tfann', optmzrn = 'adam', pad = 'SAME', reg = None, tol = 1e-2, verbose = False, X = None, Y = None):
        '''
        Common arguments for all artificial neural network regression models
        _IS:        Shape of a single input sample
        _NA:        Network architecture (see _CreateANN)
        batchSize:  Size of training batches to use (use all if None)
        learnRate:  The learning rate parameter for the optimizer
        loss:       The name of the loss function (l2, l1, smce, sgme, cos, log, hinge) 
        name:       The name of the model for variable scope, saving, and restoring
        maxIter:    Maximum number of training iterations
        optmzrn:    The optimizer method to use ('adam', 'grad', 'adagrad', or 'ftrl')
        reg:        Weight of the regularization term (None for no regularization)
        tol:        Training ends if error falls below this tolerance
        verbose:    Print training information
        X:          Tensor that is used as input to model; if None a new
                    placeholder is created and used. If specified a feed_dict
                    must be passed to train, predict, and score calls!
        '''
        self.IS = list(_IS)                         #Input shape
        self.NA = _NA                               #Network architecture specification
        self.LF = _GetLossFn(loss)                  #Handle to loss function to use
        self.batSz = batchSize                      #Batch size
        self.init = None                            #Initializer operation
        self.lr = learnRate                         #Learning rate
        self.mIter = maxIter                        #Maximum number of iterations
        self.name = name                            #Name of model for variable scope
        self.optmzrn = optmzrn                      #Optimizer method name
        self.pad = pad                              #Default padding method
        self.reg = reg                              #Regularization strength
        self.stopIter = False                       #Flag for stopping training early
        self.tol = tol                              #Error tolerance
        self.vrbse = verbose                        #Verbose output
        self.X = X                                  #TF Variable used as input to model
        self.Y = Y                                  #TF Variable used as target for model
        #Data members to be populated in a subclass
        self.loss = None
        self.optmzr = None
        self.O = None
        self.TFVar = None
        with tf.variable_scope(self.name):
            self.TFVar = set(tf.global_variables())                 #All variables before model
            self.CreateModel()                                      #Populate TensorFlow graph
            self.TFVar = set(tf.global_variables()) - self.TFVar    #Only variables in this model
            self.Initialize()                                       #Start TF session and initialize vars 
            
    def CreateModel(self):
        '''
        Builds TF graph for ANN model
        '''
        if self.X is None:  #If no input variable was specified make a new placeholder
            self.X = tf.placeholder("float", [None] + self.IS)   #Input data matrix of samples
        self.O, self.W, self.B = _CreateANN(self.pad, self.NA, self.X)#Create graph; O is output from feedforward
        if self.Y is None:
            self.Y = tf.placeholder("float", self.O[-1].shape)     #Target matrix
        self.loss = tf.reduce_mean(self.LF(self.O[-1], self.Y))    #Loss term
        if self.reg is not None:                                #Regularization can prevent over-fitting
            self.loss += _CreateL2Reg(self.W, self.B) * self.reg
        self.optmzr = _GetOptimizer(self.optmzrn, self.lr).minimize(self.loss)
        
    def fit(self, A, Y, FD = None):
        '''
        Fit the ANN to the data
        A: numpy matrix where each row is a sample
        Y: numpy matrix of target values
        '''
        m = len(A)
        FD = {self.X:A, self.Y:Y} if FD is None else FD     #Feed dictionary
        #Loop up to mIter times gradually scaling up the batch size (if range is provided)
        for i, BSi in enumerate(_GetBatchRange(self.batSz, self.mIter)):
            if BSi is None:                          #Compute loss and optimize simultaneously for all samples
                err, _ = self.GetSes().run([self.loss, self.optmzr], feed_dict = FD)
            else:                                           #Train m samples using random batches of size self.bs
                err = 0.0
                for j in range(0, m, BSi):                  #Compute loss and optimize simultaneously for batch
                    bi = np.random.choice(m, BSi, False)    #Randomly chosen batch indices
                    BFD = {k:v[bi] for k, v in FD.items()}  #Feed dictionary for this batch
                    l, _ = self.GetSes().run([self.loss, self.optmzr], feed_dict = BFD)
                    err += l                                #Accumulate loss over all batches
                err /= len(range(0, m, BSi))                #Average over all batches
            if self.vrbse:
                print("Iter {:5d}\t{:16.8f} (Batch Size: {:5d})".format(i + 1, err, -1 if BSi is None else BSi))
            if err < self.tol or self.stopIter:
                break   #Stop if tolerance was reached or flag was set
                
    def GetBias(self, i):
        '''
        Warning! Does not work with restored models currently.
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
        Warning! Does not work with restored models currently.
        Gets the i-th weight matrix from the model.
        '''
        return self.W[i].eval(session = self.GetSes())

    def Initialize(self):
        '''
        Initialize variables and start the TensorFlow session
        '''
        ANN.sessc += 1                              #Increment session counter
        sess = self.GetSes()
        self.init = tf.variables_initializer(self.TFVar)
        sess.run(self.init)                         #Initialize all variables for this model
        
    def predict(self, A, FD = None):
        '''
        Predict the output given the input (only run after calling fit)
        A: The input values for which to predict outputs
        return: The predicted output values (one row per input sample)
        '''
        FD = {self.X:A} if FD is None else FD
        return self.GetSes().run(self.O[-1], feed_dict = FD)
        
    def PredictFull(self, A, FD = None):
        '''
        Predict the output given the input (only run after calling fit)
        returning all intermediate results
        A: The input values for which to predict outputs
        return: The predicted output values with intermediate results
        (one row per input sample)
        '''
        FD = {self.X:A} if FD is None else FD
        return self.GetSes().run(self.O, feed_dict = FD)

    def Reinitialize(self):
        '''
        Reinitialize variables and start the TensorFlow session
        '''
        sess = self.GetSes()
        sess.run(self.init) #Re-Initialize all variables for this model
        
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
            saver = tf.train.Saver()
            saver.restore(self.GetSes(), p + n)
        except Exception as e:
            print('Error restoring: ' + p + n)
            return False
        return True

    def SaveModel(self, p):
        '''
        Save model to file
        p:  Path to save file like "path/"
        '''
        saver = tf.train.Saver()
        saver.save(self.GetSes(), p)
        
    def score(self, A, Y, FD = None):
        '''
        Predicts the loss function for given input and target matrices
        A: The input data matrix
        Y: The target matrix
        return: The loss
        '''
        FD = {self.X:A, self.Y:Y} if FD is None else FD
        return self.GetSes().run(self.loss, feed_dict = FD)
        
    def SetBias(self, i, _B):
        '''
        Warning! Does not work with restored models currently.
        Sets the i-th bias from the model.
        '''
        self.B[i].assign(_B).eval(session = self.GetSes())

    def SetWeightMatrix(self, i, _W):
        '''
        Warning! Does not work with restored models currently.
        Sets the i-th weight matrix from the model.
        '''
        self.W[i].assign(_W).eval(session = self.GetSes())
        
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
    pass
   
class MLPB(ANNR):
    '''
    Multi-Layer Perceptron for Binary Data
    '''
    
    def fit(self, A, Y, FD = None):
        '''
        Fit the MLP to the data
        A: numpy matrix where each row is a sample
        y: numpy matrix of target values
        '''
        A = self.Tx(A)          #Transform data for better performance
        Y = self.Tx(Y)          #with tanh activation
        super().fit(A, Y, FD)
        
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
    
class RFSANNR(ANNR):
    '''
    Rectangular Fully-Supervised Artificial Neural Network for Regression
    Requires targets for hidden layers.
    '''

    def CreateModel(self):
        if self.X is None:  #If no input variable was specified make a new placeholder
            self.X = tf.placeholder("float", [None, self.IS])       #Input data matrix
        self.O, self.W, self.B = _CreateANN(self.pad, self.NA, self.X)
        self.RO = tf.stack(self.O, axis = 1)
        if self.Y is None:
            self.Y = tf.placeholder("float", self.O.shape)          #Target value matrix
        self.loss = tf.reduce_mean(self.LF(self.RO, self.Y))                
        if self.reg is not None:                                    #Regularization can prevent over-fitting
            self.loss += _CreateL2Reg(self.W, self.B) * self.reg
        self.optmzr = _GetOptimizer(self.optmzrn, self.lr).minimize(self.loss)  #Get optimizer method to minimize the loss function

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

    def fit(self, A, Y, FD = None):
        '''
        Fit the MLP to the data
        A: numpy matrix where each row is a sample
        y: numpy matrix of target values
        '''
        Y = self.To1Hot(Y)
        super().fit(A, Y, FD)

    def __init__(self, _IS, _NA, batchSize = None, learnRate = 1e-3, loss = 'smce', maxIter = 1024, 
                 name = 'tfann', optmzrn = 'adam', pad = 'SAME', reg = 1e-3, tol = 1e-2, verbose = False, X = None, Y = None):
        super().__init__(_IS, _NA, batchSize, learnRate, loss, maxIter, name, optmzrn, pad, reg, tol, verbose, X, Y)
        a = len(self.O[-1].shape) - 1               #Class axis is last axis
        self.YHL = tf.argmax(self.O[-1], axis = a)  #Index of label with max probability
        self._classes = None                        #Lookup table for class labels

    def predict(self, A, FD = None):
        '''
        Predict the output given the input (only run after calling fit)
        A: The input values for which to predict outputs
        return: The predicted output values (one row per input sample)
        '''
        #Return prediction labels recovered from 1-hot encoding
        FD = {self.X:A} if FD is None else FD
        return self._classes[self.GetSes().run(self.YHL, feed_dict = FD)]

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
        Creates an array of 1-hot vectors based on a vector of class labels.
        The class label axis for the encoding is the last axis.
        Y: The vector of class labels
        #return: The 1-Hot encoding of Y
        '''
        self._classes, inv = np.unique(Y.ravel(), return_inverse = True)
        b = np.zeros([len(inv), len(self._classes)])
        b[np.arange(b.shape[0]), inv] = 1
        return b.reshape(list(Y.shape) + [len(self._classes)])
        
class MLPMC():
    '''
    Multi-Layer Perceptron for Multi-Classification
    Uses n classifiers to solve a multi-classification problem with n output columns.
    '''
    
    def __init__(self, _IS, _NA, batchSize = None, learnRate = 1e-3, loss = 'smce', 
                 maxIter = 1024, name = 'tfann', optmzrn = 'adam', pad = 'SAME', reg = None, tol = 1e-2, verbose = False, X = None, Y = None):
        '''
        layers:     Only specify the input sizes and hidden layers sizes. Output layers are
                    automatically inferred from the cl parameter
        cl:         The lists of possible classes for each column (from left to right)    
        '''
        self.M = [ANNC(_IS, _NAi, batchSize, learnRate, loss, maxIter, name, optmzrn, pad, reg, tol, verbose, X, Y) for _NAi in _NA]
 
    def fit(self, A, Y, FD = None):
        for i, Mi in enumerate(self.M): #Loop over all columns in output
            Mi.fit(A, Y[:, i], FD)      #Fit model for i-th column
            
    def predict(self, A):
        return np.column_stack([Mi.predict(A) for Mi in self.M]) #Stack network predictions together
    
    def score(self, A, Y):
        '''
        The score is # correct / total across all rows and column
        '''
        YH = self.predict(A)
        return (YH == Y).sum() / np.prod(Y.shape)