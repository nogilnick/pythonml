# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 18:43:21 2016

@author: nicholas smith
"""

import tensorflow as tf
import numpy as np

#Return the classification accuracy
#given a vector of target labels and
#predicted labels
#y: The target labels
#yHat: The predicted labels
#return: The percentage correct
def _Accuracy(y, yHat):
    n = len(y)
    return np.sum(y == yHat) / n

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
    return None

#Gives the next batch of samples of size self.batSz or the remaining
#samples if there are not that many
#A: Samples to choose from
#y: Targets to choose from
#cur: The next sample to use
#batSz: Size of the batch
#return: A tuple of the new samples and targets
def _NextBatch(A, y, cur, batSz):
    m = len(A)
    nxt = cur + batSz
    if(nxt > m):
        nxt = m
    return (A[cur:nxt], y[cur:nxt])

#Multi-Layer Perceptron for Classification
class MLPC:

    #Predicted outputs
    pred = None
    #The loss function
    loss = None
    #The optimization method
    optmzr = None
    #Max number of iterations
    mItr = None
    #Error tolerance
    tol = None
    #Tensorflow session
    sess = None
    #Input placeholder
    x = None
    #Output placeholder
    y = None
    #Boolean for toggling verbose output
    vrbse = None
    #Batch size
    batSz = None
    #The class labels
    _classes = None

    def __init__(self, layers, actvFn = 'tanh', learnRate = 0.001, decay = 0.9, maxItr = 2000,
                 tol = 1e-2, batchSize = None, verbose = False, reg = 0.001):
        #Parameters
        self.tol = tol
        self.mItr = maxItr
        self.n = len(layers)
        self.vrbse = verbose
        self.batSz = batchSize
        #Input size
        self.x = tf.placeholder("float", [None, layers[0]])
        #Output size
        self.y = tf.placeholder("float", [None, layers[-1]])
        #Setup the weight and bias variables
        weight, bias = _CreateVars(layers)   
        #Create the tensorflow model
        self.pred = _CreateMLP(self.x, weight, bias, _GetActvFn(actvFn))
        #Cross entropy loss function
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y))
        #Use regularization to prevent over-fitting
        if(reg is not None):
            self.loss += _CreateL2Reg(weight, bias) * reg
        self.optmzr = tf.train.AdamOptimizer(learning_rate=learnRate).minimize(self.loss)
        
    #Fit the MLP to the data
    #param A: numpy matrix where each row is a sample
    #param y: numpy matrix of target values
    def fit(self, A, y):
        m = len(A)
        y = self.to1Hot(y)
        #Start the tensorflow session and initializer
        #all variables
        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)
        #Begin training
        for i in range(self.mItr):
            #Batch mode or all at once
            if(self.batSz is None):
                self.sess.run(self.optmzr, feed_dict={self.x:A, self.y:y})
            else:
                for j in range(0, m, self.batSz):
                    batA, batY = _NextBatch(A, y, j, self.batSz)
                    self.sess.run(self.optmzr, feed_dict={self.x:batA, self.y:batY})
            err = np.sqrt(np.sum(self.sess.run(self.loss, feed_dict={self.x:A, self.y:y})) / m)
            if(self.vrbse):
                print("Iter " + str(i + 1) + ": " + str(err))
            if(err < self.tol):
                break
    
    #Predict the output given the input (only run after calling fit)
    #param A: The input values for which to predict outputs 
    #return: The predicted output values (one row per input sample)
    def predict(self, A):
        if(self.sess == None):
            print("Error: MLPC has not yet been fitted.")
            return None
        #Get the predicted indices
        res = np.argmax(self.sess.run(self.pred, feed_dict={self.x:A}), 1)
        res.shape = [-1]
        #Return prediction using the original labels
        return np.array([self._classes[i] for i in res])
        
    #Predicts the ouputs for input A and then computes the classification error
    #The predicted values and the actualy values
    #param A: The input values for which to predict outputs 
    #param y: The actual target values
    #return: The percent of outputs predicted correctly
    def score(self, A, y):
        yHat = self.predict(A)
        return _Accuracy(y, yHat)

    #Creates an array of 1-hot vectors
    #based on a vector of class labels
    #y: The vector of class labels
    #return: The 1-Hot encoding of y
    def to1Hot(self, y):
        lbls = list(set(list(y)))
        lbls.sort()
        lblDic = {}
        self._classes = []
        for i in range(len(lbls)):
            lblDic[lbls[i]] = i
            self._classes.append(lbls[i])
        b = np.zeros([len(y), len(lbls)])
        for i in range(len(y)):
            b[i, lblDic[y[i]]] = 1
        return b
        
                
    #Clean up resources
    def __del__(self):
        self.sess.close()
    
    #Create the MLP variables for TF graph
    #_X: The input matrix
    #_W: The weight matrices
    #_B: The bias vectors
    #_AF: The activation function
    def _createMLPC(self, _X, _W, _B, _AF):
        n = len(_W)
        for i in range(n - 1):
            _X = _AF(tf.matmul(_X, _W[i]) + _B[i]) 
        return tf.matmul(_X, _W[n - 1]) + _B[n - 1]
        
#Multi-Layer Perceptron for Regression
class MLPR:
    #Predicted outputs
    pred = None
    #The loss function
    loss = None
    #The optimization method
    optmzr = None
    #Max number of iterations
    mItr = None
    #Error tolerance
    tol = None
    #Tensorflow session
    sess = None
    #Input placeholder
    x = None
    #Output placeholder
    y = None
    #Boolean for toggling verbose output
    vrbse = None
    #Batch size
    batSz = None

    def __init__(self, layers, actvFn = 'tanh', learnRate = 0.001, decay = 0.9, maxItr = 2000,
                 tol = 1e-2, batchSize = None, verbose = False, reg = 0.001):
        #Parameters
        self.tol = tol
        self.mItr = maxItr
        self.vrbse = verbose
        self.batSz = batchSize
        #Input size
        self.x = tf.placeholder("float", [None, layers[0]])
        #Output size
        self.y = tf.placeholder("float", [None, layers[-1]])
        #Setup the weight and bias variables
        weight, bias = _CreateVars(layers)       
        #Create the tensorflow MLP model
        self.pred = _CreateMLP(self.x, weight, bias, _GetActvFn(actvFn))
        #Use L2 as the cost function
        self.loss = tf.reduce_sum(tf.nn.l2_loss(self.pred - self.y))
        #Use regularization to prevent over-fitting
        if(reg is not None):
            self.loss += _CreateL2Reg(weight, bias) * reg
        #Use ADAM method to minimize the loss function
        self.optmzr = tf.train.AdamOptimizer(learning_rate=learnRate).minimize(self.loss)
        
    #Fit the MLP to the data
    #param A: numpy matrix where each row is a sample
    #param y: numpy matrix of target values
    def fit(self, A, y):
        m = len(A)
        #Start the tensorflow session and initializer
        #all variables
        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)
        #Begin training
        for i in range(self.mItr):
            #Batch mode or all at once
            if(self.batSz is None):
                self.sess.run(self.optmzr, feed_dict={self.x:A, self.y:y})
            else:
                for j in range(0, m, self.batSz):
                    batA, batY = _NextBatch(A, y, j, self.batSz)
                    self.sess.run(self.optmzr, feed_dict={self.x:batA, self.y:batY})
            err = np.sqrt(self.sess.run(self.loss, feed_dict={self.x:A, self.y:y}) * 2.0 / m)
            if(self.vrbse):
                print("Iter " + str(i + 1) + ": " + str(err))
            if(err < self.tol):
                break
    
    #Predict the output given the input (only run after calling fit)
    #param A: The input values for which to predict outputs 
    #return: The predicted output values (one row per input sample)
    def predict(self, A):
        if(self.sess == None):
            print("Error: MLP has not yet been fitted.")
            return None
        res = self.sess.run(self.pred, feed_dict={self.x:A})
        return res
        
    #Predicts the ouputs for input A and then computes the RMSE between
    #The predicted values and the actualy values
    #param A: The input values for which to predict outputs 
    #param y: The actual target values
    #return: The RMSE
    def score(self, A, y):
        scr = np.sqrt(self.sess.run(self.loss, feed_dict={self.x:A, self.y:y}) * 2.0 / len(A))
        return scr
        
    #Clean-up resources
    def __del__(self):
        self.sess.close()
