'''
TFPlot.py
This file functions to plot performance of a
neural network during training.
'''
import numpy as np
import math
import matplotlib.pyplot as mpl
import matplotlib.animation as mpla
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import scale
from TFANN import MLPB
from StockPredictor import ParseData

#Converts a binary vector (left to right format) to an integer
#x:            A binary vector
#return:    The corresponding integer
def BinVecToInt(x):
    #Accumulator variable
    num = 0
    #Place multiplier
    mult = 1
    for i in x:
        #Cast is needed to prevent conversion to floating point
        num = num + int(i) * mult
        #Multiply by 2
        mult <<= 1
    return num

#Converts an integer to 
def IntToBinVec(x, v = None):
    #If no vector is passed create a new one
    if(v is None):
        dim = int(np.log2(x)) + 1
        v = np.zeros([dim], dtype = np.int)
    #v will contain the binary vector
    c = 0
    while(x > 0):
        #If the vector has been filled; return truncating the rest
        if c >= len(v):
            break
        #Test if the LSB is set
        if(x & 1 == 1):
            #Set the bits in right-to-left order
            v[c] = 1
        #Onto the next column and bit
        c += 1
        x >>= 1
    return v
     
#Plot the model R learning the data set A, Y
#R: A regression model
#A: The data samples
#Y: The target vectors
def PlotLearn(R, A, Y):
    intA = [BinVecToInt(j) for j in A]
    intY = [BinVecToInt(j) for j in Y]
    fig, ax = mpl.subplots(figsize=(20, 10))
    ax.plot(intA, intY, label ='Orig')
    l, = ax.plot(intA, intY, label ='Pred')
    ax.legend(loc = 'upper left')
    #Updates the plot in ax as model learns data
    def UpdateF(i):
        R.fit(A, Y)
        YH = R.predict(A)
        S = MSE(Y, YH)
        intYH = [BinVecToInt(j) for j in YH]
        l.set_ydata(intYH)
        ax.set_title('Iteration: ' + str(i * 64) + ' - MSE: ' + str(S))
        return l,

    ani = mpla.FuncAnimation(fig, UpdateF, frames = 2000, interval = 128, repeat = False)
    #ani.save('foo.gif')
    mpl.show()
    return ani

if __name__ == "__main__":
    #Data in data.csv is assumed to contain 'High' and 'Timestamp' columns; 
    #code below will scale it to unsigned ints
    #In a true example, data in data.csv would be high dimensional binary vectors
    D = ParseData('data.csv')[['Timestamp', 'High']]
    D['Timestamp'] = D['Timestamp'] - np.min(D['Timestamp'])
    D = np.floor(scale(D) * 16000)
    D = D - np.min(D, axis = 0)
    aInt = np.int64(D[:, 0])
    yInt = np.int64(D[:, 1])
    #Max number of bits needed to represent vectors in A
    aMaxLen = math.floor(math.log2(np.max(aInt))) + 1
    #Max number of bits needed to represent vectors in Y
    yMaxLen = math.floor(math.log2(np.max(yInt))) + 1
    n = aInt.shape[0]
    A = np.zeros([n, aMaxLen], dtype=np.int)
    Y = np.zeros([n, yMaxLen], dtype=np.int)
    for i in range(n):
        IntToBinVec(yInt[i], Y[i])
        IntToBinVec(aInt[i], A[i])
    nIn = A.shape[1]
    nOut = Y.shape[1]
    nHid = (nOut + nIn) // 2
    l = [('F', nHid), ('AF', 'tanh')] * 4 + [('F', nOut)]
    mlpr = MLPB([nIn], l, batchSize = 32, maxIter = 64, tol = 0.0001, learnRate = 25e-5, verbose = True, reg = 1e-7)
    
    a = PlotLearn(mlpr, A, Y)