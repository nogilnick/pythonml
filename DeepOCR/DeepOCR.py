import numpy as np
import argparse
import os
import string
import sys
from skimage.io import imread
from sklearn.model_selection import ShuffleSplit
from TFANN import ANNC
import tensorflow as tf
from scipy.stats import mode as Mode

NC = len(string.ascii_letters + string.digits + ' ')    #Number of possible characters
MAX_CHAR = 64           #Max # characters per block
IS = [18, 640, 3]       #Image size for CNN
NNET = 5                #Use 5 networks with a bagging approach
#Shared placeholders for images and target data
TFIM = tf.placeholder("float", [None] + IS, name = 'TFIM') 
TFYH = tf.placeholder("float", [None, MAX_CHAR, NC], name = 'TFYH')

def DivideIntoSubimages(I):
    '''
    Divides an image into chunks to feed into OCR net
    '''
    h, w, c = I.shape
    H, W = h // IS[0], w // IS[1]
    HP = IS[0] * H
    WP = IS[1] * W
    I = I[0:HP, 0:WP]     #Discard any extra pixels
    return I.reshape(H, IS[0], -1, IS[1], c).swapaxes(1, 2).reshape(-1, IS[0], IS[1], c)
    
def MakeNet(nn = 'ocrnet'):
    #Architecture of the neural network
    #The input volume is reduce to the shape of the output in conv layers
    #18 / 2 * 3 * 3 = 1 and 640 / 2 * 5 = 64 output.shape
    ws = [('C', [5, 5,  3, NC // 2], [1, 2, 2, 1]), ('AF', 'relu'),     
          ('C', [4, 4, NC // 2, NC], [1, 3, 1, 1]), ('AF', 'relu'), 
          ('C', [3, 5, NC,      NC], [1, 3, 5, 1]), ('AF', 'relu'),
          ('R', [-1, MAX_CHAR, NC])]
    #Create the neural network in TensorFlow
    return ANNC(IS, ws, batchSize = 512, learnRate = 2e-5, maxIter = 64, name = nn, reg = 1e-5, tol = 1e-2, verbose = True, X = TFIM, Y = TFYH)
    
def FitModel(cnnc, A, Y, T, FN):
    print('Fitting model...')
    ss = ShuffleSplit(n_splits = 1)
    trn, tst = next(ss.split(A))
    #Fit the network
    cnnc.fit(A[trn], Y[trn])
    #The predictions as sequences of character indices
    YH = []
    for i in np.array_split(np.arange(A.shape[0]), 32): 
        YH.append(cnnc.predict(A[i]))
    YH = np.vstack(YH)
    #Convert from sequence of char indices to strings
    PS = np.array([''.join(YHi) for YHi in YH])
    #Compute the accuracy
    S1 = SAcc(PS[trn], T[trn])
    S2 = SAcc(PS[tst], T[tst])
    print('Train: ' + str(S1))
    print('Test: ' + str(S2))
    for PSi, Ti, FNi in zip(PS, T, FN):
        if np.random.rand() > 0.99: #Randomly select rows to print
            print(FNi + ': ' + Ti + ' -> ' + PSi)
    print('Fitting with CV data...')
    #Fit remainder
    cnnc.SetMaxIter(4)
    cnnc.fit(A, Y)
    return cnnc
    
def FuseResults(T): #Take mode result from 5 networks
    return (Mode(np.stack([Ti.ravel() for Ti in T]))[0]).reshape(T[0].shape)
    
def ImageToString(I):
    '''
    Uses OCR to transform an image into a string
    '''
    SI = DivideIntoSubimages(I)
    ss = SubimageShape(I)
    YH = FuseResults(TFS.run(YHL, feed_dict = {TFIM: SI}))
    return JoinStrings(CNN[-1]._classes[YH], ss)
    
def JoinStrings(YH, ss):
    '''
    Rejoin substrings according to position of subimages
    '''
    YH = np.array([''.join(YHi) for YHi in YH]).reshape(ss)
    return '\n'.join(''.join(YHij for YHij in YHi) for YHi in YH)

def LoadData(FP = '.'):
    '''
    Loads the OCR dataset. A is matrix of images (NIMG, Height, Width, Channel).
    Y is matrix of characters (NIMG, MAX_CHAR)
    FP:     Path to OCR data folder
    return: Data Matrix, Target Matrix, Target Strings
    '''
    TFP = os.path.join(FP, 'Trn.csv')
    A, Y, T, FN = [], [], [], []
    with open(TFP) as F:
        for i, Li in enumerate(F):
            FNi, Yi = Li.strip().split(',')                     #filename,string
            T.append(Yi)
            A.append(imread(os.path.join(FP, FNi))[:, :, :3])   #Read image and discard alpha channel
            Y.append(list(Yi) + [' '] * (MAX_CHAR - len(Yi)))   #Pad strings with spaces
            FN.append(FNi)
    return np.stack(A), np.stack(Y), np.stack(T), np.stack(FN)

def SAcc(T, PS):
    return sum(sum(i == j for i, j in zip(S1, S2)) / len(S1) for S1, S2 in zip(T, PS)) / len(T)

def SubimageShape(I):
    '''
    Get number of (rows, columns) of subimages
    '''
    h, w, c = I.shape
    return h // IS[0], w // IS[1]

def LoadNets():
    CNN = [MakeNet('ocrnet' + str(i)) for i in range(NNET)]
    if not CNN[-1].RestoreModel('TFModel/', 'ocrgraph'):
        A, Y, T, FN = LoadData()
        for CNNi in CNN:
            FitModel(CNNi, A, Y, T, FN)
        CNN[-1].SaveModel(os.path.join('TFModel', 'ocrgraph'))
        with open('TFModel/_classes.txt', 'w') as F:
            F.write('\n'.join(CNN[-1]._classes))
    else:
        with open('TFModel/_classes.txt') as F:
            cl = F.read().splitlines()
        for CNNi in CNN:
            CNNi.RestoreClasses(cl)
    return CNN
    
CNN = LoadNets()                    #CNNs
YHL = [CNNi.YHL for CNNi in CNN]    #Prediction placeholders
TFS = CNN[-1].GetSes()              #Get tensorflow session
if __name__ == "__main__":
    P = argparse.ArgumentParser(description = 'Deep learning based OCR')
    P.add_argument('-f', action = 'store_true', help = 'Force model training')
    P.add_argument('Img', metavar = 'I', type = str, nargs = '+', help = 'Image files')
    PA = P.parse_args()
    if PA.f:
        FitModel()
    for img in PA.Img:
        I = imread(img)[:, :, :3]    #Read image and discard alpha
        S = ImageToString(I)
        print(S)