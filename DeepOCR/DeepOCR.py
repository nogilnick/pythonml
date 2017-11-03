import numpy as np
import os
import string
import sys
from skimage.io import imread
from sklearn.model_selection import ShuffleSplit
from TFANN import CNNC

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
    
def ImageToString(I):
    '''
    Uses OCR to transform an image into a string
    '''
    SI = DivideIntoSubimages(I)
    YH = cnnc.predict(SI)
    ss = SubimageShape(I)
    return JoinStrings(YH, ss)
    
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
    TFP = os.path.join(FP, 'Train.csv')
    A, Y, T, FN = [], [], [], []
    with open(TFP) as F:
        for Li in F:
            FNi, Yi = Li.strip().split(',')                     #filename,string
            T.append(Yi)
            A.append(imread(os.path.join(FP, 'Out', FNi)))
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

NC = len(string.ascii_letters + string.digits + ' ')
MAX_CHAR = 64
IS = (18, 640, 3)       #Image size for CNN
#Architecture of the neural network
ws = [('C', [4, 4,  3, NC // 2], [1, 2, 2, 1]), ('C', [4, 4, NC // 2, NC], [1, 2, 1, 1]), ('C', [8, 5, NC, NC], [1, 8, 5, 1]), ('R', [-1, 64, NC])]
#Create the neural network in TensorFlow
cnnc = CNNC(IS, ws, batchSize = 64, learnRate = 5e-5, maxIter = 4, reg = 1e-5, tol = 1e-2, verbose = True)
if not cnnc.RestoreModel('TFModel/', 'ocrnet'):
    A, Y, T, FN = LoadData()
    ss = ShuffleSplit(n_splits = 1, random_state = 42)
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
        print(FNi + ': ' + Ti + ' -> ' + PSi)
    cnnc.SaveModel(os.path.join('TFModel', 'ocrnet'))
    with open('TFModel/_classes.txt', 'w') as F:
        F.write('\n'.join(cnnc._classes))
else:
    with open('TFModel/_classes.txt') as F:
        cnnc.RestoreClasses(F.read().splitlines())
    
if __name__ == "__main__":
    for img in sys.argv[1:]:
        I = imread(img)
        S = ImageToString(I)
        print(S)