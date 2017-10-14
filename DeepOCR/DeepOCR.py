import numpy as np
import os
import string
from skimage.io import imread
from sklearn.model_selection import ShuffleSplit
from TFANN import CNNR

def SAcc(T, PS):
    return sum(sum(i == j for i, j in zip(S1, S2)) / len(S1) for S1, S2 in zip(T, PS)) / len(T)

#The possible chars
CS = ['\x00'] + list(string.ascii_letters) + list(string.digits)
#Map from char to index
CM = dict(zip(CS, range(len(CS))))
#Number of possible chars
NC = len(CM)

if __name__ == "__main__":
    FP = 'C:/Path/To/Dataset/OCR'
    TFP = os.path.join(FP, 'Train.csv')
    A, Y, T, FN = [], [], [], []
    with open(TFP) as F:
        for Li in F:
            FNi, Yi = Li.strip().split(',')         #filename,string
            Yi = (Yi + ('\x00' * (64 - len(Yi))))   #Pad the string with null chars
            R = [[0 if i != CM[j] else 1 for i in range(NC)] for j in Yi]
            Y.append(R)
            A.append(imread(os.path.join(FP, FNi)))
            T.append(Yi)
            FN.append(FNi)
    A = np.stack(A)     #Input images
    Y = np.stack(Y)     #Target strings in 1-hot encoding
    T = np.stack(T)
    #Architecture of the neural network
    ws = [('C', [4, 4,  3, NC // 2], [1, 2, 2, 1]), ('C', [4, 4, NC // 2, NC], [1, 2, 1, 1]), ('C', [8, 5, NC, NC], [1, 8, 5, 1]), ('R', [-1, 64, NC])]
    #Create the neural network in TensorFlow
    cnnc = CNNR(A[0].shape, ws, batchSize = 64, learnRate = 5e-5, loss = 'smce', maxIter = 12, reg = 1e-5, tol = 1e-2, verbose = True)
    ss = ShuffleSplit(n_splits = 1)
    trn, tst = next(ss.split(A))
    #Fit the network
    cnnc.fit(A[trn], Y[trn])
    #The predictions as sequences of character indices
    YH = np.zeros((Y.shape[0], Y.shape[1]), dtype = np.int)
    for i in np.array_split(np.arange(A.shape[0]), 32): 
        YH[i] = np.argmax(cnnc.predict(A[i]), axis = 2)
    #Convert from sequence of char indices to strings
    PS = np.array([''.join(CS[j] for j in YHi) for YHi in YH])
    #Compute the accuracy
    S1 = SAcc(PS[trn], T[trn])
    S2 = SAcc(PS[tst], T[tst])
    print('Train: ' + str(S1))
    print('Test: ' + str(S2))
    for PSi, Ti, FNi in zip(PS, T, FN):
        print(FNi + ': ' + Ti + ' -> ' + PSi)