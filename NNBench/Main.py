import numpy as np
import time
from sklearn.model_selection import KFold
import os
#For command line arguments
import sys
#Plotting the benchmark results
import matplotlib.pyplot as mpl
from TFANN import ANNR

#Generate data with nf features and ns samples. If new data
#is generated, write it to file so it can be reused across all benchmarks
def GenerateData(nf = 256, ns = 16384):
    try:    #Try to read data from file
        A = np.loadtxt('bdatA.csv', delimiter = ',')
        Y = np.loadtxt('bdatY.csv', delimiter = ',').reshape(-1, 1)
    except OSError:     #New data needs to be generated
        x = np.linspace(-1, 1, num = ns).reshape(-1, 1)
        A = np.concatenate([x] * nf, axis = 1)
        Y = ((np.sum(A, axis = 1) / nf) ** 2).reshape(-1, 1)
        A = (A + np.random.rand(ns, nf)) / (2.0)
        np.savetxt('bdatA.csv', A, delimiter = ',')
        np.savetxt('bdatY.csv', Y, delimiter = ',')
    return (A, Y)
    
#R:     Regressor network to use
#A:     The sample data matrix
#Y:     Target data matrix
#nt:    Number of times to divide the sample matrix
#fn:    File name to write results
def MakeBenchDataFeature(R, A, Y, nt, fn):
    #Divide samples into nt pieces on for each i run benchmark with chunks 0, 1, ..., i
    step = A.shape[1] // nt
    TT = np.zeros((nt, 3))
    for i in range(1, nt):
        #Number of features
        TT[i, 0] = len(range(0, (i * step)))
        print('{:8d} feature benchmark.'.format(int(TT[i, 0])))
        #Training and testing times respectively
        TT[i, 1], TT[i, 2] = RunBenchmark(R, A[:, 0:(i * step)], Y[:, 0:(i * step)])
    #Save benchmark data to csv file
    np.savetxt(fn, TT, delimiter = ',', header = 'Samples,Train,Test')
    
#R:     Regressor network to use
#A:     The sample data matrix
#Y:     Target data matrix
#nt:    Number of times to divide the sample matrix
#fn:    File name to write results
def MakeBenchDataSample(R, A, Y, nt, fn):
    #Divide samples into nt peices on for each i run benchmark with chunks 0, 1, ..., i
    step = A.shape[0] // nt
    TT = np.zeros((nt, 3))
    for i in range(1, nt):
        #Number of samples
        TT[i, 0] = len(range(0, (i * step)))
        print('{:8d} sample benchmark.'.format(int(TT[i, 0])))
        trnt, tstt = RunBenchmark(R, A[0:(i * step)], Y[0:(i * step)])
        #Training time
        TT[i, 1] = trnt
        #Testing time
        TT[i, 2] = tstt
    #Save benchmark data to csv file
    np.savetxt(fn, TT, delimiter = ',', header = 'Samples,Train,Test')
    
def Main():
    if len(sys.argv) <= 1:
        return
    A, Y = GenerateData(ns = 2048)
    #Create layer sizes; make 6 layers of nf neurons followed by a single output neuron
    L = [A.shape[1]] * 6 + [1]
    print('Layer Sizes: ' + str(L))
    if sys.argv[1] == 'theano':
        print('Running theano benchmark.')
        from TheanoANN import TheanoMLPR
        #Create the Theano MLP
        tmlp = TheanoMLPR(L, batchSize = 128, learnRate = 1e-5, maxIter = 100, tol = 1e-3, verbose = True)
        MakeBenchDataSample(tmlp, A, Y, 16, 'TheanoSampDat.csv')
        print('Done. Data written to TheanoSampDat.csv.')
    if sys.argv[1] == 'theanogpu':
        print('Running theano GPU benchmark.')
        #Set optional flags for the GPU
        #Environment flags need to be set before importing theano
        os.environ["THEANO_FLAGS"] = "device=gpu"
        from TheanoANN import TheanoMLPR
        #Create the Theano MLP
        tmlp = TheanoMLPR(L, batchSize = 128, learnRate = 1e-5, maxIter = 100, tol = 1e-3, verbose = True)
        MakeBenchDataSample(tmlp, A, Y, 16, 'TheanoGPUSampDat.csv')
        print('Done. Data written to TheanoGPUSampDat.csv.')
    if sys.argv[1] == 'tensorflow':
        print('Running tensorflow benchmark.')
        #Create the Tensorflow model
        NA = [('F', A.shape[1]), ('AF', 'tanh')] * 6 + [('F', 1)]
        mlpr = ANNR([A.shape[1]], NA, batchSize = 128, learnRate = 1e-5, maxIter = 100, tol = 1e-3, verbose = True)
        MakeBenchDataSample(mlpr, A, Y, 16, 'TfSampDat.csv')
        print('Done. Data written to TfSampDat.csv.')
    if sys.argv[1] == 'plot':
        print('Displaying results.')
        try:
            T1 = np.loadtxt('TheanoSampDat.csv', delimiter = ',', skiprows = 1)
        except OSError:
            T1 = None
        try:
            T2 = np.loadtxt('TfSampDat.csv', delimiter = ',', skiprows = 1)
        except OSError:
            T2 = None
        try:
            T3 = np.loadtxt('TheanoGPUSampDat.csv', delimiter = ',', skiprows = 1)
        except OSError:
            T3 = None
        fig, ax = mpl.subplots(1, 2)
        if T1 is not None:
            PlotBenchmark(T1[:, 0], T1[:, 1], ax[0], '# Samples', 'Train', 'Theano')
            PlotBenchmark(T1[:, 0], T1[:, 2], ax[1], '# Samples', 'Test', 'Theano')
        if T2 is not None:
            PlotBenchmark(T2[:, 0], T2[:, 1], ax[0], '# Samples', 'Train', 'Tensorflow')
            PlotBenchmark(T2[:, 0], T2[:, 2], ax[1], '# Samples', 'Test', 'Tensorflow') 
        if T3 is not None:
            PlotBenchmark(T3[:, 0], T3[:, 1], ax[0], '# Samples', 'Train', 'Theano GPU')
            PlotBenchmark(T3[:, 0], T3[:, 2], ax[1], '# Samples', 'Test', 'Theano GPU') 
        mpl.show()
    
#Plots benchmark data on a given matplotlib axes object
#X:     X-axis data
#Y:     Y-axis data
#ax:    The axes object
#name:  Name of plot for title
#lab:   Label of the data for the legend
def PlotBenchmark(X, Y, ax, xlab, name, lab):
    ax.set_xlabel(xlab)
    ax.set_ylabel('Avg. Time (s)')
    ax.set_title(name + ' Benchmark')
    ax.plot(X, Y, linewidth = 1.618, label = lab)
    ax.legend(loc = 'upper left')
    
#Runs a benchmark on a MLPR model 
#R:     Regressor network to use
#A:     The sample data matrix
#Y:     Target data matrix
def RunBenchmark(R, A, Y):
    #Record training times
    t0 = time.time()
    R.fit(A, Y)
    t1 = time.time()
    trnt = t1 - t0
    #Record testing time
    t0 = time.time()
    YH = R.predict(A)
    t1 = time.time()
    tstt = t1 - t0
    return (trnt, tstt)
    
if __name__ == "__main__":
    Main()
