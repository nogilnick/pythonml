#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 07:36:10 2016

@author: f002
"""

from StockPredictor import StockPredictor, ParseData, PlotData
from sklearn.neighbors import KNeighborsRegressor
#Used to get command line arguments
import sys
#Used to check validity of date
from datetime import datetime

#from TFANN import ANNR
    
#Display usage information
def PrintUsage():
    print('Usage:\n')
    print('\tpython stocks.py <csv file> <start date> <end date> <D|W|M>')
    print('\tD: Daily prediction')
    print('\tD: Weekly prediction')
    print('\tD: Montly prediction')

#Main program
def Main(args):
    if(len(args) != 3 and len(args) != 4):
        PrintUsage()
        return
    #Test if file exists
    try:
        open(args[0])
    except Exception as e:
        print('Error opening file: ' + args[0])
        print(str(e))
        PrintUsage()
        return
    #Test validity of start date string
    try:
        datetime.strptime(args[1], '%Y-%m-%d').timestamp()
    except Exception as e:
        print(e)
        print('Error parsing date: ' + args[1])
        PrintUsage()
        return
    #Test validity of end date string
    try:
        datetime.strptime(args[2], '%Y-%m-%d').timestamp()
    except Exception as e:
        print('Error parsing date: ' + args[2])
        PrintUsage()
        return    
    #Test validity of final optional argument
    if(len(args) == 4):
        predPrd = args[3].upper()
        if(predPrd == 'D'):
            predPrd = 'daily'
        elif(predPrd == 'W'):
            predPrd = 'weekly'
        elif(predPrd == 'M'):
            predPrd = 'monthly'
        else:
            PrintUsage()
            return
    else:
        predPrd = 'daily'
    #Everything looks okay; proceed with program
    #Grab the data frame
    D = ParseData(args[0])
    #The number of previous days of data used
    #when making a prediction
    numPastDays = 16
    PlotData(D)
    #Number of neurons in the input layer
    i = numPastDays * 7 + 1
    #Number of neurons in the output layer
    o = D.shape[1] - 1
    #Number of neurons in the hidden layers
    h = int((i + o) / 2)
    #The list of layer sizes
    #layers = [('F', h), ('AF', 'tanh'), ('F', h), ('AF', 'tanh'), ('F', o)]
    #R = ANNR([i], layers, maxIter = 1000, tol = 0.01, reg = 0.001, verbose = True)
    R = KNeighborsRegressor(n_neighbors = 5)
    sp = StockPredictor(R, nPastDays = numPastDays)
    #Learn the dataset and then display performance statistics
    sp.Learn(D)
    sp.TestPerformance()
    #Perform prediction for a specified date range
    P = sp.PredictDate(args[1], args[2], predPrd)
    #Keep track of number of predicted results for plot
    n = P.shape[0]
    #Append the predicted results to the actual results
    D = P.append(D)
    #Predicted results are the first n rows
    PlotData(D, range(n + 1))   
    return (P, n)
    

#Main entry point for the program
if __name__ == "__main__":
    #Main(sys.argv[1:])
    p, n = Main(['yahoostock.csv', '2016-11-02', '2016-12-31', 'D'])