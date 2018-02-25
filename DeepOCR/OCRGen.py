import numpy as np
from numpy.random import randint, choice
import os
import string
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as mpl
    
def GenBlock(MINC, MAXC, s = (100, 100), o = (0, 0)):
    '''
    Generates a single image block. This is the image size the CNN uses
    '''
    Si = ''.join(choice(CS, randint(MINC, MAXC)))
    img = Image.new('RGB', s, "black")
    draw = ImageDraw.Draw(img)
    draw.text(o, Si, (255, 255, 255), font = TF)
    return np.array(img), Si
        
def GenImage(NR, NC, IS, MINC = 10, MAXC = 64, NIMG = 128, DP = 'Trn'):
    '''
    NR:   Number of rows (blocks)
    NC:   Number of columns (blocks)
    IS:   Image size
    MINC: Minimum number of characters per line
    MAXC: Maximum number of characters per line
    NIMG: Number of images to generate
    DP:   Directory path to write images
    '''
    Y = []
    MS = GetFontSize(MAXC)
    NB = NR * NC #Number of blocks total = Rows * Cols
    for i in range(NIMG):               #Write images to ./Out/ directory
        Im, Ym = MergeBlock(NR, NC, [GenBlock(MINC, MAXC, IS) for _ in range(NB)])
        FNi = os.path.join(DP, '{:05d}.png'.format(i))
        mpl.imsave(FNi, Im)
        Y.append(FNi + ',' + Ym)
    with open(DP + '.csv', 'w') as F:   #Write CSV file
        F.write('\n'.join(Y))
   
def GetFontSize(MAXC):
    '''
    Gets the maximum size of an image containing characters in CS
    of maximum length MAXC
    '''
    img = Image.new('RGB', (1, 1), "black")
    draw = ImageDraw.Draw(img)
    h, w = 0, 0
    for CSi in CS:  #Get max height and width possible characters CS
        tsi = draw.textsize(CSi * MAXC, font = TF) 
        h = max(tsi[0], h)
        w = max(tsi[1], w)
    return (h, w)   
    
def MergeBlock(NR, NC, T):
    '''
    Merges blocks into combined images that are NR blocks tall and NC blocks wide
    NR:  Number of rows (blocks)
    NC:  Number of columns (blocks)
    T:   List of outputs from GenBlock
    ret: Merged image, Merged string
    '''
    B = np.array([t[0] for t in T])
    Y = np.array([t[1] for t in T])
    n, r, c, _ = B.shape
    return Unblock(B, r * NR, c * NC), '@'.join(''.join(Yi) for Yi in Y.reshape(NR, NC))
     
def Unblock(I, h, w):
    '''
    I:   Array of shape (n, NR, NC, c)
    h:   Height of new array
    w:   Width of new array
    ret: Array of shape (h, w, c)
    '''
    n, NR, NC, c = I.shape
    return I.reshape(h // NR, -1, NR, NC, c).swapaxes(1, 2).reshape(h, w, c)
       
TF = ImageFont.truetype('consola.ttf', 18)
#Possible characters to use
CS = list(string.ascii_letters + string.digits + ' ')

if __name__ == "__main__":
    nc, xc = 10, 64
    ms = GetFontSize(xc)
    print('CNN Image Size: ' + str(ms))
    GenImage(1, 1, ms, nc, xc, NIMG = 9999, DP = 'Trn') #Training data
    GenImage(4, 2, ms, nc, xc, NIMG = 256,  DP = 'Tst') #Testing data