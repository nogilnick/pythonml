import numpy as np
from numpy.random import randint, choice
import os
import string
from PIL import Image, ImageFont, ImageDraw

TF = ImageFont.truetype('consola.ttf', 18)

def MakeImg(t, f, fn, s = (100, 100), o = (0, 0)):
    '''
    Generate an image of text
    t:      The text to display in the image
    f:      The font to use
    fn:     The file name
    s:      The image size
    o:      The offest of the text in the image
    '''
    img = Image.new('RGB', s, "black")
    draw = ImageDraw.Draw(img)
    draw.text(o, t, (255, 255, 255), font = f)
    img.save(fn)
    
def GetFontSize(S):
    img = Image.new('RGB', (1, 1), "black")
    draw = ImageDraw.Draw(img)
    return draw.textsize(S, font = TF)

def GenSingleLine(MINC = 10, MAXC = 64, NIMG = 128, DP = 'Out'):               #Font path
    #The possible characters to use
    CS = list(string.ascii_letters) + list(string.digits)
    MS = font.getsize('0' * MAXC)   #Size needed to fit MAXC characters
    Y = []
    for i in range(NIMG):               #Write images to ./Out/ directory
        Si = ''.join(RC(CS, randint(MINC, MAXC)))
        FNi = str(i) + '.png'
        MakeImg(Si, TF, os.path.join(DP, FNi), MS)
        Y.append(FNi + ',' + Si)
    with open('Train.csv', 'w') as F:   #Write CSV file
        F.write('\n'.join(Y))
        
def GenMultiLine(ML = 5, MINC = 10, MAXC = 64, NIMG = 128, DP = 'Img'):
    #The possible characters to use
    CS = list(string.ascii_letters) + list(string.digits)
    MS = GetFontSize('\n'.join(ML * ['0' * MAXC]))
    Y = []
    for i in range(NIMG):               #Write images to ./Out/ directory
        Si = '\n'.join(''.join(choice(CS, randint(MINC, MAXC))) for _ in range(randint(1, ML + 1)))
        FNi = str(i) + '.png'
        MakeImg(Si, TF, os.path.join(DP, FNi), MS)
        Y.append(FNi + ',' + Si.replace('\n', '@'))
        
if __name__ == "__main__":
    GenMultiLine(ML = 10, MAXC = 128)