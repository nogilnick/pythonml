import numpy as np
import string
from PIL import Image, ImageFont, ImageDraw

def MakeImg(t, f, fn, s = (100, 100), o = (16, 8)):
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
    draw.text(OFS, t, (255, 255, 255), font = f)
    img.save(fn)

#The possible characters to use
CS = list(string.ascii_letters) + list(string.digits)
RTS = list(np.random.randint(10, 64, size = 8192)) + [64]
#The random strings
S = [''.join(np.random.choice(CS, i)) for i in RTS]
#Get the font
font = ImageFont.truetype("LiberationMono-Regular.ttf", 16)
#The largest size needed
MS = max(font.getsize(Si) for Si in S)
#Computed offset
OFS = ((640 - MS[0]) // 2, (32 - MS[1]) // 2)
#Image size
MS = (640, 32)
Y = []
for i, Si in enumerate(S):
    MakeImg(Si, font, str(i) + '.png', MS, OFS)
    Y.append(str(i) + '.png,' + Si)
#Write CSV file
with open('Train.csv', 'w') as F:
    F.write('\n'.join(Y))