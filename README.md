# pythonml

Repository containing various python machine learning modules. Several of the modules have corresponding blog plosts on my website: [https://nicholastsmith.wordpress.com/](https://nicholastsmith.wordpress.com/).

## bprop.py

A straightforward impelementation of the backpropagation algorithm for MLP networks. [See this blog post](https://nicholastsmith.wordpress.com/2016/03/27/multi-layer-perceptrons-and-backpropagation-a-derivation-and-implementation-in-python/) for more information.

## DeepOCR

An implementation of OCR using TensorFlow. See [the related blog post](https://nicholastsmith.wordpress.com/2017/10/14/deep-learning-ocr-using-tensorflow-and-python/) for more details. Sample code (requires a model to be trained):

```python
from skimage.io import imread
#Takes a while to load model
from DeepOCR import ImageToString
A = imread('Foo.png')
S = ImageToString(A)
print(S)
```

## TFANN

A neural network module containing implementations of MLP, and CNN networks in TensorFlow. The classes in the module adhere to the scikit-learn fit, predict, score interface. Sample code for building an MLP regression model:

```python
import numpy as np
from TFANN import ANNR

A = np.random.rand(32, 4)
Y = np.random.rand(32, 1)
a = ANNR([4], [('F', 4), ('AF', 'tanh'), ('F', 1)], maxIter = 16, name = 'mlpr1')
a.fit(A, Y)
S = a.score(A, Y)
YH = a.predict(A)
```

For building an MLP classification model:

```python
import numpy as np
from TFANN import ANNC

A = np.random.rand(32, 4)
Y = np.array((16 * [1]) + (16 * [0]))
a = ANNC([4], [('F', 4), ('AF', 'tanh'), ('F', 2)], maxIter = 16, name = 'mlpc2')
a.fit(A, Y)
S = a.score(A, Y)
YH = a.predict(A)
```

For building an CNN classification model:

```python
import numpy as np
from TFANN import CNNC

A = np.random.rand(32, 9, 9, 3)
Y = np.array((16 * [1]) + (16 * [0]))
ws = [('C', [3, 3, 3, 4], [1, 1, 1, 1]), ('AF', 'relu'), 
      ('P', [1, 4, 4, 1], [1, 2, 2, 1]), ('F', 16), 
      ('AF', 'relu'), ('F', 2)]
a = ANNC([9, 9, 3], ws, maxIter = 12, name = "cnnc1")
a.fit(A, Y)
S = a.score(A, Y)
YH = a.predict(A)
```

## TheanoANN.py

A neural network module containing implementations of MLP networks in Theano. The classes in the module adhere to the scikit-learn fit, predict, score interface.

## Stocks

Code for predicting stock price based on historical data. [See this blog post](https://nicholastsmith.wordpress.com/2016/11/04/stock-market-prediction-in-python-part-2/) for more information.
