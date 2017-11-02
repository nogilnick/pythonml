# pythonml

Repository containing various python machine learning modules. Several of the modules have corresponding blog plosts on my website: [https://nicholastsmith.wordpress.com/](https://nicholastsmith.wordpress.com/).

## bprop.py

A straightforward impelementation of the backpropagation algorithm for MLP networks. [See this blog post](https://nicholastsmith.wordpress.com/2016/03/27/multi-layer-perceptrons-and-backpropagation-a-derivation-and-implementation-in-python/) for more information.

## DeepOCR

An implementation of OCR using TensorFlow. See [the related blog post](https://nicholastsmith.wordpress.com/2017/10/14/deep-learning-ocr-using-tensorflow-and-python/) for more details.

## TFANN

A neural network module containing implementations of MLP, and CNN networks in TensorFlow. The classes in the module adhere to the scikit-learn fit, predict, score interface. Sample code for building an MLP regression model:

```python
import numpy as np
from TFANN import MLPR

A = np.random.rand(32, 4)
Y = np.random.rand(32, 1)
a = MLPR([4, 4, 1], batchSize = 4, learnRate = 1e-4, maxIter = 16, verbose = True)
a.fit(A, Y)
s = a.score(A, Y)
YH = a.predict(A)
```

For building a classification model:

```python
import numpy as np
from TFANN import MLPC

A = np.random.rand(32, 4)
Y = np.array((16 * [1]) + (16 * [0]))
a = MLPC([4, 4, 2], batchSize = 4, learnRate = 1e-4, maxIter = 16, verbose = True)
a.fit(A, Y)
s = a.score(A, Y)
YH = a.predict(A)
```

## TheanoANN.py

A neural network module containing implementations of MLP networks in Theano. The classes in the module adhere to the scikit-learn fit, predict, score interface.

## Stocks

Code for predicting stock price based on historical data. [See this blog post](https://nicholastsmith.wordpress.com/2016/11/04/stock-market-prediction-in-python-part-2/) for more information.