# MLP using theano
import numpy as np
import theano
import theano.tensor as T
from sklearn.datasets import load_iris

iris = load_iris()
train_x = np.concatenate([iris.data] * 1, axis = 1)
train_y = iris.target
nn_input_dim = train_x.shape[1]
nn_hdim = 6
epsilon = 0.01
batch_size = 30
nn_output_dim = len(iris.target_names)

x = T.matrix('x')
y = T.lvector('y')

W1 = theano.shared(np.random.randn(nn_input_dim, nn_hdim), name='W1')
b1 = theano.shared(np.zeros(nn_hdim), name='b1')
W2 = theano.shared(np.random.randn(nn_hdim, nn_output_dim), name='W2')
b2 = theano.shared(np.zeros(nn_output_dim), name='b2')

z1 = x.dot(W1) + b1
a1 = T.nnet.softmax(z1)
z2 = a1.dot(W2) + b2
a2 = T.nnet.softmax(z2)

loss = T.nnet.categorical_crossentropy(a2, y).mean()
prediction = T.argmax(a2, axis=1)

forward_prop = theano.function([x], a2)
calculate_loss = theano.function([x, y], loss)
predict = theano.function([x], prediction)
accuracy = theano.function([x], T.sum(T.eq(prediction, train_y)))

dW2 = T.grad(loss, W2)
db2 = T.grad(loss, b2)
dW1 = T.grad(loss, W1)
db1 = T.grad(loss, b1)

gradient_step = theano.function(
    [x, y],
    updates=((W2, W2 - epsilon * dW2),
             (W1, W1 - epsilon * dW1),
             (b2, b2 - epsilon * db2),
             (b1, b1 - epsilon * db1)))


def build_model(num_passes = 1000000):
    np.random.seed(0)
    W1.set_value(np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim))
    b1.set_value(np.zeros(nn_hdim))
    W2.set_value(np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim))
    b2.set_value(np.zeros(nn_output_dim))

    for i in range(0, num_passes):

        batch_indices = np.random.randint(150,size=30)
        batch_x, batch_y = train_x[batch_indices], train_y[batch_indices]
        gradient_step(batch_x, batch_y)

        if i % 1000 == 0:
            print("Loss after iteration {0}: {1}".format(i, calculate_loss(train_x, train_y)))
            print(accuracy(train_x))


build_model()