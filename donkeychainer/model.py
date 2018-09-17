import numpy as np
import chainer
from chainer import links as L
from chainer import functions as F
from chainer import serializers
from chainer import initializers
from chainer import Chain
from chainer import training
from chainer import optimizers
from chainer import iterators

STEERING_AXIS = 0
THROTTLE = 1

""""""""""""""""
model straightforward
training very different

"""""

class Linear(Chain):
    """
    assume input image size is (3, 120, 160), or CHW
    """

    def __init__(self):
        super(Linear, self).__init__()
        self.dtype = np.float32
        W = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        bias = initializers.Zero(self.dtype)
        
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=3,  out_channels=24, ksize=5, stride=2,
                    initialW=W, initial_bias=bias) # shape=(24, 59, 79)
            self.conv2 = L.Convolution2D(in_channels=24, out_channels=32, ksize=5, stride=2,
                    initialW=W, initial_bias=bias) # shape=(32, 28, 38)
            self.conv3 = L.Convolution2D(in_channels=32, out_channels=64, ksize=5, stride=2,
                    initialW=W, initial_bias=bias) # shape=(64, 13, 18)
            self.conv4 = L.Convolution2D(in_channels=64, out_channels=64, ksize=3, stride=2,
                    initialW=W, initial_bias=bias) # shape=(64, 6, 9)
            self.conv5 = L.Convolution2D(in_channels=64, out_channels=64, ksize=3, stride=1,
                    initialW=W, initial_bias=bias) # shape=(64, 4, 7)
            self.l1 = L.Linear(None,100, initialW=W, initial_bias=bias)    # shape=(100)
            self.l2 = L.Linear(None, 50, initialW=W, initial_bias=bias)    # shape=(100)
            self.l3 = L.Linear(2, initialW=W, initial_bias=bias)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.dropout(self.l1(h), 0.1)
        h = F.dropout(self.l2(h), 0.1)
        return self.l3(h)

    def get_loss_func(self):
        def lf(X, Y):
            # Now, X[n] is a tuple of (where n being a batch index):
            # X[n][0] image: current image
            # X[n][1] prev_image: previous image
            # X[n][2] prev_label: (previous angle, previous throttle)

            # Currently, we only use X[n][0].
            # Conver to a batch of current images, of type ndarray
            X = np.array(list(map(lambda x: x[0], X)))
            A = self(X)
            error = Y - A
            loss = F.sum(error**2)
            chainer.report({'loss': loss}, observer=self)
            return loss

        return lf
        

class Simple(Chain):
    """
    assume input image size is (3, 120, 160), or CHW
    """

    def __init__(self,c1=24,c2=36,drop_out_ratio = 0.25):
        self.drop_out_ratio = drop_out_ratio
        self.dtype = np.float32
        W = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        bias = initializers.Zero(self.dtype)
        super(Simple, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=None, out_channels=c1, ksize=4, stride=2,
                    initialW=W, initial_bias=bias)  # shape=(c1, 59, 79)
            self.conv2 = L.Convolution2D(in_channels=None, out_channels=c2, ksize=3, stride=2,
                    initialW=W, initial_bias=bias) # shape=(c2, 14, 19)
            self.l1 = L.Linear(None,3, initialW=W, initial_bias=bias)    # shape=(3)

    def __call__(self, x):
        h = x
        h = F.relu(self.conv1(h))
        h = F.max_pooling_2d(h,ksize=3,stride=2)
        h = F.relu(self.conv2(h))
        if self.drop_out_ratio>0:
            h = F.dropout(h, self.drop_out_ratio)

        #h = F.max_pooling_2d(h,ksize=(2,3),stride=2)
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = self.l1(h)

        throttle = F.sigmoid(h[:,0:1])
        angle_plus = 0.5 * F.sigmoid(h[:,1:2])
        angle_minus = 0.5 * F.sigmoid(h[:,2:3])
        return F.concat((throttle,angle_plus - angle_minus))



    def get_loss_func(self):
        def lf(X, Y):
            # Now, X[n] is a tuple of (where n being a batch index):
            # X[n][0] image: current image
            # X[n][1] prev_image: previous image
            # X[n][2] prev_label: (previous angle, previous throttle)

            # Currently, we only use X[n][0].
            # Conver to a batch of current images, of type ndarray
            X = np.array(list(map(lambda x: x[0], X)))
            A = self(X)
            loss = F.mean_squared_error(A, Y.astype(np.float32))
            chainer.report({'loss': loss}, observer=self)
            return loss

        return lf

