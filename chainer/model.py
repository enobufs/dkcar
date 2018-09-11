import numpy as np
import chainer
from chainer import Sequential,links as L, functions as F, serializers, Chain, training,optimizers,iterators

STEERING_AXIS = 0
THROTTLE = 1

""""""""""""""""
model straightforward
training very different

"""""

def default_linear():
    #out_size = (in_size-kernel+2*pad)/stride + 1 rounded up
    Sequential(#shape=(3, 120, 160)
        L.Convolution2D(in_channels=3, out_channels=24, ksize=5, stride=2), F.relu,#shape=(24, 59, 79)
        L.Convolution2D(in_channels=24, out_channels=32, ksize=5, stride=2), F.relu,  # shape=(32, 28, 38)
        L.Convolution2D(in_channels=32, out_channels=64, ksize=5, stride=2), F.relu,  # shape=(64, 13, 18)
        L.Convolution2D(in_channels=64, out_channels=64, ksize=3, stride=2), F.relu,  # shape=(64, 6, 9)
        L.Convolution2D(in_channels=64, out_channels=64, ksize=3, stride=1), F.relu,  # shape=(64, 4, 7)
        L.Linear(None,100),# shape=(100)
        lambda x: F.dropout(x,0.1),
        L.Linear(None, 50),  # shape=(100)
        lambda x: F.dropout(x, 0.1),
        L.Linear(2)
    )

class ChainerPilot(Chain):

    def load(self, model_path):
        self.model = serializers.load_npz(model_path)

    def shutdown(self):
        pass

    def train(self, train_gen, val_gen,
              saved_model_path, steps=100, train_split=0.8,
              #verbose=1, min_delta=.0005, patience=5, use_early_stop=True,
              gpu_id=-1):
        epochs = 100
        batchsize = 1

        train_iter = iterators.SerialIterator(train_gen, batchsize)
        test_iter = iterators.SerialIterator(val_gen, batchsize, False, False)

        # selection of your optimizing method
        optimizer = optimizers.Adam()

        # Give the optimizer a reference to the model
        optimizer.setup(self.model)

        # Get an updater that uses the Iterator and Optimizer
        updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

        # Setup a Trainer
        trainer = training.Trainer(updater, (epochs, 'epoch'), out='mnist_result')
        trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))

        return hist

class ChainerLinear(ChainerPilot):
    def __init__(self,model=None):
        super(ChainerLinear,self).__init__()
        with self.init_scope():
            if model:
                self.model = model
            else:
                self.model = default_linear()

    def run(self, img_arr:np.array):
        img_arr=img_arr.swapaxes(0, 2)#chainer uses channel on the 2nd axis
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        with chainer.using_config("train", False):
            outputs = self.model(img_arr).data
        return outputs[0][STEERING_AXIS], outputs[0][THROTTLE]
