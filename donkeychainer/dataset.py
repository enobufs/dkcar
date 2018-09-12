import os
import glob
import re
import json
from PIL import Image
import numpy as np
from chainer.datasets import tuple_dataset
from chainer.datasets import split_dataset


def load_image( infilename ):
    img = Image.open(infilename)
    img.load()
    #img = img.convert('RGB')
    data = np.asarray(img, dtype="int32")
    return data

def load_data(path):
    #files = [f for f in listdir(path) if isfile(join(path, f))]
    files = glob.glob(os.path.join(path, "*.json"))
    images = []
    labels = []

    for file in files:
        mo = re.match( r'.+/record_([0-9]+)\.json$', file, re.M|re.I)
        if mo == None:
            continue

        seq = int(mo.group(1))

        with open(file) as f:
            data = json.load(f)

            image_path = os.path.join(path, data['cam/image_array'])
            img = load_image(image_path)
            img = img.transpose(2, 0, 1)
            images.append(img)
            labels.append((data['user/angle'], data['user/throttle']))

    scale = 1
    images = np.array(images, dtype=np.float32)
    images *= scale / 255.

    return tuple_dataset.TupleDataset(images, labels)

def split_data(dataset, ratio=0.7):
    split_at = int(len(dataset) * ratio)
    return split_dataset(dataset, split_at)


def main():
    dataset = load_data("./data/tub_20180824_1")
    train, test = split_data(dataset, ratio=0.7)


if __name__ == '__main__':
    main()


"""
batchsize = 100

# Load the MNIST dataset
train, test = chainer.datasets.get_mnist()

print(type(train))
print(len(train))
print("train[0]:", train[0])
print("len(train[0]):", len(train[0]))
print("len(train[0][0]):", len(train[0][0]))

print("train[1]:", train[0])
print("len(train[1]):", len(train[0]))
print("len(train[1][0]):", len(train[0][0]))

train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                             repeat=False, shuffle=False)
 """
