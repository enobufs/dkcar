import os
import glob
import re
import json
from PIL import Image
import numpy as np
from chainer.datasets import tuple_dataset
#from chainer.datasets import split_dataset
from chainer.datasets import split_dataset_random
from chainer.datasets import ConcatenatedDataset


# Load image file and return image data in CHW format with [0:1) float32.
def load_image( infilename ):
    img = Image.open(infilename)
    img.load()
    #img = img.convert('RGB')

    scale = 1
    narr = np.asarray(img, dtype=np.float32)
    narr *= scale / 255.
    narr = narr.transpose(2, 0, 1) # HWC to CHW

    return narr

def mask_image(image, mask):
    return image * mask

def sort_files(in_files):
    # make a list of (seq, file_name)
    tmp_list = []
    for file_name in in_files:
        mo = re.match( r'.+[/\\]record_([0-9]+)\.json$', file_name, re.M|re.I)
        if mo == None:
            continue

        seq = int(mo.group(1))
        tmp_list.append((seq, file_name))

    # sort by seq
    # 1. take second element for sort
    def get_key(elem):
        return elem[0]

    # 2. sort list with key
    tmp_list.sort(key=get_key)

    return list(map(lambda x: x[1], tmp_list))

def load_data(paths, mask=None):
    path_list = paths.split(',')
    datasets = []
    for path in path_list:
        #files = [f for f in listdir(path) if isfile(join(path, f))]
        files = glob.glob(os.path.join(path, "*.json"))

        # sort files by sequence number in the file name
        files = sort_files(files)

        images = []
        labels = []
        prev_image = None
        prev_label = None

        for file_name in files:
            with open(file_name) as f:
                data = json.load(f)

                image_path = os.path.join(path, data['cam/image_array'])
                img = load_image(image_path)
                if mask is not None:
                    try:
                        img = mask_image(img, mask)
                    except:
                        print("Error when processing image file at", image_path)
                        raise
                images.append((img, prev_image, prev_label))
                prev_image = img;
                prev_label = (data['user/angle'], data['user/throttle'])
                labels.append(prev_label)

        datasets.append(tuple_dataset.TupleDataset(images, labels))

    return ConcatenatedDataset(*datasets)

def split_data(dataset, ratio=0.7):
    split_at = int(len(dataset) * ratio)
    return split_dataset_random(dataset, split_at, 123)


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
