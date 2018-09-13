# dkcar

## Features
* Uses Chainer for training
* Intel's iDeep (MKL-DNN) is supported

## Installation
First you will need to install donkeycar SDK (python package):
```
cd `project-parent-folder`
git clone git@github.com:enobufs/donkey.git
cd donkey
git checkout 2.5.1-enobufs
pip install -e .
```

Then, checkout enobufs' dkcar project.
```
cd `project-parent-folder`
git clone git@github.com:enobufs/dkcar.git
cd dkcar
git checkout feature/chainer
```

## Training

First, copy your data under `./data` folder inside `dkcar` project. For example
```
mkdir -p `project-parent-folder`/dkcar/data
cp -r <my_tub_folder> `project-parent-folder`/dkcar/data/tub_20180909_1
```

Then, run the following command:
```
python train.py train --tub ./data/tub_20180909_1 --model ./models/tub_2010909_1
```

If your environment has ideep4py installed, you can add `--use_ideep` flag, like this:
```
python train.py train --tub ./data/tub_20180909_1 --model ./models/tub_2010909_1 --use_ideep
```

## Training with Docoker
```
# On your host PC
cd dkcar

# Build docker image
make docker-build

# Run a docker container and attach to a bash shell
make docker-shell

# Now you are on the container. The dkcar folder you see is mounted from your host PC.
cd dkcar

# Run the training
python train.py train --tub ./data/tub_20180909_1 --model ./models/tub_2010909_1 --use_ideep
```
> IDeep is installed on this image.

## TODO
* The model `Line` is not working correctly yet
* Implement `Categorical` - the one equivalent to KerasCategorical in the donkey@2.5.1.
* Integrate the train.py into manage.py (donkey car's main routine).

