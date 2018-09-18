# dkcar

## Features
* Uses Chainer for training
* In addition to training with GPU, Intel's iDeep (MKL-DNN) is supported.

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

## Description of files
### `manage.py`
A collection of operations to be run on donkey car.
Commands:
* drive
* train (to be removed)


### `local.py`
A collection of operations to be run on host PC. 
Commands:
* train
* infer


## Training

First, copy your data under `./data` folder inside `dkcar` project. For example
```
mkdir -p `project-parent-folder`/dkcar/data
cp -r <my_tub_folder> `project-parent-folder`/dkcar/data/tub_20180909_1
```

Then, run the following command:
```
python local.py train --tub ./data/tub_20180909_1 --model ./models/tub_2010909_1
```

If your environment has ideep4py installed, you can add `--use_ideep` flag, like this:
```
python local.py train --tub ./data/tub_20180909_1 --model ./models/tub_2010909_1 --use_ideep
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
python local.py train --tub ./data/tub_20180909_1 --model ./models/tub_2010909_1 --use_ideep
```
> IDeep is installed on this image.

## View dataset
You can run the following command to view dataset in the specified folder.
```
python local.py view --tub ./data/tub_20180909_1
```

## Velocity detection (EXPERIMENTAL)
Run the following command to detect velocity from images in the specified folder.
```
python local.py detect-velocity --tub ./data/tub_20180909_1
```

## TODO
* The model `Line` is not working correctly yet
* Implement `Categorical` - the one equivalent to KerasCategorical in the donkey@2.5.1.
* Integrate the local.py into manage.py (donkey car's main routine).

