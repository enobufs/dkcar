#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    train.py (view) [--tub=<tub1,tub2,..tubn>] [--mask=<img-mask-path>]
    train.py (infer) [--tub=<tub1,tub2,..tubn>] [--model=<model>] [--use_ideep] [--mask=<img-mask-path>]
    train.py (train) [--tub=<tub1,tub2,..tubn>] [--model=<model>] [--base_model=<base_model>] [--use_ideep] [--mask=<img-mask-path>]

Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. Comma separated. Use quotes to use wildcards. ie "~/tubs/*"
    --models <path>  Path to model.
    --js             Use physical joystick.
    --chaos          Add periodic random steering when manually driving
"""
import os
from docopt import docopt

import time

from donkeychainer import dataset as ds
from donkeychainer import model
from donkeychainer import tool

import numpy as np
import matplotlib
if os.environ.get('DISPLAY') == None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import chainer
from chainer import training
from chainer import functions as F
from chainer.training import extensions

Model = model.Linear

def view(cfg, tub_names, image_mask_path=None):
    mask = None

    if image_mask_path:
        print('Using image mask at:', image_mask_path)
        mask = ds.load_image(image_mask_path)

    dataset = ds.load_data(tub_names, mask)

    for data in dataset:
        x = data[0][0]; # (3, 120, 160)
        #x = x[:,40:]

        x = x.transpose(1, 2, 0)
        x = x[...,::-1]
        cv2.imshow('Lane Detection', x)
        cv2.waitKey(50)

    cv2.destroyAllWindows()

def infer(cfg, tub_names, model_path, use_ideep=False, image_mask_path=None):
    gpu = -1
    mask = None

    m = Model()
    print('loading model from {}'.format(model_path))
    chainer.serializers.load_npz(model_path, m)

    if use_ideep:
        # Enable iDeep's function computations
        chainer.config.use_ideep = "always"
        # Enable iDeep's opitimizer computations
        m.to_intel64()
    elif gpu >= 0:
        m.to_gpu()

    if image_mask_path:
        print('Using image mask at:', image_mask_path)
        mask = ds.load_image(image_mask_path)

    dataset = ds.load_data(tub_names, mask)

    angle = []
    throttle = []

    for idx, data in enumerate(dataset):
        #if idx >= 10:
        #    break

        # Infer the first image
        x = data[0][0];
        x = chainer.Variable(x.reshape(1, 3, 120, 160))
        with chainer.using_config('train', False):
            startAt = time.time()
            res = m(x)
            elapsed = round((time.time() - startAt) * 1000, 3)
            print("processed data at {}, took {} msec".format(idx, elapsed))
            angle.append((data[1][0], res.data[0,0]))
            throttle.append((data[1][1], res.data[0,1]))

    print("plotting ...")

    for d in angle:
        x = d[0]
        y = d[1]
        area = np.pi * (2**2)
        color = 'b'
        plt.scatter(x, y, s=area, c=color, alpha=0.5)

    for d in throttle:
        x = d[0] # expected
        y = d[1] # inferred
        area = np.pi * (2**2)
        color = 'r'
        plt.scatter(x, y, s=area, c=color, alpha=0.5)

    """
    fig = plt.figure()
    fig.suptitle('Linear Model Performance (Blue: angle, Red: throttle)', fontsize=14)
    plt.xlabel('Expected', fontsize=18)
    plt.ylabel('Inferred', fontsize=16)
    """
    plt.show()

def train(cfg, tub_names, new_model_path, use_ideep=False, image_mask_path=None):
    epochs = 40
    batchsize = 32
    gpu = -1
    plot = True
    resume = False
    mask = None

    out_dir = tool.make_output_dir('results')

    if image_mask_path:
        print('Using image mask at:', image_mask_path)
        mask = ds.load_image(image_mask_path)

    dataset = ds.load_data(tub_names, mask)

    train, test = ds.split_data(dataset, ratio=0.7)

    m = Model()

    if use_ideep:
        # Enable iDeep's function computations
        chainer.config.use_ideep = "always"
        # Enable iDeep's opitimizer computations
        m.to_intel64()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(m)

    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
            train_iter, optimizer, device=gpu, loss_func=m.get_loss_func())

    trainer = training.Trainer(updater, (epochs, 'epoch'), out=out_dir)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, m, device=gpu, eval_func=m.get_loss_func()))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    #trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(resume, trainer)

    # Run the training
    trainer.run()


    print("Training COMPLETE!")

    if new_model_path is None:
        new_model_path = "{}/weights.npz".format(out_dir)

    chainer.serializers.save_npz(new_model_path, m)
    print("Saved trained model at {}".format(new_model_path))
    print("See other results under {}".format(out_dir))

if __name__ == '__main__':
    np.random.seed(1)
    args = docopt(__doc__)
    cfg = {}

    tub = args['--tub']
    use_ideep = args['--use_ideep']
    img_mask_path = args['--mask']

    if args['view']:
        view(cfg, tub, img_mask_path)

    elif args['infer']:
        model_path = args['--model']
        infer(cfg, tub, model_path, use_ideep, img_mask_path)

    elif args['train']:
        new_model_path = args['--model']
        train(cfg, tub, new_model_path, use_ideep, img_mask_path)

