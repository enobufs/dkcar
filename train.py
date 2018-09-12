#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    train.py [--model=<model>] [--js] [--chaos]
    train.py [--tub=<tub1,tub2,..tubn>]  [--base_model=<base_model>] [--no_cache]

Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. Comma separated. Use quotes to use wildcards. ie "~/tubs/*"
    --js             Use physical joystick.
    --chaos          Add periodic random steering when manually driving
"""
import os
from docopt import docopt

import donkeycar as dk
import time

from donkeycar.parts.datastore import TubGroup #, TubWriter
from donkeychainer import dataset as ds
from donkeychainer import model

import numpy as np
import chainer
from chainer import training
from chainer import functions as F
from chainer.training import extensions


def train(cfg, tub_names, new_model_path, base_model_path=None ):
    epochs = 40
    batchsize = 32
    gpu = -1
    plot = True
    resume = False


    dataset = ds.load_data(tub_names)

    train, test = ds.split_data(dataset, ratio=0.7)

    m = model.Linear()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(m)

    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
            train_iter, optimizer, device=gpu, loss_func=m.get_loss_func())

    trainer = training.Trainer(updater, (epochs, 'epoch'), out='results')

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, m, device=gpu, eval_func=m.get_loss_func()))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

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
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()

    tub = args['--tub']
    new_model_path = args['--model']
    base_model_path = args['--base_model']
    train(cfg, tub, new_model_path, base_model_path)





