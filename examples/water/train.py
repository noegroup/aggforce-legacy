#! /usr/bin/env python

###SBATCH --partition=micro,small,big,gpu
###SBATCH --nodes=1 
###SBATCH --cpus-per-task=8
###SBATCH --mem=1GB 
###SBATCH --time=00:59:00

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=1GB

"""Train a CG model for the water dimer.
"""
import os
import sys
sys.path.append(os.getcwd())

from argparse import ArgumentParser
from model import DimerEnergy, DimerData
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from bgflow.utils import as_numpy


#checkpoint_callback = ModelCheckpoint(
#            dirpath=TRAIN_PATH,
#                monitor="validation_loss",
#                    save_top_k=-1,  # -1 to save all
#                        every_n_epochs=1,
#                            filename="{epoch}-{validation_loss:.4f}",
#                                save_last=True,
#                                )


def dirname(rigid, force_map, train_fraction):
    return f"logs_rbf200_random/rigid:{rigid}_{force_map}_trainfraction{train_fraction:.4f}"


def train():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DimerEnergy.add_model_specific_args(parser)
    parser = DimerData.add_model_specific_args(parser)
    args = parser.parse_args()

    data = DimerData(**vars(args))
    data.prepare_data()
    data.setup()

    energy = DimerEnergy(**vars(args))

    logger = TensorBoardLogger(dirname(data.rigid, energy.force_map.__class__.__name__, data.train_fraction), name="")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_last=False)
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(energy, data)

    # sample
#    loaded = DimerEnergy.load_from_checkpoint(
#        trainer.checkpoint_callback.best_model_path
#    )
#    evaluate(loaded, trainer.checkpoint_callback.best_model_path+".npz")



if __name__ == "__main__":
    train()

