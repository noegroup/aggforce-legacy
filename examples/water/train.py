"""Train a CG model for the water dimer.
"""

from argparse import ArgumentParser
from model import DimerEnergy, DimerData
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


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

    logger = TensorBoardLogger(f"logs/rigid:{data.rigid}_{energy.force_map.__class__.__name__}")
    checkpoint_callback = ModelCheckpoint(every_n_epochs=25)
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(energy, data)


if __name__ == "__main__":
    train()

