
import argparse
import pytorch_lightning as pl
import bgflow as bg
import torch
import torch.utils.data

from bgmol.datasets import WaterDimerRigidTIP3P, WaterDimerFlexibleTIP3P


KT = 2.494338785445972
DOWNLOAD_DATA = False


def kjmol2kt(x):
    return x / KT


class DimerData(pl.LightningDataModule):
    def __init__(self, rigid=True, random_seed=1, train_fraction=0.8, test_fraction=0.2, batch_size=64, test_batch_size=512, **kwargs):
        super().__init__()
        self.rigid = rigid
        self.random_seed = random_seed
        self.train_fraction = train_fraction
        self.test_fraction = test_fraction
        self.train_batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.train_set = None
        self.test_set = None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DimerData")
        parser.add_argument("--rigid", action=argparse.BooleanOptionalAction)
        parser.add_argument("--random-seed", type=int, default=1)
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--test-batch-size", type=int, default=512)
        parser.add_argument("--train-fraction", type=float, default=0.8)
        parser.add_argument("--test-fraction", type=float, default=0.2)
        return parent_parser

    def prepare_data(self) -> None:
        dataset = WaterDimerRigidTIP3P(download=DOWNLOAD_DATA) if self.rigid else WaterDimerFlexibleTIP3P(download=DOWNLOAD_DATA)
        self.train_set, _, self.test_set = dataset.torch_datasets(
            fields=["xyz", "forces"],
            val_fraction=1.-self.train_fraction-self.test_fraction,
            test_fraction=self.test_fraction
        )
        print("#Train data", len(self.train_set))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.test_batch_size)


class CGEnergy(bg.Energy):
    """An RBF net on the distance between two CG beads"""
    def __init__(self, n_rbf, trainable_rbf=False):
        super().__init__(dim=(2, 3))
        self.weights = torch.nn.Parameter(torch.randn(n_rbf))
        if trainable_rbf:
            self.sigma = torch.nn.Parameter(torch.ones(n_rbf) * 1 / n_rbf)
            self.centers = torch.nn.Parameter(torch.linspace(0.0, 1.0, n_rbf))
        else:
            self.sigma = 1 / n_rbf
            self.register_buffer("centers", torch.linspace(0.0, 1.0, n_rbf))
        self.rbf = lambda x: torch.exp(-x ** 2 / (2 * self.sigma ** 2))
        # self.repulsion_params = torch.nn.Parameter(torch.randn(2))
        # self.attraction_params = torch.nn.Parameter(torch.randn(2))
        # self.softplus = torch.nn.Softplus()

    def _energy(self, r):
        distances = torch.linalg.norm(r[:, 0, :] - r[:, 1, :], dim=-1)
        rbf_energy = (self.weights * self.rbf(distances[..., None] - self.centers)).sum(dim=-1, keepdim=True)
        # factor1, exponent1 = self.softplus(self.repulsion_params)
        # factor2, exponent2 = self.softplus(self.attraction_params)
        # repulsion = factor1 * (0.1 * distances) ** (-12)
        # attraction = - (factor2 / distances) ** (1+exponent2)
        # energy = rbf_energy + (repulsion + attraction)[...,None]
        # return energy
        return rbf_energy


class PositionSliceMap(torch.nn.Module):
    def forward(self, r):
        return r[:, [0, 3], :]


class ForceSliceMap(torch.nn.Module):
    def forward(self, r, f):
        return f[:, [0, 3], :]


class ForceAggregationMap(torch.nn.Module):
    def forward(self, r, f):
        return f[:, [0, 3], :] + f[:, [1, 4], :] + f[:, [2, 5], :]


class DimerEnergy(pl.LightningModule):
    FORCE_MAPS = {
        "slice": ForceSliceMap,
        "agg": ForceAggregationMap
    }

    def __init__(self, n_rbf=40, lr=1e-2, force_map="slice", trainable_rbf=False, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.cgmodel = CGEnergy(n_rbf, trainable_rbf=trainable_rbf)
        self.lr = lr
        self.position_map = PositionSliceMap()
        self.force_map = DimerEnergy.FORCE_MAPS[force_map]()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DimerEnergy")
        parser.add_argument("--n-rbf", type=int, default=40)
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--force-map", type=str, default="slice")
        parser.add_argument("--trainable-rbf", action=argparse.BooleanOptionalAction, default=False) 
        return parent_parser

    def force_residual(self, positions, forces):
        # reduce units
        forces = kjmol2kt(forces)
        cg_positions = self.position_map(positions)
        cg_forces = self.force_map(positions, forces)
        model_forces = self.cgmodel.force(cg_positions)
        mse = (model_forces - cg_forces)**2
        mse = mse.mean()
        return mse

    def training_step(self, batch, batch_idx):
        loss = self.force_residual(*batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.force_residual(*batch)
        self.log("val_loss", loss)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
