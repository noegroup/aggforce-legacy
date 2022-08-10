

import pytorch_lightning as pl
import bgflow as bg
import torch
import torch.utils.data

from bgmol.datasets import WaterDimerRigidTIP3P, WaterDimerFlexibleTIP3P


KT = 2.494338785445972


def kjmol2kt(x):
    return x / KT


class DimerData(pl.LightningDataModule):
    def __init__(self, rigid=True, random_seed=1, train_fraction=0.8, test_fraction=0.2, batch_size=512, test_batch_size=512, **kwargs):
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
        parser.add_argument("--rigid", type=bool, default=True)
        parser.add_argument("--random-seed", type=int, default=1)
        parser.add_argument("--batch-size", type=int, default=512)
        parser.add_argument("--test-batch-size", type=int, default=512)
        parser.add_argument("--train-fraction", type=float, default=0.8)
        parser.add_argument("--test-fraction", type=float, default=0.2)
        return parent_parser

    def prepare_data(self) -> None:
        dataset = WaterDimerRigidTIP3P() if self.rigid else WaterDimerFlexibleTIP3P()
        self.train_set, _, self.test_set = dataset.torch_datasets(
            fields=["xyz", "forces"],
            val_fraction=1.-self.train_fraction-self.test_fraction,
            test_fraction=self.test_fraction
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.test_batch_size)


class CGEnergy(bg.Energy):
    """An RBF net on the distance between two CG beads"""
    def __init__(self, n_rbf):
        super().__init__(dim=(2, 3))
        self.weights = torch.nn.Parameter(torch.randn(n_rbf))
        self.register_buffer("centers", torch.linspace(0, 1.0, n_rbf))
        sigma = 1 / n_rbf
        self.rbf = lambda x: torch.exp(-x ** 2 / (2 * sigma ** 2))

    def _energy(self, r):
        distances = torch.linalg.norm(r[:, 0, :] - r[:, 1, :], dim=-1)
        return (self.weights * self.rbf(distances[..., None] - self.centers)).sum(dim=-1, keepdim=True)


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

    def __init__(self, n_rbf=40, lr=1e-3, force_map="slice", **kwargs):
        super().__init__()
        self.cgmodel = CGEnergy(n_rbf)
        self.lr = lr
        self.position_map = PositionSliceMap()
        self.force_map = DimerEnergy.FORCE_MAPS[force_map]()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DimerEnergy")
        parser.add_argument("--n-rbf", type=int, default=40)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--force-map", type=str, default="slice")
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
