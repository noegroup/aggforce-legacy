
import mdtraj as md
from matplotlib import pyplot as plt
import numpy as np
import bgflow as bg


def make_traj(traj, system):
    if isinstance(traj, md.Trajectory):
        return traj
    else:
        return md.Trajectory(
            bg.utils.as_numpy(traj).reshape(-1, 6, 3),
            topology=system.mdtraj_topology
        )


def compute_distances(traj):
    return md.compute_distances(traj, [[0, 3]], periodic=False)


def plot_distances(distances):
    fig, ax = plt.subplots(1, 3, figsize=(12,2))
    hist, edges = np.histogram(distances, bins=np.linspace(0,0.5,200), density=True)
    centers = 0.5*(edges[1:] + edges[:-1])
    ax[0].plot(distances)
    ax[1].plot(centers, hist)
    pmf = -np.log(hist)
    ax[2].plot(centers, pmf - pmf.min())
    ax[0].set_title('O-O distance (time series)')
    ax[1].set_title('(histogram)')
    ax[1].set_xlim(0,0.5)
    ax[2].set_title('(PMF)')
    ax[2].set_xlim(0,0.5)
    return centers, pmf