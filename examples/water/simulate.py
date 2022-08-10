"""Run simulations to produce ground truth data for the water dimer.

Running these simulations requires the following libraries to be installed:
- click
- openmm
- bgmol
- tqdm
"""

import os
import click
import openmm
from tqdm.auto import tqdm
from openmm import LangevinIntegrator
from openmm import unit
import mdtraj as md
import bgmol


DEFAULT_HARMONIC_RESTRAINT = 3 * unit.kilojoules_per_mole / unit.nanometer ** 2


def make_dimer_system(constrained=True, harmonic_restraint=DEFAULT_HARMONIC_RESTRAINT):
    return bgmol.system_by_name('WaterCluster', n_waters=2, K=harmonic_restraint, constrained=constrained)


@click.command()
@click.option("--constrained/--unconstrained", default=True, help="If the water molecules should be rigid.")
@click.option("-t", "--temperature", default=300, type=float, help="Temperature in kelvin.")
@click.option("-dt", "--timestep", default=1.0, type=float, help="Time step in femtoseconds.")
@click.option("-i", "--report-interval", default=1.0, type=float, help="Time between subsequent reports in picoseconds.")
@click.option("-n", "--n-samples", default=50000, help="How many data points to create.")
@click.option("-e", "--equilibration", default=10.0, help="How many picoseconds for equilibration.")
@click.option("--platform", default=None, type=click.Choice(["cpu"]), help="Whether to force running on a single CPU thread.")
def simulate(constrained, temperature, timestep, report_interval, n_samples, equilibration, platform):
    """Simulate the dimer system."""
    dimer = make_dimer_system(constrained=constrained)
    # create simulation object
    integrator = LangevinIntegrator(temperature, 1/unit.picoseconds, timestep * unit.femtoseconds)
    if platform == "cpu":
        platform = openmm.Platform.getPlatformByName("CPU")
        platform.setPropertyDefaultValue("Threads", '1')
    simulation = dimer.create_openmm_simulation(integrator=integrator, platform=platform)
    simulation.reporters.clear()
    n_equilibration = int(equilibration * 1000 / timestep)
    simulation.step(n_equilibration)
    # create reporters
    stub = "water_{}".format('constrained' if constrained else 'unconstrained')
    index = 0
    while os.path.isfile(stub + f"_{index}.h5"):
        index += 1
    out_stub = f"{stub}_{index}"
    print(f"Running simulation {out_stub}")
    interval = int(report_interval * 1000 / timestep)
    simulation.reporters = dimer.create_openmm_reporters(out_stub=out_stub, interval=interval, forces=True)
    # Run simulation
    for _ in tqdm(range(n_samples)):
        simulation.step(interval)
    del simulation

    # Test
    traj = md.load_hdf5(f"{out_stub}.h5")
    print(f"Simulation finished. Created a trajectory with {traj.n_frames} frames")


if __name__ == "__main__":
    simulate()


