from aiida_workgraph import task
from ase import Atoms


@task()
def calc_atomization_energy(molecule: Atoms, molecule_output: dict, atom_output: dict):
    energy = atom_output["energy"] * len(molecule) - molecule_output["energy"]
    return energy


@task.graph
def AtomizationEnergy(
    atom: Atoms = None, molecule: Atoms = None, metadata: dict = None
):
    """Workgraph for atomization energy calculation using EMT calculator."""
    from .base import emt_calculator

    pw_atom_out = emt_calculator(atoms=atom, metadata=metadata).result
    pw_mol_out = emt_calculator(atoms=molecule, metadata=metadata).result
    # create the task to calculate the atomization energy
    return calc_atomization_energy(
        molecule=molecule, atom_output=pw_atom_out, molecule_output=pw_mol_out
    ).result
