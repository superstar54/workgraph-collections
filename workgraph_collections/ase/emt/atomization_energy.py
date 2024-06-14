from aiida_workgraph import task, WorkGraph
from ase import Atoms


@task()
def calc_atomization_energy(molecule: Atoms, molecule_output: dict, atom_output: dict):
    energy = atom_output["energy"] * len(molecule) - molecule_output["energy"]
    return energy


@task.graph_builder(outputs=[["calc_atomization_energy.result", "result"]])
def atomization_energy(atom: Atoms = None, molecule: Atoms = None):
    """Workgraph for atomization energy calculation using EMT calculator."""
    from .base import emt_calculator

    wg = WorkGraph("Atomization energy")
    pw_atom = wg.tasks.new(
        emt_calculator, name="scf_atom", run_remotely=True, atoms=atom
    )
    pw_mol = wg.tasks.new(
        emt_calculator, name="scf_mol", run_remotely=True, atoms=molecule
    )
    # create the task to calculate the atomization energy
    wg.tasks.new(
        calc_atomization_energy,
        name="calc_atomization_energy",
        molecule=molecule,
        atom_output=pw_atom.outputs["results"],
        molecule_output=pw_mol.outputs["results"],
        run_remotely=True,
    )
    return wg
