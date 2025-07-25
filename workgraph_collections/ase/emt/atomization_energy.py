from aiida_workgraph import task, WorkGraph
from ase import Atoms


@task()
def calc_atomization_energy(molecule: Atoms, molecule_output: dict, atom_output: dict):
    energy = atom_output["energy"] * len(molecule) - molecule_output["energy"]
    return energy


@task.graph_builder(
    outputs=[{"name": "result", "from": "calc_atomization_energy.result"}]
)
def atomization_energy(atom: Atoms = None, molecule: Atoms = None):
    """Workgraph for atomization energy calculation using EMT calculator."""
    from .base import emt_calculator

    wg = WorkGraph("Atomization energy")
    pw_atom = wg.add_task(
        "workgraph.pythonjob", function=emt_calculator, name="scf_atom", atoms=atom
    )
    pw_mol = wg.add_task(
        "workgraph.pythonjob", function=emt_calculator, name="scf_mol", atoms=molecule
    )
    # create the task to calculate the atomization energy
    wg.add_task(
        "workgraph.pythonjob",
        function=calc_atomization_energy,
        name="calc_atomization_energy",
        molecule=molecule,
        atom_output=pw_atom.outputs["results"],
        molecule_output=pw_mol.outputs["results"],
    )
    return wg
