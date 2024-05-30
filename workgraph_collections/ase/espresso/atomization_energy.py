from aiida_workgraph import node, WorkGraph
from ase import Atoms


@node()
def calc_atomization_energy(molecule, molecule_output, atom_output):
    energy = atom_output["energy"] * len(molecule) - molecule_output["energy"]
    return energy


@node.graph_builder(outputs=[["calc_atomization_energy.result", "result"]])
def atomization_energy(atom: Atoms = None, molecule: Atoms = None):
    """Workgraph for atomization energy calculation using Espresso calculator."""

    from .base import espresso_calculator

    wg = WorkGraph("atomization_energy")
    pw_atom = wg.nodes.new(
        espresso_calculator, name="scf_atom", run_remotely=True, atoms=atom
    )
    pw_mol = wg.nodes.new(
        espresso_calculator, name="scf_mol", run_remotely=True, atoms=molecule
    )
    # create the node to calculate the atomization energy
    wg.nodes.new(
        calc_atomization_energy,
        name="calc_atomization_energy",
        molecule=molecule,
        atom_output=pw_atom.outputs["results"],
        molecule_output=pw_mol.outputs["results"],
        run_remotely=True,
    )
    return wg
