from aiida_workgraph import node, WorkGraph

@node()
def calc_atomization_energy(mol, molecule_output, atom_output):
    energy = atom_output["energy"]*len(mol) - molecule_output["energy"]
    return energy


def atomization_energy(name="atomization_energy", atom=None, mol=None):
    from .base import pw_calculator
    wg = WorkGraph("atomization_energy")
    pw_atom = wg.nodes.new(pw_calculator, name="scf_atom", run_remotely=True,
                           atoms=atom)
    pw_mol = wg.nodes.new(pw_calculator, name="scf_mol", run_remotely=True,
                            atoms=mol)
    # create the node to calculate the atomization energy
    wg.nodes.new(calc_atomization_energy, name="calc_atomization_energy",
                 mol=mol,
                 atom_output=pw_atom.outputs["result"],
                 molecule_output=pw_mol.outputs["result"],
                 run_remotely=True,)
    return wg