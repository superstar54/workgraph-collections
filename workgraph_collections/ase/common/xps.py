from aiida_workgraph import node
from ase import Atoms


@node()
def get_marked_atoms(atoms: Atoms = None, atoms_list: list = None, marker: str = "X"):
    """Get the marked atoms for each atom."""
    structures = {"ground": atoms.copy()}
    for data in atoms_list:
        index, orbital = data
        symbol = atoms[index].symbol
        label = f"{symbol}_{orbital}_{index}"
        marked_atoms = atoms.copy()
        marked_atoms[index].symbol = marker
        structures[label] = marked_atoms

    return structures


@node()
def binding_energy(corrections: dict = None, **scf_outputs: dict):
    output_ground = scf_outputs.pop("ground")
    results = {}
    for key, output in scf_outputs.items():
        # key is like "C_1s_0"
        symbol, orbital, index = key.split("_")
        de = output["energy"] - output_ground["energy"]
        e = de + corrections[symbol]
        results[key] = e
    return results
