from aiida_workgraph import node
from ase import Atoms


@node()
def get_non_equivalent_site(
    atoms: Atoms = None,
    min_cell_length: float = 4.0,
    elements_list: list = None,
    marker: str = "X",
    is_molecule: bool = True,
):
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer, SpacegroupAnalyzer
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.build import make_supercell
    import numpy as np

    if elements_list is None:
        raise "Elements shoud not be None."
    if is_molecule:
        structure = AseAtomsAdaptor().get_molecule(atoms)
        # Set unit cell based on Martyna-Tuckerman approach
        x_extent = max(
            atoms.positions[:, 0].max() - atoms.positions[:, 0].min(), min_cell_length
        )
        y_extent = max(
            atoms.positions[:, 1].max() - atoms.positions[:, 1].min(), min_cell_length
        )
        z_extent = max(
            atoms.positions[:, 2].max() - atoms.positions[:, 2].min(), min_cell_length
        )
        atoms.set_cell([2 * x_extent, 2 * y_extent, 2 * z_extent])
        atoms.center()

        pga = PointGroupAnalyzer(structure)
        eq_sets = pga.get_equivalent_atoms()["eq_sets"]
        structures = {"ground": {"atoms": atoms.copy()}}
        for index, data in eq_sets.items():
            if atoms[index].symbol in elements_list:
                label = f"{atoms[index].symbol}_{index}"
                marked_atoms = atoms.copy()
                marked_atoms[index].symbol = marker
                structures[label] = {
                    "symbol": atoms[index].symbol,
                    "indices": data,
                    "atoms": marked_atoms,
                }
    else:
        # Get the current dimensions of the unit cell
        cell_length = atoms.get_cell_lengths_and_angles()[:3]
        # Calculate the number of times to repeat the structure in each dimension to ensure the minimum distance
        repeats = [int(np.ceil(min_cell_length / dim)) for dim in cell_length]
        # Create the supercell
        supercell = make_supercell(atoms, P=np.diag(repeats))
        #
        structure = AseAtomsAdaptor().get_structure(atoms)
        sga = SpacegroupAnalyzer(structure)
        distinct_sites = sga.get_symmetrized_structure()
        structures = {}
        for i in range(len(distinct_sites.equivalent_sites)):
            site = distinct_sites.equivalent_sites[i][0]
            index = distinct_sites.equivalent_indices[i][0]
            label = f"{site.species_string}_{index}"
            marked_atoms = supercell.copy()
            marked_atoms[index].symbol = marker
            structures[label] = {
                "symbol": site.species_string,
                "atoms": marked_atoms,
                "indices": distinct_sites.equivalent_indices[i],
            }

    return structures


@node()
def get_marked_atoms(atoms: Atoms = None, atoms_list: list = None, marker: str = "X"):
    """Get the marked atoms for each atom."""
    structures = {"ground": {"atoms": atoms.copy()}}
    for data in atoms_list:
        index, orbital = data
        symbol = atoms[index].symbol
        label = f"{symbol}_{orbital}_{index}"
        marked_atoms = atoms.copy()
        marked_atoms[index].symbol = marker
        structures[label] = {
            "indices": {index},
            "atoms": marked_atoms,
        }

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
