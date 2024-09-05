from aiida_workgraph import task
from ase import Atoms


def find_non_equivalent_sites(
    atoms: Atoms = None,
    element_list: list = None,
    is_molecule: bool = True,
):
    """Find the non-equivalent sites for a molecule or crystal.

    Returns:
        dict: A dictionary of non-equivalent sites with the index of
        the first site as the key and the set of indices as the value.
    """
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer, SpacegroupAnalyzer
    from pymatgen.io.ase import AseAtomsAdaptor

    non_equivalent_sites = {}
    if is_molecule:
        structure = AseAtomsAdaptor().get_molecule(atoms)
        pga = PointGroupAnalyzer(structure)
        eq_sets = pga.get_equivalent_atoms()["eq_sets"]
        for key, value in eq_sets.items():
            if structure[key].species_string in element_list:
                non_equivalent_sites[key] = set(value)
    else:
        structure = AseAtomsAdaptor().get_structure(atoms)
        sga = SpacegroupAnalyzer(structure)
        distinct_sites = sga.get_symmetrized_structure()
        for indices in distinct_sites.equivalent_indices:
            if structure[indices[0]].species_string in element_list:
                non_equivalent_sites[indices[0]] = set(indices)
    return non_equivalent_sites


def mark_non_equivalent_sites(atoms, non_equivalent_sites, marker="X"):
    """Mark the non-equivalent sites for a molecule or crystal."""
    structures = {}
    for index, indices in non_equivalent_sites.items():
        label = f"{atoms[index].symbol}_{index}"
        marked_atoms = atoms.copy()
        marked_atoms[index].symbol = marker
        structures[label] = marked_atoms
        marked_atoms.info["equivalent_sites"] = indices
    return structures


def create_supercell(atoms, min_cell_length, is_molecule):
    """Create a supercell for the core-hole calculation."""
    from ase.build import make_supercell
    import numpy as np

    if is_molecule:
        # Set unit cell based on Martyna-Tuckerman approach
        x_extent = max(
            2 * (atoms.positions[:, 0].max() - atoms.positions[:, 0].min()),
            min_cell_length,
        )
        y_extent = max(
            2 * (atoms.positions[:, 1].max() - atoms.positions[:, 1].min()),
            min_cell_length,
        )
        z_extent = max(
            2 * (atoms.positions[:, 2].max() - atoms.positions[:, 2].min()),
            min_cell_length,
        )
        atoms.set_cell([x_extent, y_extent, z_extent])
        atoms.center()
    else:
        # Get the current dimensions of the unit cell
        cell_length = atoms.get_cell_lengths_and_angles()[:3]
        # Calculate the number of times to repeat the structure in each dimension to ensure the minimum distance
        repeats = [int(np.ceil(min_cell_length / dim)) for dim in cell_length]
        # Create the supercell
        atoms = make_supercell(atoms, P=np.diag(repeats))
    return atoms


@task(
    outputs=[
        {"name": "structures", "identifier": "workgraph.namespace"},
    ]
)
def get_marked_structures(
    atoms: Atoms = None,
    atom_list: list = None,
    element_list: list = None,
    marker: str = "X",
    min_cell_length: float = 4.0,
    is_molecule: bool = True,
):
    """
    Generate a supercell structure with non-equivalent sites marked for a molecule or crystal system.

    This function takes a molecular or crystal structure, finds the non-equivalent atomic sites, and marks them
    with a specified marker. It also creates a supercell to ensure that the minimum cell length is satisfied.

    Args:
        atoms (Atoms): The input structure in ASE `Atoms` format.
        atom_list (list, optional): A list of atom indices to mark as non-equivalent.
        element_list (list, optional): A list of elements for which non-equivalent sites will be identified. Either `atom_list`
                                       or `element_list` must be provided.
        marker (str, optional): The symbol used to mark non-equivalent sites. Default is "X".
        min_cell_length (float, optional): The minimum cell length for the supercell. Default is 4.0.
        is_molecule (bool, optional): Whether the system is a molecule (`True`) or a crystal (`False`). Default is `True`.

    Returns:
        dict: A dictionary containing the original structure, supercell, and marked structures. The dictionary keys
              are the structure identifiers, and the values are the corresponding ASE `Atoms` objects.

    Raises:
        AssertionError: If neither `atom_list` nor `element_list` is provided.
        ValueError: If hydrogen (H) is found in `element_list` or `atom_list`.
    """
    from workgraph_collections.ase.common.core_level import (
        find_non_equivalent_sites,
        mark_non_equivalent_sites,
        create_supercell,
    )

    # we need either atom_list or element_list
    assert (
        atom_list is not None or element_list is not None
    ), "atom_list or element_list must be provided."
    # step 1: find the non-equivalent sites
    if element_list:
        if "H" in element_list:
            raise ValueError("H is not allowed in element_list.")
        non_equivalent_sites = find_non_equivalent_sites(
            atoms, element_list, is_molecule
        )
    else:
        if "H" in [atoms[index].symbol for index in atom_list]:
            raise ValueError("H is not allowed in atom_list.")
        non_equivalent_sites = {index: {index} for index in atom_list}
    # step 1: create a supercell
    supercell = create_supercell(atoms, min_cell_length, is_molecule)
    structures = {"original": atoms.copy(), "supercell": supercell}
    # step 3: mark the non-equivalent sites
    structures.update(
        mark_non_equivalent_sites(supercell, non_equivalent_sites, marker)
    )
    return {"structures": structures}


@task()
def get_binding_energy(core_hole_pseudos: dict = None, **scf_outputs: dict):
    """
    Calculate the binding energy for each core level.

    The binding energy is calculated as the difference between the core hole SCF energy and the ground state energy,
    plus a correction value for each element. The keys of the input SCF outputs should be formatted as 'symbol_index'.

    Args:
        core_hole_pseudos (dict): A dictionary containing information about the core hole correction for each element.
                                  The structure is:
                                  {
                                      "Element": {
                                          "correction": float  # Correction energy to be added to binding energy
                                      }
                                  }
        scf_outputs (dict): Keyword arguments representing SCF outputs, where each key corresponds to a core hole and
                            the ground state, and each value is a dictionary with energy information.
                            Example structure:
                            {
                                "ground": {"energy": float},  # Ground state SCF energy
                                "C_1s_0": {"energy": float},  # Core hole SCF energy for C 1s, index 0
                                ...
                            }

    Returns:
        dict: A dictionary where the keys are the core level identifiers (e.g., 'C_0') and the values are the
              calculated binding energies.

    """
    output_ground = scf_outputs.pop("ground")
    results = {}
    for key, output in scf_outputs.items():
        # key is like "C_0"
        symbol, index = key.split("_")
        de = output["energy"] - output_ground["energy"]
        e = de + core_hole_pseudos[symbol]["correction"]
        results[key] = e
    return results
