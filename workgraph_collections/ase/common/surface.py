from aiida_workgraph import task, spec
from typing import List
from ase import Atoms


def get_slab_from_miller_indices_ase(
    atoms: Atoms,
    indices: List[int],
    layers: int = 1,
    vacuum: float = 5.0,
    tol: float = 1e-5,
    periodic: bool = True,
    center_slab: bool = True,
) -> Atoms:
    """Generate a slab from a bulk structure using ASE's surface module."""
    from ase.build import surface

    slab = surface(
        atoms, indices, layers=layers, vacuum=vacuum, tol=tol, periodic=periodic
    )
    slab.info["slab_info"] = {
        "indices": indices,
        "layers": layers,
        "vacuum": vacuum,
        "tol": tol,
    }
    if not center_slab:
        slab.positions[:, 2] -= vacuum
    return slab


@task(outputs=spec.namespace(slabs=spec.dynamic(Atoms)))
def get_slabs_from_miller_indices_ase(
    atoms: Atoms,
    indices: List[List[int]],
    layers: int = 1,
    vacuum: float = 5.0,
    tol: float = 1e-5,
    periodic: bool = True,
    center_slab: bool = True,
):
    """Generate multiple slabs from a bulk structure using ASE's surface module"""
    slabs = {}
    for index in indices:
        slab = get_slab_from_miller_indices_ase(
            atoms,
            index,
            layers=layers,
            vacuum=vacuum,
            tol=tol,
            periodic=periodic,
            center_slab=center_slab,
        )
        slabs["slab" + "".join(map(str, index)).replace("-", "m")] = slab
    return {"slabs": slabs}


@task()
def get_slab_from_miller_indices_pymatgen(
    atoms: Atoms,
    miller_index: List[int],
    min_slab_size: float = 1.0,
    min_vacuum_size: float = 5.0,
    center_slab: bool = False,
    max_normal_search: int | None = None,
    in_unit_planes: bool = False,
):
    from pymatgen.core.surface import SlabGenerator
    from pymatgen.io.ase import AseAtomsAdaptor

    slabs_with_info = {}
    structure = AseAtomsAdaptor.get_structure(atoms)
    gen = SlabGenerator(
        structure,
        miller_index,
        min_slab_size,
        min_vacuum_size,
        center_slab=center_slab,
        max_normal_search=max_normal_search,
        in_unit_planes=in_unit_planes,
    )
    slabs = gen.get_slabs()
    for slab in slabs:
        slabs_with_info = slab.to_ase_atoms()
        slab_info = {
            "miller_index": miller_index,
            "shift": round(slab.shift, 3),
            "scale_factor": slab.scale_factor,
        }
        slabs_with_info.info["slab_info"] = slab_info
    return slabs_with_info


@task(outputs=spec.namespace(structures=spec.dynamic(Atoms)))
def get_adsorption_structure(
    slab: Atoms,
    adsorbate: Atoms,
    distance: float = 2.0,
    put_inside: bool = True,
    symm_reduce: float = 1e-2,
    near_reduce: float = 1e-2,
    positions: tuple = ("ontop", "bridge", "hollow"),
):
    """Generate adsorption structures on a slab using pymatgen's AdsorbateSiteFinder."""
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    from pymatgen.io.ase import AseAtomsAdaptor

    slab = AseAtomsAdaptor.get_structure(slab)
    adsorbate = AseAtomsAdaptor.get_molecule(adsorbate)
    ads_site_finder = AdsorbateSiteFinder(slab)
    ads_sites = ads_site_finder.find_adsorption_sites(
        distance=distance,
        put_inside=put_inside,
        symm_reduce=symm_reduce,
        near_reduce=near_reduce,
        positions=positions,
    )
    structures = {}
    for position, coords in ads_sites.items():
        # skip the "all" key
        if position == "all":
            continue
        count = 0
        for coord in coords:
            structure = ads_site_finder.add_adsorbate(adsorbate, coord)
            atoms = structure.to_ase_atoms()
            atoms.info["ads_info"] = {
                "distance": distance,
                "position": position,
            }
            structures[f"{position}_{count}"] = atoms
            count += 1
    return {"structures": structures}


def add_adsorbate_to_nanoparticle(
    atoms: Atoms,
    adsorbate: Atoms,
    index: int,
    distance=2.0,
    cutoff: float = 3.0,
    surface_atom_indices: list = None,
):
    """Add an adsorbate to a nanoparticle surface.
    Ensure the adsorbate molecule points out along the normal to the surface.
    """
    import numpy as np

    surface_atom_indices = surface_atom_indices or []
    # Get distances to find neighbors around the surface atom
    distances = atoms.get_distances(index, indices=None, mic=True)
    neighbor_indices = np.where(distances < cutoff)[0]
    # remove the surface atom itself
    neighbor_indices = [i for i in neighbor_indices if i != index]
    if surface_atom_indices:
        neighbor_indices = [i for i in neighbor_indices if i in surface_atom_indices]
    neighbor_vectors = atoms.positions[neighbor_indices] - atoms.positions[index]
    # normalize the vectors, and the shorter the distance, the stronger the force
    neighbor_vectors *= (
        cutoff / (np.linalg.norm(neighbor_vectors, axis=1)[:, np.newaxis]) ** 2
    )
    # Estimate surface normal from the average of the neighbor vectors
    if len(neighbor_vectors) >= 1:
        surface_normal = np.sum(neighbor_vectors, axis=0)
    else:
        surface_normal = np.array([0, 0, 1])
    mol = adsorbate.copy()
    surface_normal /= -np.linalg.norm(surface_normal)
    # Rotation matrix to align the molecule along the surface normal
    axis = np.array([0, 0, 1])  # CO molecule initially aligned along z-axis
    angle = np.arccos(np.dot(axis, surface_normal)) * 180 / np.pi
    rotation_axis = np.cross(axis, surface_normal)  # Axis to rotate around
    if np.linalg.norm(rotation_axis) > 0:  # Avoid dividing by zero
        rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize the rotation axis
        mol.rotate(angle, v=rotation_axis, center=(0, 0, 0))  # Rotate molecule

    # Translate the molecule along the surface normal
    offset = distance * surface_normal
    # Translate the molecule to the surface atom's position, offset by the normal
    mol.translate(atoms.positions[index] + offset)
    atoms += mol

    return atoms
