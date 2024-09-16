from aiida_workgraph import task
from typing import List


@task.pythonjob()
def get_slab_from_miller_indices_ase(
    atoms,
    indices: List[int],
    layers: int = 1,
    vacuum: float = 5.0,
    tol: float = 1e-5,
    periodic: bool = True,
    center_slab: bool = True,
):
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


@task.pythonjob(outputs=[{"name": "slabs", "identifier": "workgraph.namespace"}])
def get_slabs_from_miller_indices_ase(
    atoms,
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


@task.pythonjob()
def get_slab_from_miller_indices_pymatgen(
    atoms,
    miller_index,
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
