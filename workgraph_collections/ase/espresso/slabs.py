from aiida_workgraph import task, namespace, dynamic
from workgraph_collections.ase.common.surface import get_slabs_from_miller_indices_ase
from ase import Atoms
from typing import List, Annotated
from workgraph_collections.ase.espresso.relax import RelaxWorkgraph


@task.graph(outputs=namespace(parameters=dynamic(dict), structures=dynamic(Atoms)))
def RelaxSlabs(
    slabs: Annotated[dict, dynamic(Atoms)],
    inputs: Annotated[dict, RelaxWorkgraph.inputs.exclude("atoms")],
):
    """Run the scf calculation for each atoms."""

    parameters = {}
    structures = {}
    for key, atoms in slabs.items():
        scf_out = RelaxWorkgraph(atoms=atoms, **inputs)
        parameters[key] = scf_out.parameters
        structures[key] = scf_out.atoms
    return {"parameters": parameters, "structures": structures}


@task()
def get_surface_energy(
    bulk_atoms: Atoms,
    bulk_parameters: dict,
    slab_parameters: dynamic(dict),
    slab_structures: dynamic(Atoms),
):
    """Calculate the surface energy."""
    from ase.units import J, eV
    import numpy as np

    # Get the bulk energy
    bulk_energy = bulk_parameters["energy"]

    # Calculate the surface energy
    surface_energies = {}
    slab_energies = {}
    for key, slab_params in slab_parameters.items():
        slab_atoms = slab_structures[key]
        slab_energy = slab_params["energy"]
        slab_energies[key] = slab_energy
        # get the area of the slab in the xy plane
        area = np.linalg.norm(np.cross(slab_atoms.cell[0], slab_atoms.cell[1]))
        # calculate the surface energy in eV/A^2
        surface_energies[key] = (
            slab_energy - bulk_energy * len(slab_atoms) / len(bulk_atoms)
        ) / (2 * area)
        # convert to J/m^2
        surface_energies[key] *= eV / J * (10**20)

    return surface_energies


@task.graph(
    outputs=namespace(
        parameters=dynamic(dict), structures=dynamic(Atoms), surface_energy=dict
    )
)
def SlabsWorkgraph(
    atoms: Atoms = None,
    command: str = "pw.x",
    computer: str = "localhost",
    miller_indices: List[list] = None,
    layers: int = 3,
    vacuum: float = 5.0,
    pseudopotentials: dict = None,
    pseudo_dir: str = None,
    kpts: list = None,
    kspacing: float = None,
    input_data: dict = None,
    metadata: dict = None,
    calc_surface_energy: bool = False,
):
    """Workgraph for generating slabs and relax them.
    1. Relax the bulk structure.
    2. Generate slabs.
    3. Relax the slabs.
    """
    from workgraph_collections.ase.espresso.relax import RelaxWorkgraph

    input_data = input_data or {}

    # -------- relax bulk -----------
    relax_bulk_out = RelaxWorkgraph(
        command=command,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        pseudo_dir=pseudo_dir,
        atoms=atoms,
        metadata=metadata,
        computer=computer,
        kpts=kpts,
        kspacing=kspacing,
    )
    atoms = relax_bulk_out.atoms
    # -------- generate_slabs -----------
    generate_slabs_out = get_slabs_from_miller_indices_ase(
        atoms=atoms,
        indices=miller_indices,
        layers=layers,
        vacuum=vacuum,
    )
    # -------- relax_slabs -----------
    relax_slabs_out = RelaxSlabs(
        slabs=generate_slabs_out.slabs,
        inputs={
            "command": command,
            "input_data": input_data,
            "kpts": kpts,
            "kspacing": kspacing,
            "pseudopotentials": pseudopotentials,
            "pseudo_dir": pseudo_dir,
            "metadata": metadata,
            "computer": computer,
        },
    )
    if calc_surface_energy:
        surface_energy_out = get_surface_energy(
            bulk_atoms=atoms,
            bulk_parameters=relax_bulk_out.parameters,
            slab_parameters=relax_slabs_out.parameters,
            slab_structures=relax_slabs_out.structures,
        )
    return {
        "parameters": relax_slabs_out.parameters,
        "structures": relax_slabs_out.structures,
        "surface_energy": surface_energy_out.result,
    }
