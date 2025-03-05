from aiida_workgraph import task, WorkGraph
from workgraph_collections.ase.common.surface import get_slabs_from_miller_indices_ase
from ase import Atoms
from typing import List, Dict


@task.graph_builder(
    outputs=[
        {"name": "parameters", "from": "context.parameters"},
        {"name": "structures", "from": "context.structures"},
    ]
)
def relax_slabs(slabs, inputs):
    """Run the scf calculation for each atoms."""
    from aiida_workgraph import WorkGraph
    from workgraph_collections.ase.espresso.relax import relax_workgraph

    wg = WorkGraph()
    for key, atoms in slabs.items():
        scf = wg.add_task(relax_workgraph, name=f"relax_{key}", atoms=atoms)
        scf.set(inputs)
        scf.set_context(
            {f"parameters.{key}": "parameters", f"structures.{key}": "atoms"}
        )
    return wg


@task.pythonjob(
    inputs=[
        {
            "name": "slab_parameters",
            "identifier": "workgraph.namespace",
            "metadata": {"dynamic": True},
        },
        {
            "name": "slab_structures",
            "identifier": "workgraph.namespace",
            "metadata": {"dynamic": True},
        },
    ]
)
def get_surface_energy(
    bulk_atoms: Atoms,
    bulk_parameters: dict,
    slab_parameters: Dict[str, dict],
    slab_structures: Dict[str, dict],
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


@task.graph_builder(
    outputs=[
        {"name": "parameters", "from": "relax_slabs.parameters"},
        {"name": "structures", "from": "relax_slabs.structures"},
    ]
)
def slabs_workgraph(
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
    relax_bulk: bool = True,
    calc_surface_energy: bool = False,
):
    """Workgraph for generating slabs and relax them.
    1. Relax the bulk structure.
    2. Generate slabs.
    3. Relax the slabs.
    """
    from workgraph_collections.ase.espresso.relax import relax_workgraph

    input_data = input_data or {}

    wg = WorkGraph("slabs")
    # -------- relax bulk -----------
    if run_relax:
        relax_bulk_task = wg.add_task(
            relax_workgraph,
            name="relax_bulk",
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
        atoms = relax_bulk_task.outputs["atoms"]
    # -------- generate_slabs -----------
    generate_slabs_task = wg.add_task(
        get_slabs_from_miller_indices_ase,
        name="generate_slabs",
        atoms=atoms,
        indices=miller_indices,
        layers=layers,
        vacuum=vacuum,
        computer=computer,
        metadata=metadata,
    )
    # -------- relax_slabs -----------
    relax_slabs_task = wg.add_task(
        relax_slabs,
        name="relax_slabs",
        slabs=generate_slabs_task.outputs["slabs"],
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
        wg.tasks.new(
            get_surface_energy,
            name="get_surface_energy",
            bulk_atoms=atoms,
            bulk_parameters=relax_bulk_task.outputs["parameters"],
            slab_parameters=relax_slabs_task.outputs["parameters"],
            slab_structures=relax_slabs_task.outputs["structures"],
        )
    return wg
