from aiida_workgraph import task, WorkGraph
from workgraph_collections.ase.common.surface import get_adsorption_structure
from ase import Atoms
from typing import Dict


@task.graph_builder(
    outputs=[
        {"name": "parameters", "from": "context.parameters"},
        {"name": "structures", "from": "context.structures"},
    ]
)
def relax_structures(slabs, inputs):
    """Run the scf calculation for each atoms."""
    from aiida_workgraph import WorkGraph
    from workgraph_collections.ase.espresso.relax import relax_workgraph

    wg = WorkGraph()
    for key, atoms in slabs.items():
        scf = wg.tasks.new(
            relax_workgraph, name=f"relax_{key}", atoms=atoms, calculation="relax"
        )
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
        {"name": "parameters", "from": "relax_structures.parameters"},
        {"name": "structures", "from": "relax_structures.structures"},
    ]
)
def adsorption_workgraph(
    slab: Atoms = None,
    adsorbate: Atoms = None,
    distance: float = 2.0,
    positions: tuple = ("ontop", "bridge", "hollow"),
    command: str = "pw.x",
    computer: str = "localhost",
    pseudopotentials: dict = None,
    pseudo_dir: str = None,
    kpts: list = None,
    kspacing: float = None,
    input_data: dict = None,
    metadata: dict = None,
    relax_slab: bool = True,
    calc_adsorption_energy: bool = False,
    adsorbate_energy: float = None,
):
    """Workgraph for generating slabs and relax them.
    1. Relax the bulk structure.
    2. Generate slabs.
    3. Relax the slabs.
    """
    from workgraph_collections.ase.espresso.relax import relax_workgraph

    input_data = input_data or {}

    wg = WorkGraph("slabs")
    # -------- relax slab -----------
    if relax_slab:
        relax_slab_task = wg.tasks.new(
            relax_workgraph,
            name="relax_slab",
            command=command,
            input_data=input_data,
            pseudopotentials=pseudopotentials,
            pseudo_dir=pseudo_dir,
            atoms=slab,
            metadata=metadata,
            computer=computer,
            kpts=kpts,
            kspacing=kspacing,
        )
        slab = relax_slab_task.outputs["atoms"]
    # -------- generate slab with adsorbate -----------
    add_adsorbate_task = wg.tasks.new(
        get_adsorption_structure,
        name="add_adsorbate",
        slab=slab,
        adsorbate=adsorbate,
        distance=distance,
        positions=positions,
        computer=computer,
        metadata=metadata,
    )
    # -------- relax_structures -----------
    relax_structures_task = wg.tasks.new(
        relax_structures,
        name="relax_structures",
        structures=add_adsorbate_task.outputs["structures"],
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
    if calc_adsorption_energy:
        if adsorbate_energy is None:
            raise ValueError("adsorbate_energy must be provided")
        wg.tasks.new(
            get_surface_energy,
            name="get_surface_energy",
            bulk_parameters=relax_slab_task.outputs["parameters"],
            adsorbate_energy=adsorbate_energy,
            slab_parameters=relax_structures_task.outputs["parameters"],
        )
    return wg
