from aiida_workgraph import task, WorkGraph
from workgraph_collections.ase.common.surface import get_slabs_from_miller_indices_ase
from ase import Atoms
from typing import List


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
        scf = wg.tasks.new(relax_workgraph, name=f"relax_{key}", atoms=atoms)
        scf.set(inputs)
        scf.set_context(
            {"parameters": f"parameters.{key}", "atoms": f"structures.{key}"}
        )
    return wg


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
    run_relax: bool = True,
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
        relax_task = wg.tasks.new(
            relax_workgraph,
            name="relax",
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
        atoms = relax_task.outputs["atoms"]
    # -------- generate_slabs -----------
    generate_slabs_task = wg.tasks.new(
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
    wg.tasks.new(
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
    return wg
