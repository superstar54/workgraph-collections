from aiida_workgraph import WorkGraph, task
from ase import Atoms
from .pw import pw_calculator


@task()
def should_run_while(is_converged: bool):
    """Negate the convergence flag to determine if the workflow should continue."""
    return not is_converged


@task(
    outputs=[
        {"name": "current_number_of_bands"},
        {"name": "is_converged"},
        {"name": "current_atoms"},
    ]
)
def inspect_relax(
    parameters: dict,
    current_atoms: Atoms,
    prev_atoms: Atoms = None,
    volume_threshold=0.1,
):
    """Inspect the results of the relaxation calculation to check for convergence."""
    current_number_of_bands = parameters.get_dict()["number_of_bands"]
    is_converged = False
    if prev_atoms is not None:
        prev_volume = prev_atoms.value.get_volume()
        volume_difference = (
            abs(prev_volume - current_atoms.value.get_volume()) / prev_volume
        )
        if volume_difference < volume_threshold:
            is_converged = True

    results = {
        "current_number_of_bands": current_number_of_bands,
        "is_converged": is_converged,
        "current_atoms": current_atoms,
    }
    return results


@task.graph(
    outputs=[
        {"name": "atoms", "from": "relax.atoms"},
        {"name": "parameters", "from": "context.parameters"},
    ]
)
def relax_workgraph(
    atoms: Atoms = None,
    command: str = "pw.x",
    computer: str = "localhost",
    calculation: str = "vc-relax",
    input_data: dict = None,
    pseudopotentials: dict = None,
    pseudo_dir: str = ".",
    kpts: list = None,
    kspacing: float = None,
    run_scf: bool = True,
    max_iterations: int = 5,
    volume_threshold: float = 0.1,
    metadata: dict = None,
):
    """Construct a WorkGraph to relax a structure using Quantum ESPRESSO's pw.x.

    Parameters:
        atoms (ase.Atoms): Initial atomic structure to relax.
        command (str): Command to execute pw.x.
        inputs (dict): Additional inputs for the calculations (e.g., 'relax' and 'scf' inputs).
        pseudopotentials (dict): Mapping of elements to their pseudopotential files.
        pseudo_dir (str): Directory containing pseudopotential files.
        run_scf (bool): Whether to run an SCF calculation after relaxation.
        max_iterations (int): Maximum number of relaxation iterations.
        volume_threshold (float): Volume change threshold to determine convergence.

    Returns:
        WorkGraph: The constructed workgraph for the relaxation workflow.
    """
    from aiida_workgraph.orm.atoms import AtomsData

    input_data = {} if input_data is None else input_data
    # create workgraph
    wg = WorkGraph("BandsStructure")
    wg.context = {
        "current_atoms": atoms if isinstance(atoms, AtomsData) else AtomsData(atoms),
        "prev_atoms": None,
        "current_number_of_bands": None,
        "prev_cell_volume": None,
        "is_converged": False,
    }
    # Add a task to compare the convergence status
    should_run_while_task = wg.add_task(
        should_run_while, name="should_run_while", is_converged="{{is_converged}}"
    )
    # Create a while loop that continues until convergence or max_iterations is reached
    while_task = wg.add_task(
        "While",
        name="while",
        conditions=should_run_while_task.outputs.result,
        max_iterations=max_iterations,
    )
    relax_task = wg.add_task(
        pw_calculator,
        name="relax",
        atoms="{{current_atoms}}",
        command=command,
        computer=computer,
        metadata=metadata,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        pseudo_dir=pseudo_dir,
        kpts=kpts,
        kspacing=kspacing,
        calculation=calculation,
    )
    atoms = relax_task.outputs["atoms"]
    relax_task.set_context({"parameters": "parameters"})
    # -------- inspect relax -----------
    inspect_relax_task = wg.add_task(
        inspect_relax,
        name="inspect_relax",
        parameters=relax_task.outputs["parameters"],
        prev_atoms="{{current_atoms}}",
        current_atoms=relax_task.outputs["atoms"],
        volume_threshold=volume_threshold,
    )
    # Update context variables with outputs from inspect_relax_task
    inspect_relax_task.set_context(
        {
            "current_number_of_bands": "current_number_of_bands",
            "is_converged": "is_converged",
            "current_atoms": "current_atoms",
        }
    )
    while_task.children.add(["relax", inspect_relax_task])
    # Optionally add an SCF calculation after relaxation is complete
    if run_scf:
        scf_task = wg.add_task(
            pw_calculator,
            name="scf",
            atoms=atoms,
            command=command,
            computer=computer,
            metadata=metadata,
            input_data=input_data,
            pseudopotentials=pseudopotentials,
            pseudo_dir=pseudo_dir,
            calculation="scf",
            kpts=kpts,
            kspacing=kspacing,
        )
        scf_task.set_context({"parameters": "parameters"})

    return wg
