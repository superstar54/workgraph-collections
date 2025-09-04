from aiida_workgraph import task, spec
from ase import Atoms
from .pw import pw_calculator


@task()
def should_run_while(is_converged: bool):
    """Negate the convergence flag to determine if the workflow should continue."""
    return not is_converged


@task(
    outputs=spec.namespace(current_number_of_bands=int, is_converged=bool),
)
def inspect_relax(
    parameters: dict,
    atoms: Atoms,
    prev_atoms: Atoms = None,
    volume_threshold=0.1,
):
    """Inspect the results of the relaxation calculation to check for convergence."""
    current_number_of_bands = parameters["number_of_bands"]
    is_converged = False
    if prev_atoms is not None:
        prev_volume = prev_atoms.get_volume()
        volume_difference = abs(prev_volume - atoms.get_volume()) / prev_volume
        if volume_difference < volume_threshold:
            is_converged = True

    results = {
        "current_number_of_bands": current_number_of_bands,
        "is_converged": is_converged,
    }
    return results


@task.graph(
    outputs=spec.namespace(atoms=Atoms, parameters=dict),
)
def RelaxWorkgraph(
    atoms: Atoms = None,
    prev_atoms: Atoms = None,
    current_number_of_bands: int = None,
    is_converged: bool = False,
    number_of_iterations: int = 0,
    inputs: dict = None,
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
    output_parameters: dict = None,
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
    input_data = {} if input_data is None else input_data
    if not is_converged and number_of_iterations < max_iterations:
        relax_task_out = pw_calculator(
            atoms=atoms,
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
        # -------- inspect relax -----------
        inspect_relax_out = inspect_relax(
            parameters=relax_task_out.parameters,
            prev_atoms=atoms,
            atoms=relax_task_out.atoms,
            volume_threshold=volume_threshold,
        )
        return RelaxWorkgraph(
            atoms=relax_task_out.atoms,
            prev_atoms=atoms,
            current_number_of_bands=inspect_relax_out.current_number_of_bands,
            is_converged=inspect_relax_out.is_converged,
            number_of_iterations=number_of_iterations + 1,
            inputs=inputs,
            command=command,
            computer=computer,
            input_data=input_data,
            pseudopotentials=pseudopotentials,
            pseudo_dir=pseudo_dir,
            kpts=kpts,
            kspacing=kspacing,
            run_scf=run_scf,
            max_iterations=max_iterations,
            volume_threshold=volume_threshold,
            output_parameters=relax_task_out.parameters,
        )
    else:
        # Optionally add an SCF calculation after relaxation is complete
        if run_scf:
            scf_task_out = pw_calculator(
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
    return {
        "atoms": atoms,
        "parameters": scf_task_out.parameters if run_scf else output_parameters,
    }
