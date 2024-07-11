from aiida_workgraph import task, WorkGraph
from workgraph_collections.ase.common.eos import generate_scaled_atoms, fit_eos
from ase import Atoms


@task.graph_builder(outputs=[{"name": "scf_results", "from": "context.results"}])
def all_scf(scaled_atoms, scf_inputs):
    """Run the scf calculation for each atoms."""
    from aiida_workgraph import WorkGraph
    from .base import pw_calculator

    wg = WorkGraph()
    for key, atoms in scaled_atoms.items():
        scf = wg.tasks.new(
            "PythonJob", function=pw_calculator, name=f"scf_{key}", atoms=atoms
        )
        scf.set(scf_inputs)
        # save the output parameters to the context
        scf.to_context = [["results", f"results.{key}"]]
    return wg


@task.graph_builder(outputs=[{"name": "result", "from": "fit_eos.result"}])
def eos_workgraph(
    atoms: Atoms = None,
    command: str = "pw.x",
    computer: str = "localhost",
    scales: list = None,
    pseudopotentials: dict = None,
    pseudo_dir: str = None,
    kpts: list = None,
    input_data: dict = None,
    metadata: dict = None,
    run_relax: bool = True,
):
    """Workgraph for EOS calculation.
    1. Get the scaled atoms.
    2. Run the SCF calculation for each scaled atoms.
    3. Fit the EOS.
    """
    from .base import pw_calculator
    from copy import deepcopy

    input_data = input_data or {}

    wg = WorkGraph("EOS")
    # -------- relax -----------
    if run_relax:
        relax_task = wg.tasks.new(
            "PythonJob",
            function=pw_calculator,
            name="relax",
            atoms=atoms,
            metadata=metadata,
            computer=computer,
        )
        relax_input_data = deepcopy(input_data)
        relax_input_data.setdefault("CONTROL", {})
        relax_input_data["CONTROL"]["calculation"] = "vc-relax"
        relax_task.set(
            {
                "command": command,
                "input_data": relax_input_data,
                "kpts": kpts,
                "pseudopotentials": pseudopotentials,
                "pseudo_dir": pseudo_dir,
            }
        )
        atoms = relax_task.outputs["atoms"]
    # -------- scale_atoms -----------
    scale_atoms_task = wg.tasks.new(
        "PythonJob",
        function=generate_scaled_atoms,
        name="scale_atoms",
        atoms=atoms,
        scales=scales,
        computer=computer,
        metadata=metadata,
    )
    # -------- all_scf -----------
    all_scf1 = wg.tasks.new(
        all_scf,
        name="all_scf",
        scaled_atoms=scale_atoms_task.outputs["scaled_atoms"],
        scf_inputs={
            "command": command,
            "input_data": input_data,
            "kpts": kpts,
            "pseudopotentials": pseudopotentials,
            "pseudo_dir": pseudo_dir,
            "metadata": metadata,
            "computer": computer,
        },
    )
    # -------- fit_eos -----------
    wg.tasks.new(
        "PythonJob",
        function=fit_eos,
        name="fit_eos",
        volumes=scale_atoms_task.outputs["volumes"],
        scf_results=all_scf1.outputs["scf_results"],
        computer=computer,
        metadata=metadata,
    )
    return wg
