from aiida_workgraph import WorkGraph, task
from ase import Atoms
from .pw import pw_calculator, dos_calculator, projwfc_calculator
from aiida import orm


@task.graph_builder()
def pdos_workgraph(
    atoms: Atoms = None,
    pw_command: str = "pw.x",
    dos_command: str = "dos.x",
    projwfc_command: str = "projwfc.x",
    inputs: dict = None,
    pseudopotentials: dict = None,
    pseudo_dir: str = ".",
    scf_parent_folder: orm.RemoteData = None,
    run_scf: bool = True,
    run_relax: bool = False,
):
    """Generate PdosWorkGraph."""
    inputs = {} if inputs is None else inputs
    # create workgraph
    wg = WorkGraph("PDOS")
    # -------- relax -----------
    if run_relax:
        relax_task = wg.tasks.new(
            "PythonJob",
            function=pw_calculator,
            name="relax",
            atoms=atoms,
            calculation="vc-relax",
            command=pw_command,
            pseudopotentials=pseudopotentials,
            pseudo_dir=pseudo_dir,
        )
        relax_inputs = inputs.get("relax", {})
        relax_task.set(relax_inputs)
        atoms = relax_task.outputs["atoms"]
    # -------- scf -----------
    if run_scf:
        scf_task = wg.tasks.new(
            "PythonJob",
            function=pw_calculator,
            name="scf",
            atoms=atoms,
            calculation="scf",
            command=pw_command,
            pseudopotentials=pseudopotentials,
            pseudo_dir=pseudo_dir,
        )
        scf_inputs = inputs.get("scf", {})
        scf_task.set(scf_inputs)
        scf_parent_folder = scf_task.outputs["remote_folder"]
    # -------- nscf -----------
    nscf_task = wg.tasks.new(
        "PythonJob",
        function=pw_calculator,
        name="nscf",
        atoms=atoms,
        calculation="nscf",
        command=pw_command,
        pseudopotentials=pseudopotentials,
        pseudo_dir=pseudo_dir,
        parent_folder=scf_parent_folder,
        parent_output_folder="out",
        parent_folder_name="out",
    )
    nscf_inputs = inputs.get("nscf", {})
    nscf_task.set(nscf_inputs)
    # -------- dos -----------
    dos_task = wg.tasks.new(
        "PythonJob",
        function=dos_calculator,
        name="dos",
        command=dos_command,
        parent_folder=nscf_task.outputs["remote_folder"],
        parent_output_folder="out",
        parent_folder_name="out",
    )
    dos_input = inputs.get("dos", {"input_data": {}})
    dos_input["input_data"].update({"outdir": "parent_folder"})
    dos_task.set(dos_input)
    # -------- projwfc -----------
    projwfc_task = wg.tasks.new(
        "PythonJob",
        function=projwfc_calculator,
        name="projwfc",
        command=projwfc_command,
        parent_folder=nscf_task.outputs["remote_folder"],
        parent_output_folder="out",
        parent_folder_name="out",
    )
    projwfc_input = inputs.get("projwfc", {"input_data": {}})
    projwfc_input["input_data"].update({"outdir": "parent_folder"})
    projwfc_task.set(projwfc_input)
    return wg
