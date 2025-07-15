from aiida_workgraph import WorkGraph, task
from ase import Atoms
from .pw import pw_calculator, pp_calculator
from workgraph_collections.bader import bader_calculator


@task.graph()
def bader_workgraph(
    atoms: Atoms = None,
    pw_command: str = "pw.x",
    pp_command: str = "dos.x",
    bader_command: str = "bader.x",
    computer: str = "localhost",
    inputs: dict = None,
    pseudopotentials: dict = None,
    pseudo_dir: str = ".",
):
    """Generate PdosWorkGraph."""
    inputs = {} if inputs is None else inputs
    # create workgraph
    wg = WorkGraph("Bader")
    wg.context = {
        "current_number_of_bands": None,
    }
    # -------- scf -----------
    scf_task = wg.add_task(
        "workgraph.pythonjob",
        function=pw_calculator,
        name="scf",
        atoms=atoms,
        command=pw_command,
        computer=computer,
        pseudopotentials=pseudopotentials,
        pseudo_dir=pseudo_dir,
    )
    scf_inputs = inputs.get("scf", {})
    scf_task.set(scf_inputs)
    # -------- pp valence -----------
    pp_valence = wg.add_task(
        "workgraph.pythonjob",
        function=pp_calculator,
        name="pp_valence",
        command=pp_command,
        computer=computer,
        parent_folder=scf_task.outputs["remote_folder"],
        parent_output_folder="out",
        parent_folder_name="out",
    )
    pp_valence_inputs = inputs.get("pp_valence", {})
    pp_valence_inputs["input_data"] = {
        "INPUTPP": {"plot_num": 0},
        "PLOT": {
            "iflag": 3,
            "output_format": 6,
            "fileout": "charge_density.cube",
        },
    }
    pp_valence.set(pp_valence_inputs)
    # -------- pp all -----------
    pp_all = wg.add_task(
        "workgraph.pythonjob",
        function=pp_calculator,
        name="pp_all",
        command=pp_command,
        computer=computer,
        parent_folder=scf_task.outputs["remote_folder"],
        parent_output_folder="out",
        parent_folder_name="out",
    )
    pp_all_inputs = inputs.get("pp_all", {})
    pp_all_inputs["input_data"] = {
        "INPUTPP": {"plot_num": 21},
        "PLOT": {
            "iflag": 3,
            "output_format": 6,
            "fileout": "charge_density.cube",
        },
    }
    pp_all.set(pp_all_inputs)
    # -------- bader -----------
    bader_task = wg.add_task(
        "workgraph.pythonjob",
        function=bader_calculator,
        name="bader",
        computer=computer,
        command=bader_command,
        charge_density_folder="pp_valence_remote_folder",
        reference_charge_density_folder="pp_all_remote_folder",
    )
    wg.links.new(pp_valence.outputs["remote_folder"], bader_task.inputs["copy_files"])
    wg.links.new(pp_all.outputs["remote_folder"], bader_task.inputs["copy_files"])
    bader_inputs = inputs.get("bader", {})
    bader_task.set(bader_inputs)
    return wg
