from aiida_workgraph import task
from ase import Atoms
from .pw import pw_calculator
from .base import pp_calculator
from workgraph_collections.bader import bader_calculator


@task.graph()
def BaderWorkgraph(
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
    # -------- scf -----------
    scf_inputs = inputs.get("scf", {})
    scf_task_out = pw_calculator(
        atoms=atoms,
        command=pw_command,
        computer=computer,
        pseudopotentials=pseudopotentials,
        pseudo_dir=pseudo_dir,
        **scf_inputs
    )
    # -------- pp valence -----------
    pp_valence_inputs = inputs.get("pp_valence", {})
    pp_valence_inputs["input_data"] = {
        "INPUTPP": {"plot_num": 0},
        "PLOT": {
            "iflag": 3,
            "output_format": 6,
            "fileout": "charge_density.cube",
        },
    }
    pp_valence_out = pp_calculator(
        command=pp_command,
        computer=computer,
        parent_folder=scf_task_out.remote_folder,
        parent_output_folder="out",
        parent_folder_name="out",
        **pp_valence_inputs
    )
    # -------- pp all -----------

    pp_all_inputs = inputs.get("pp_all", {})
    pp_all_inputs["input_data"] = {
        "INPUTPP": {"plot_num": 21},
        "PLOT": {
            "iflag": 3,
            "output_format": 6,
            "fileout": "charge_density.cube",
        },
    }
    pp_all_out = pp_calculator(
        command=pp_command,
        computer=computer,
        parent_folder=scf_task_out.remote_folder,
        parent_output_folder="out",
        parent_folder_name="out",
        **pp_all_inputs
    )
    # -------- bader -----------
    bader_inputs = inputs.get("bader", {})
    bader_task_out = bader_calculator(
        computer=computer,
        command=bader_command,
        charge_density_folder="pp_valence_remote_folder",
        reference_charge_density_folder="pp_all_remote_folder",
        copy_files={
            "pp_valence_remote_folder": pp_valence_out.remote_folder,
            "pp_all_remote_folder": pp_all_out.remote_folder,
        },
        **bader_inputs
    )
    return bader_task_out.result
