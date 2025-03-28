from aiida_workgraph import WorkGraph, task
from ase import Atoms
from .pw import pw_calculator
from .base import dos_calculator, projwfc_calculator
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
    with WorkGraph("PDOS") as wg:
        # -------- relax -----------
        if run_relax:
            relax_inputs = inputs.get("relax", {})
            relax_output = pw_calculator(
                command=pw_command,
                atoms=atoms,
                calculation="vc-relax",
                pseudopotentials=pseudopotentials,
                pseudo_dir=pseudo_dir,
                **relax_inputs,
            )
            atoms = relax_output.atoms
        # -------- scf -----------
        if run_scf:
            scf_inputs = inputs.get("scf", {})
            scf_output = pw_calculator(
                command=pw_command,
                atoms=atoms,
                calculation="scf",
                pseudopotentials=pseudopotentials,
                pseudo_dir=pseudo_dir,
                **scf_inputs,
            )
            scf_parent_folder = scf_output.remote_folder
        # -------- nscf -----------
        nscf_inputs = inputs.get("nscf", {})
        nscf_output = pw_calculator(
            command=pw_command,
            atoms=atoms,
            calculation="nscf",
            pseudopotentials=pseudopotentials,
            pseudo_dir=pseudo_dir,
            parent_folder=scf_parent_folder,
            parent_output_folder="out",
            parent_folder_name="out",
            **nscf_inputs,
        )
        # -------- dos -----------
        dos_input = inputs.get("dos", {"input_data": {}})
        dos_calculator(
            command=dos_command,
            parent_folder=nscf_output.remote_folder,
            parent_output_folder="out",
            parent_folder_name="out",
            **dos_input,
        )
        # -------- projwfc -----------
        projwfc_input = inputs.get("projwfc", {"input_data": {}})
        projwfc_calculator(
            command=projwfc_command,
            parent_folder=nscf_output.remote_folder,
            parent_output_folder="out",
            parent_folder_name="out",
            **projwfc_input,
        )
        return wg
