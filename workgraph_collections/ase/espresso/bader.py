from aiida_workgraph import WorkGraph, node
from ase import Atoms
from .base import pw_calculator, pp_calculator
from workgraph_collections.bader import bader_calculator


@node.graph_builder()
def bader_workgraph(
    atoms: Atoms = None,
    pw_command: str = "pw.x",
    pp_command: str = "dos.x",
    bader_command: str = "bader.x",
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
    scf_node = wg.nodes.new(
        pw_calculator,
        name="scf",
        atoms=atoms,
        command=pw_command,
        pseudopotentials=pseudopotentials,
        pseudo_dir=pseudo_dir,
        run_remotely=True,
    )
    scf_inputs = inputs.get("scf", {})
    scf_node.set(scf_inputs)
    # -------- pp valence -----------
    pp_valence = wg.nodes.new(
        pp_calculator,
        name="pp_valence",
        command=pp_command,
        parent_folder=scf_node.outputs["remote_folder"],
        parent_output_folder="out",
        parent_folder_name="out",
        run_remotely=True,
    )
    pp_valence_inputs = inputs.get("pp_valence", {})
    pp_valence.set(pp_valence_inputs)
    # -------- pp all -----------
    pp_all = wg.nodes.new(
        pp_calculator,
        name="pp_all",
        command=pp_command,
        parent_folder=scf_node.outputs["remote_folder"],
        parent_output_folder="out",
        parent_folder_name="out",
        run_remotely=True,
    )
    pp_all_inputs = inputs.get("pp_all", {})
    pp_all.set(pp_all_inputs)
    # -------- bader -----------
    bader_node = wg.nodes.new(
        bader_calculator,
        name="bader",
        command=bader_command,
        charge_density_folder="pp_valence_remote_folder",
        reference_charge_density_folder="pp_all_remote_folder",
        run_remotely=True,
    )
    wg.links.new(pp_valence.outputs["remote_folder"], bader_node.inputs["copy_files"])
    wg.links.new(pp_all.outputs["remote_folder"], bader_node.inputs["copy_files"])
    bader_inputs = inputs.get("bader", {})
    bader_node.set(bader_inputs)
    return wg
