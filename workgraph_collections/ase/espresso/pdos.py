from aiida_workgraph import WorkGraph, node
from ase import Atoms
from .base import pw_calculator, dos_calculator, projwfc_calculator
from aiida import orm


@node.graph_builder()
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
        relax_node = wg.nodes.new(
            pw_calculator,
            name="relax",
            atoms=atoms,
            calculation="vc-relax",
            command=pw_command,
            pseudopotentials=pseudopotentials,
            pseudo_dir=pseudo_dir,
            run_remotely=True,
        )
        relax_inputs = inputs.get("relax", {})
        relax_node.set(relax_inputs)
        atoms = relax_node.outputs["atoms"]
    # -------- scf -----------
    if run_scf:
        scf_node = wg.nodes.new(
            pw_calculator,
            name="scf",
            atoms=atoms,
            calculation="scf",
            command=pw_command,
            pseudopotentials=pseudopotentials,
            pseudo_dir=pseudo_dir,
            run_remotely=True,
        )
        scf_inputs = inputs.get("scf", {})
        scf_node.set(scf_inputs)
        scf_parent_folder = scf_node.outputs["remote_folder"]
    # -------- nscf -----------
    nscf_node = wg.nodes.new(
        pw_calculator,
        name="nscf",
        atoms=atoms,
        calculation="nscf",
        command=pw_command,
        pseudopotentials=pseudopotentials,
        pseudo_dir=pseudo_dir,
        parent_folder=scf_parent_folder,
        parent_output_folder="out",
        parent_folder_name="out",
        run_remotely=True,
    )
    nscf_inputs = inputs.get("nscf", {})
    nscf_node.set(nscf_inputs)
    # -------- dos -----------
    dos_node = wg.nodes.new(
        dos_calculator,
        name="dos",
        command=dos_command,
        parent_folder=nscf_node.outputs["remote_folder"],
        parent_output_folder="out",
        parent_folder_name="out",
        run_remotely=True,
    )
    dos_input = inputs.get("dos", {"input_data": {}})
    dos_input["input_data"].update({"outdir": "parent_folder"})
    dos_node.set(dos_input)
    # -------- projwfc -----------
    projwfc_node = wg.nodes.new(
        projwfc_calculator,
        name="projwfc",
        command=projwfc_command,
        parent_folder=nscf_node.outputs["remote_folder"],
        parent_output_folder="out",
        parent_folder_name="out",
        run_remotely=True,
    )
    projwfc_input = inputs.get("projwfc", {"input_data": {}})
    projwfc_input["input_data"].update({"outdir": "parent_folder"})
    projwfc_node.set(projwfc_input)
    return wg
