from aiida_workgraph import WorkGraph, node
from ase import Atoms
from .base import pw_calculator


@node()
def find_kpoint_path(
    atoms: Atoms, path: str = None, npoints: int = None, density: int = None
):
    """Find kpoint path for band structure calculation."""
    lat = atoms.cell.get_bravais_lattice()
    path = path or lat.special_path
    kpts = atoms.cell.bandpath(path, npoints=npoints, density=density)
    return kpts


@node.graph_builder()
def bands_workgraph(
    atoms: Atoms = None,
    pw_command: str = "pw.x",
    inputs: dict = None,
    pseudopotentials: dict = None,
    pseudo_dir: str = ".",
    nkpoints: int = None,
    density: int = None,
    kpoints_path: str = None,
):
    """Generate BandsStructure WorkGraph."""
    inputs = {} if inputs is None else inputs
    # create workgraph
    wg = WorkGraph("BandsStructure")
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
    # -------- kpoints path -----------
    find_kpoints_path_node = wg.nodes.new(
        find_kpoint_path,
        name="find_kponits_path",
        atoms=atoms,
        path=kpoints_path,
        npoints=nkpoints,
        density=density,
        run_remotely=True,
    )
    find_kpoints_path_inputs = inputs.get("find_kpoints_path", {})
    find_kpoints_path_node.set(find_kpoints_path_inputs)
    # -------- bands -----------
    bands_node = wg.nodes.new(
        pw_calculator,
        name="bands",
        atoms=atoms,
        command=pw_command,
        pseudopotentials=pseudopotentials,
        pseudo_dir=pseudo_dir,
        kpts=find_kpoints_path_node.outputs["result"],
        parent_folder=scf_node.outputs["remote_folder"],
        parent_output_folder="out",
        parent_folder_name="out",
        run_remotely=True,
    )
    bands_inputs = inputs.get("bands", {})
    bands_node.set(bands_inputs)
    return wg
