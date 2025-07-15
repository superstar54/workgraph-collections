from aiida_workgraph import WorkGraph, task
from ase import Atoms
from .pw import pw_calculator
from aiida import orm


@task()
def find_kpoint_path(
    atoms: Atoms, path: str = None, npoints: int = None, density: int = None
):
    """Find kpoint path for band structure calculation."""
    lat = atoms.cell.get_bravais_lattice()
    path = path or lat.special_path
    kpts = atoms.cell.bandpath(path, npoints=npoints, density=density)
    return kpts


@task.graph()
def bands_workgraph(
    atoms: Atoms = None,
    pw_command: str = "pw.x",
    inputs: dict = None,
    pseudopotentials: dict = None,
    pseudo_dir: str = ".",
    nkpoints: int = None,
    density: int = None,
    kpoints_path: str = None,
    scf_parent_folder: orm.RemoteData = None,
    run_scf: bool = True,
    run_relax: bool = False,
):
    """Generate BandsStructure WorkGraph."""
    inputs = {} if inputs is None else inputs
    # create workgraph
    wg = WorkGraph("BandsStructure")
    wg.context = {
        "current_number_of_bands": None,
    }
    # -------- relax -----------
    if run_relax:
        relax_task = wg.add_task(
            "workgraph.pythonjob",
            function=pw_calculator,
            name="relax",
            atoms=atoms,
            command=pw_command,
            pseudopotentials=pseudopotentials,
            pseudo_dir=pseudo_dir,
        )
        relax_inputs = inputs.get("relax", {})
        relax_task.set(relax_inputs)
        atoms = relax_task.outputs["atoms"]
    # -------- scf -----------
    if run_scf:
        scf_task = wg.add_task(
            "workgraph.pythonjob",
            function=pw_calculator,
            name="scf",
            atoms=atoms,
            command=pw_command,
            pseudopotentials=pseudopotentials,
            pseudo_dir=pseudo_dir,
        )
        scf_inputs = inputs.get("scf", {})
        scf_task.set(scf_inputs)
        scf_parent_folder = scf_task.outputs["remote_folder"]
    # -------- kpoints path -----------
    find_kpoints_path_task = wg.add_task(
        "workgraph.pythonjob",
        function=find_kpoint_path,
        name="find_kponits_path",
        atoms=atoms,
        path=kpoints_path,
        npoints=nkpoints,
        density=density,
    )
    find_kpoints_path_inputs = inputs.get("find_kpoints_path", {})
    find_kpoints_path_task.set(find_kpoints_path_inputs)
    # -------- bands -----------
    bands_task = wg.add_task(
        "workgraph.pythonjob",
        function=pw_calculator,
        name="bands",
        atoms=atoms,
        command=pw_command,
        pseudopotentials=pseudopotentials,
        pseudo_dir=pseudo_dir,
        kpts=find_kpoints_path_task.outputs.result,
        parent_folder=scf_parent_folder,
        parent_output_folder="out",
        parent_folder_name="out",
    )
    bands_inputs = inputs.get("bands", {})
    bands_task.set(bands_inputs)
    return wg
