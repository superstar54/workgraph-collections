from aiida_workgraph import task, namespace
from ase import Atoms
from .pw import pw_calculator
from aiida import orm


@task()
def find_kpoints_path(
    atoms: Atoms, path: str = None, npoints: int = None, density: int = None
):
    """Find kpoint path for band structure calculation."""
    lat = atoms.cell.get_bravais_lattice()
    path = path or lat.special_path
    kpts = atoms.cell.bandpath(path, npoints=npoints, density=density)
    return kpts


@task.graph(
    outputs=namespace(
        bands=pw_calculator.outputs, kpoints_path=find_kpoints_path.outputs.result
    )
)
def BandsWorkgraph(
    atoms: Atoms = None,
    pw_command: str = "pw.x",
    inputs: dict = None,
    pseudopotentials: dict = None,
    pseudo_dir: str = ".",
    npoints: int = None,
    density: int = None,
    kpoints_path: str = None,
    scf_parent_folder: orm.RemoteData = None,
    run_scf: bool = True,
    run_relax: bool = False,
):
    """Generate BandsStructure WorkGraph."""
    inputs = {} if inputs is None else inputs
    if run_relax:
        relax_inputs = inputs.get("relax", {})
        relax_task_out = pw_calculator(
            atoms=atoms,
            command=pw_command,
            pseudopotentials=pseudopotentials,
            pseudo_dir=pseudo_dir,
            **relax_inputs
        )
        atoms = relax_task_out.atoms
    # -------- scf -----------
    if run_scf:
        scf_inputs = inputs.get("scf", {})
        scf_task_out = pw_calculator(
            atoms=atoms,
            command=pw_command,
            pseudopotentials=pseudopotentials,
            pseudo_dir=pseudo_dir,
            **scf_inputs
        )
        scf_parent_folder = scf_task_out.remote_folder
    # -------- kpoints path -----------
    find_kpoints_path_out = find_kpoints_path(
        atoms=atoms,
        path=kpoints_path,
        npoints=npoints,
        density=density,
    )
    # -------- bands -----------
    bands_inputs = inputs.get("bands", {})
    bands_task_out = pw_calculator(
        atoms=atoms,
        command=pw_command,
        pseudopotentials=pseudopotentials,
        pseudo_dir=pseudo_dir,
        kpts=find_kpoints_path_out.result,
        parent_folder=scf_parent_folder,
        parent_output_folder="out",
        parent_folder_name="out",
        **bands_inputs
    )
    return {"bands": bands_task_out, "kpoints_path": find_kpoints_path_out.result}
