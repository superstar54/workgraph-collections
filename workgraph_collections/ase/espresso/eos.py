from aiida_workgraph import task, WorkGraph, active_map_zone
from workgraph_collections.ase.common.eos import generate_scaled_atoms, fit_eos
from ase import Atoms


@task.graph(outputs=[{"name": "result", "from": "fit_eos.result"}])
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
    from .pw import pw_calculator
    from copy import deepcopy

    input_data = input_data or {}

    with WorkGraph("EOS") as wg:
        # -------- relax -----------
        if run_relax:
            relax_input_data = deepcopy(input_data)
            relax_input_data.setdefault("CONTROL", {})
            relax_input_data["CONTROL"]["calculation"] = "vc-relax"
            relax_output = pw_calculator(
                atoms=atoms,
                metadata=metadata,
                computer=computer,
                command=command,
                input_data=relax_input_data,
                kpts=kpts,
                pseudopotentials=pseudopotentials,
                pseudo_dir=pseudo_dir,
            )
            atoms = relax_output.atoms
        # -------- scale_atoms -----------
        scale_output = generate_scaled_atoms(atoms=atoms, scales=scales)
        with active_map_zone(scale_output.scaled_atoms) as map_zone:
            scf_output = pw_calculator(
                atoms=map_zone.item,
                command=command,
                input_data=input_data,
                kpts=kpts,
                pseudopotentials=pseudopotentials,
                pseudo_dir=pseudo_dir,
                metadata=metadata,
                computer=computer,
            )
        # -------- fit_eos -----------
        fit_eos(
            volumes=scale_output.volumes,
            scf_results=scf_output.parameters,
        )
        return wg
