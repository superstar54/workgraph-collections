from aiida_workgraph import task, spec
from workgraph_collections.ase.common.eos import generate_scaled_atoms, fit_eos
from ase import Atoms


@task.graph()
def calc_all_structures(
    structures: spec.dynamic(Atoms),
    command: str = "pw.x",
    input_data: dict = None,
    kpts: list = None,
    pseudopotentials: dict = None,
    pseudo_dir: str = None,
    metadata: dict = None,
    computer: str = "localhost",
) -> spec.namespace(results=spec.dynamic(dict)):
    """Run SCF calculations for all structures."""
    from .pw import pw_calculator

    input_data = input_data or {}
    results = {}
    for name, atoms in structures.items():
        scf_output = pw_calculator(
            atoms=atoms,
            command=command,
            input_data=input_data,
            kpts=kpts,
            pseudopotentials=pseudopotentials,
            pseudo_dir=pseudo_dir,
            metadata=metadata,
            computer=computer,
        )
        results[name] = scf_output.parameters

    return {"results": results}


@task.graph()
def EosWorkGraph(
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

    scf_outputs = calc_all_structures(
        structures=scale_output.scaled_atoms,
        command=command,
        input_data=input_data,
        kpts=kpts,
        pseudopotentials=pseudopotentials,
        pseudo_dir=pseudo_dir,
        metadata=metadata,
        computer=computer,
    )

    # -------- fit_eos -----------
    return fit_eos(
        volumes=scale_output.volumes,
        scf_results=scf_outputs.results,
    ).result
