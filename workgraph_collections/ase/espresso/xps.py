from aiida_workgraph import task, WorkGraph
from workgraph_collections.ase.common.core_level import (
    get_marked_structures,
    get_binding_energy,
)
from workgraph_collections.ase.espresso import pw_calculator
from ase import Atoms
from copy import deepcopy


@task.graph_builder(outputs=[{"name": "results", "from": "context.scf"}])
def run_scf(
    marked_atoms: dict,
    command: str = None,
    computer: str = None,
    input_data: dict = None,
    kpts: list = None,
    pseudopotentials: dict = None,
    pseudo_dir: str = None,
    core_hole_pseudos: dict = None,
    core_hole_treatment: str = "XCH_SMEAR",
    is_molecule: bool = None,
    metadata: dict = None,
) -> WorkGraph:
    """Run the scf calculation for each atoms."""
    from aiida_workgraph import WorkGraph
    from .pw import pw_calculator

    wg = WorkGraph("XPS")
    # run the ground state calculation for the supercell
    scf_ground = wg.add_task(
        "PythonJob",
        function=pw_calculator,
        name="ground",
        atoms=marked_atoms.pop("supercell"),
        computer=computer,
        metadata=metadata,
    )
    # update pseudopotentials using ground state pseudopotentials
    for key, value in core_hole_pseudos.items():
        pseudopotentials[key] = value["ground"]
    scf_ground.set(
        {
            "command": command,
            "input_data": input_data,
            "kpts": kpts,
            "pseudopotentials": pseudopotentials,
            "pseudo_dir": pseudo_dir,
        }
    )
    scf_ground.set_context({"scf.ground": "parameters"})
    # remove the original atoms
    marked_atoms.pop("original", None)
    for key, atoms in marked_atoms.items():
        scf = wg.add_task(
            "PythonJob",
            function=pw_calculator,
            name=f"scf_{key}",
            atoms=atoms,
            computer=computer,
            metadata=metadata,
        )
        # update pseudopotentials based on marked atoms
        # split key by last underscore
        label, _index = key.rsplit("_", 1)
        pseudopotentials["X"] = core_hole_pseudos[label]["core_hole"]
        # update the input data based on the core hole treatment
        input_data.setdefault("SYSTEM", {})
        if is_molecule:
            print("is_molecule: ", is_molecule)
            input_data["SYSTEM"]["assume_isolated"] = "mt"
            kpts = None  # set gamma only
            core_hole_treatment = "FULL"
        if core_hole_treatment.upper() == "XCH_SMEAR":
            input_data["SYSTEM"].update(
                {
                    "occupations": "smearing",
                    "tot_charge": 0,
                    "nspin": 2,
                    "starting_magnetization(1)": 0,
                }
            )
        elif core_hole_treatment.upper() == "XCH_FIXED":
            input_data["SYSTEM"].update(
                {
                    "occupations": "fixed",
                    "tot_charge": 0,
                    "nspin": 2,
                    "tot_magnetization": 1,
                }
            )
        elif core_hole_treatment.upper() == "FULL":
            input_data["SYSTEM"].update(
                {
                    "tot_charge": 1,
                }
            )
        scf.set(
            {
                "command": command,
                "input_data": input_data,
                "kpts": kpts,
                "pseudopotentials": pseudopotentials,
                "pseudo_dir": pseudo_dir,
            }
        )
        # save the output parameters to the context
        scf.set_context({f"scf.{key}": "parameters"})
    return wg


@task.graph_builder(outputs=[{"name": "result", "from": "binding_energy.result"}])
def xps_workgraph(
    atoms: Atoms = None,
    scf_inputs: str = None,
    marked_structures_inputs: dict = None,
    core_hole_pseudos: dict = None,
    metadata: dict = None,
    run_relax: bool = False,
):
    """Workgraph for XPS calculation.
    1. Get the marked atoms.
    2. Run the SCF calculation for for ground state, and each marked atoms
    with core hole pseudopotentials.
    3. Calculate the binding energy.
    """
    from ase.io.espresso import Namelist

    scf_inputs = scf_inputs or {}
    marked_structures_inputs = marked_structures_inputs or {}

    wg = WorkGraph("XPS")
    # -------- relax -----------
    if run_relax:
        relax_task = wg.add_task(
            "PythonJob",
            function=pw_calculator,
            name="relax",
            atoms=atoms,
        )
        relax_inputs = deepcopy(scf_inputs)
        input_data = Namelist(relax_inputs.get("input_data", {})).to_nested(binary="pw")
        input_data["CONTROL"]["calculation"] = "relax"
        relax_inputs["input_data"] = input_data
        relax_task.set(relax_inputs)
        atoms = relax_task.outputs["atoms"]
    # -------- get_marked_atoms -----------
    marked_atoms_task = wg.add_task(
        "PythonJob",
        function=get_marked_structures,
        name="marked_atoms",
        atoms=atoms,
        metadata=metadata,
    )
    marked_atoms_task.set(marked_structures_inputs)
    # ------------------ run scf -------------------
    run_scf_task = wg.add_task(
        run_scf,
        name="run_scf",
        marked_atoms=marked_atoms_task.outputs.structures,
        core_hole_pseudos=core_hole_pseudos,
        is_molecule=marked_structures_inputs.get("is_molecule", False),
    )
    run_scf_task.set(scf_inputs)
    # -------- calculate binding energy -----------
    wg.add_task(
        "PythonJob",
        function=get_binding_energy,
        name="get_binding_energy",
        core_hole_pseudos=core_hole_pseudos,
        scf_outputs=run_scf_task.outputs["results"],
        metadata=metadata,
    )
    return wg
