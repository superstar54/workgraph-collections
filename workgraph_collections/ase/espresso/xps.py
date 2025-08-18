from aiida_workgraph import task, dynamic
from workgraph_collections.ase.common.core_level import (
    get_marked_structures,
    get_binding_energy,
)
from workgraph_collections.ase.espresso import pw_calculator
from ase import Atoms
from copy import deepcopy


@task.graph(outputs=dynamic(dict))
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
) -> dict:
    """Run the scf calculation for each atoms."""
    from .pw import pw_calculator

    results = {}

    # run the ground state calculation for the supercell
    # update pseudopotentials using ground state pseudopotentials
    for key, value in core_hole_pseudos.items():
        pseudopotentials[key] = value["ground"]
    scf_ground_inputs = {
        "command": command,
        "input_data": input_data,
        "kpts": kpts,
        "pseudopotentials": pseudopotentials,
        "pseudo_dir": pseudo_dir,
    }
    scf_ground = pw_calculator(
        atoms=marked_atoms.pop("supercell"),
        computer=computer,
        metadata=metadata,
        **scf_ground_inputs
    )
    results["ground"] = scf_ground.parameters
    # remove the original atoms
    marked_atoms.pop("original", None)
    for key, atoms in marked_atoms.items():
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
        scf_inputs = {
            "command": command,
            "input_data": input_data,
            "kpts": kpts,
            "pseudopotentials": pseudopotentials,
            "pseudo_dir": pseudo_dir,
        }
        scf_out = pw_calculator(
            atoms=atoms, computer=computer, metadata=metadata, **scf_inputs
        )
        # save the output parameters to the context
        results[key] = scf_out.parameters
    return results


@task.graph(outputs=[{"name": "result", "from": "binding_energy.result"}])
def XpsWorkgraph(
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

    # -------- relax -----------
    if run_relax:

        relax_inputs = deepcopy(scf_inputs)
        input_data = Namelist(relax_inputs.get("input_data", {})).to_nested(binary="pw")
        input_data["CONTROL"]["calculation"] = "relax"
        relax_inputs["input_data"] = input_data
        relax_out = pw_calculator(name="relax", atoms=atoms, **relax_inputs)
        atoms = relax_out.atoms
    # -------- get_marked_atoms -----------
    marked_atoms_out = get_marked_structures(
        name="marked_atoms", atoms=atoms, **marked_structures_inputs
    )
    # ------------------ run scf -------------------
    run_scf_out = run_scf(
        marked_atoms=marked_atoms_out.structures,
        core_hole_pseudos=core_hole_pseudos,
        is_molecule=marked_structures_inputs.get("is_molecule", False),
        metadata=metadata,
        **scf_inputs
    )
    # -------- calculate binding energy -----------
    binding_energy = get_binding_energy(
        core_hole_pseudos=core_hole_pseudos,
        scf_outputs=run_scf_out.results,
    )
    return binding_energy.result
