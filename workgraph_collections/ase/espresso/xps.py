from aiida_workgraph import node, WorkGraph
from workgraph_collections.ase.common.xps import (
    get_marked_atoms,
    get_non_equivalent_site,
    binding_energy,
)
from ase import Atoms


@node.graph_builder(outputs=[["context.scf", "results"]])
def run_scf(
    marked_atoms: dict,
    command: str = None,
    computer: str = None,
    input_data: dict = None,
    kpts: list = None,
    pseudopotentials: dict = None,
    pseudo_dir: str = None,
    core_hole_pseudos: dict = None,
    core_hole_treatment: str = "xch",
    is_molecule: bool = None,
    metadata: dict = None,
) -> WorkGraph:
    """Run the scf calculation for each atoms."""
    from aiida_workgraph import WorkGraph
    from .base import pw_calculator

    wg = WorkGraph()
    wg.context = {"marked_atoms": marked_atoms}
    marked_atoms = marked_atoms.value
    scf_ground = wg.nodes.new(
        pw_calculator,
        name="ground",
        atoms=marked_atoms.pop("ground"),
        run_remotely=True,
        computer=computer,
        metadata=metadata,
    )
    scf_ground.set(
        {
            "command": command,
            "input_data": input_data,
            "kpts": kpts,
            "pseudopotentials": pseudopotentials,
            "pseudo_dir": pseudo_dir,
        }
    )
    scf_ground.to_context = [["results", "scf.ground"]]
    # becareful, we generate new data here, thus break the data provenance!
    # that's why I put the marked atoms in the context, so that we can link them
    for key, atoms in marked_atoms.items():
        scf = wg.nodes.new(
            pw_calculator,
            name=f"scf_{key}",
            atoms=atoms,
            run_remotely=True,
            computer=computer,
            metadata=metadata,
        )
        # update pseudopotentials based on marked atoms
        # split key by last underscore
        print("key: ", key)
        label, _index = key.rsplit("_", 1)
        pseudopotentials["X"] = core_hole_pseudos[label]
        # update the input data based on the core hole treatment
        input_data.setdefault("SYSTEM", {})
        if is_molecule:
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
        scf.to_context = [["results", f"scf.{key}"]]
    return wg


@node.graph_builder(outputs=[["binding_energy.result", "result"]])
def xps_workgraph(
    atoms: Atoms = None,
    atoms_list: list = None,
    element_list: list = None,
    scf_inputs: str = None,
    corrections: dict = None,
    metadata: dict = None,
):
    """Workgraph for XPS calculation.
    1. Get the marked atoms.
    2. Run the SCF calculation for for ground state, and each marked atoms
    with core hole pseudopotentials.
    3. Calculate the binding energy.
    """
    scf_inputs = scf_inputs or {}

    wg = WorkGraph("XPS")
    if atoms_list:
        marked_atoms_node = wg.nodes.new(
            get_marked_atoms,
            name="marked_atoms",
            atoms=atoms,
            atoms_list=atoms_list,
            run_remotely=True,
            metadata=metadata,
        )
    elif element_list:
        marked_atoms_node = wg.nodes.new(
            get_non_equivalent_site,
            name="marked_atoms",
            atoms=atoms,
            atoms_list=atoms_list,
            run_remotely=True,
            metadata=metadata,
        )
    else:
        raise "Either atoms_list or element_list should be provided."
    run_scf_node = wg.nodes.new(
        run_scf,
        name="run_scf",
        marked_atoms=marked_atoms_node.outputs["result"],
    )
    run_scf_node.set(scf_inputs)
    wg.nodes.new(
        binding_energy,
        name="binding_energy",
        corrections=corrections,
        scf_outputs=run_scf_node.outputs["results"],
        run_remotely=True,
        metadata=metadata,
    )
    return wg
