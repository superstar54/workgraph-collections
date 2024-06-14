from aiida_workgraph import node, WorkGraph
from workgraph_collections.ase.common.xps import (
    get_non_equivalent_site,
)
from ase import Atoms
from workgraph_collections.ase.espresso.base import pw_calculator


@node.graph_builder(
    outputs=[["context.scf_results", "scf"], ["context.xspectra_results", "xspectra"]]
)
def run_all_xspectra_prod(
    marked_atoms: dict,
    commands: dict = None,
    inputs: dict = None,
    eps_vectors: list = None,
    core_hole_pseudos: dict = None,
    core_hole_treatment: str = "xch",
    metadata: dict = None,
) -> WorkGraph:
    """Run the scf calculation for each atoms."""
    from aiida_workgraph import WorkGraph
    from .base import pw_calculator, xspectra_calculator
    from copy import deepcopy

    wg = WorkGraph()
    wg.context = {"marked_atoms": marked_atoms}
    marked_atoms = marked_atoms.value
    marked_atoms.pop("supercell")
    for key, data in marked_atoms.items():
        scf_node = wg.nodes.new(
            pw_calculator,
            name="scf",
            command=commands["pw"],
            atoms=data["atoms"],
            run_remotely=True,
            metadata=metadata,
        )
        scf_inputs = inputs.get("pw", {})
        scf_inputs["pseudopotentials"]["X"] = core_hole_pseudos[data["symbol"]]
        input_data = scf_inputs.get("input_data", {})
        # update the input data based on the core hole treatment
        input_data.setdefault("SYSTEM", {})
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
        scf_node.set(scf_inputs)
        scf_node.to_context = [["results", f"scf_results.{key}"]]
        for calc_number, vector in enumerate(eps_vectors):
            xspectra_node = wg.nodes.new(
                xspectra_calculator,
                name=f"xspectra_{key}_{calc_number}",
                command=commands["xspectra"],
                run_remotely=True,
                parent_folder=scf_node.outputs["remote_folder"],
                parent_output_folder="out",
                parent_folder_name="out",
                metadata=metadata,
            )
            xspectra_inputs = deepcopy(inputs.get("xspectra", {}))
            input_data = deepcopy(xspectra_inputs.get("input_data", {}))
            input_data["INPUT_XSPECTRA"]["xiabs"] = data["indices"][0] + 1
            for index in [0, 1, 2]:
                input_data["INPUT_XSPECTRA"][f"xepsilon({index + 1})"] = vector[index]
            xspectra_inputs["input_data"] = input_data
            xspectra_node.set(xspectra_inputs)
            xspectra_node.to_context = [
                ["results", f"xspectra_results.{key}.prod_{calc_number}"]
            ]
    return wg


@node.graph_builder(outputs=[["binding_energy.result", "result"]])
def xas_workgraph(
    atoms: Atoms = None,
    commands: dict = None,
    element_list: list = None,
    inputs: str = None,
    eps_vectors: list = None,
    core_hole_pseudos: dict = None,
    core_hole_treatment: str = "xch",
    metadata: dict = None,
    run_relax: bool = False,
    is_molecule: bool = False,
):
    """Workgraph for XAS calculation.
    1. Get the marked atoms.
    2. Run the SCF calculation for for ground state, and each marked atoms
    with core hole pseudopotentials.
    3. Calculate the binding energy.
    """

    inputs = inputs or {}

    wg = WorkGraph("xas")
    # -------- relax -----------
    if run_relax:
        relax_node = wg.nodes.new(
            pw_calculator,
            name="relax",
            atoms=atoms,
            calculation="vc-relax",
            run_remotely=True,
        )
        relax_inputs = inputs.get("relax", {})
        relax_node.set(relax_inputs)
        atoms = relax_node.outputs["atoms"]
    # -------- marked_atoms -----------
    marked_atoms_node = wg.nodes.new(
        get_non_equivalent_site,
        name="marked_atoms",
        atoms=atoms,
        element_list=element_list,
        is_molecule=is_molecule,
        run_remotely=True,
        metadata=metadata,
    )
    # -------- xspectra -----------
    wg.nodes.new(
        run_all_xspectra_prod,
        name="run_all_xspectra_prod",
        marked_atoms=marked_atoms_node.outputs["result"],
        commands=commands,
        inputs=inputs,
        eps_vectors=eps_vectors,
        core_hole_pseudos=core_hole_pseudos,
        core_hole_treatment=core_hole_treatment,
    )
    # wg.nodes.new(
    #     binding_energy,
    #     name="binding_energy",
    #     corrections=corrections,
    #     scf_outputs=run_all_xspectra_prod_node.outputs["results"],
    #     run_remotely=True,
    #     metadata=metadata,
    # )
    return wg