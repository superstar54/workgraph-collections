from aiida_workgraph import task, WorkGraph
from workgraph_collections.ase.common.core_level import (
    get_non_equivalent_site,
)
from ase import Atoms
from workgraph_collections.ase.espresso.base import pw_calculator


@task.graph_builder(
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
        scf_task = wg.tasks.new(
            "PythonJob",
            function=pw_calculator,
            name="scf",
            command=commands["pw"],
            atoms=data["atoms"],
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
        scf_task.set(scf_inputs)
        scf_task.set_context({"results": f"scf_results.{key}"})
        for calc_number, vector in enumerate(eps_vectors):
            xspectra_task = wg.tasks.new(
                "PythonJob",
                function=xspectra_calculator,
                name=f"xspectra_{key}_{calc_number}",
                command=commands["xspectra"],
                parent_folder=scf_task.outputs["remote_folder"],
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
            xspectra_task.set(xspectra_inputs)
            xspectra_task.to_context = [
                ["results", f"xspectra_results.{key}.prod_{calc_number}"]
            ]
    return wg


@task.graph_builder(outputs=[{"name": "result", "from": "binding_energy.result"}])
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

    wg = WorkGraph("XAS")
    # -------- relax -----------
    if run_relax:
        relax_task = wg.tasks.new(
            "PythonJob",
            function=pw_calculator,
            name="relax",
            atoms=atoms,
            calculation="vc-relax",
        )
        relax_inputs = inputs.get("relax", {})
        relax_task.set(relax_inputs)
        atoms = relax_task.outputs["atoms"]
    # -------- marked_atoms -----------
    marked_atoms_task = wg.tasks.new(
        "PythonJob",
        function=get_non_equivalent_site,
        name="marked_atoms",
        atoms=atoms,
        element_list=element_list,
        is_molecule=is_molecule,
        metadata=metadata,
    )
    # -------- xspectra -----------
    wg.tasks.new(
        run_all_xspectra_prod,
        name="run_all_xspectra_prod",
        marked_atoms=marked_atoms_task.outputs["result"],
        commands=commands,
        inputs=inputs,
        eps_vectors=eps_vectors,
        core_hole_pseudos=core_hole_pseudos,
        core_hole_treatment=core_hole_treatment,
    )
    # wg.tasks.new(
    #     binding_energy,
    #     name="binding_energy",
    #     corrections=corrections,
    #     scf_outputs=run_all_xspectra_prod_task.outputs["results"],
    #     run_remotely=True,
    #     metadata=metadata,
    # )
    return wg
