from aiida_workgraph import WorkGraph, task, build_node
from aiida.orm import List, Dict, StructureData, Code, Bool
from workgraph_collections.common.xps import binding_energy
from aiida_quantumespresso.workflows.functions.get_xspectra_structures import (
    get_xspectra_structures,
)
from aiida_quantumespresso.workflows.functions.get_marked_structures import (
    get_marked_structures,
)

# add a output socket manually
GetXspectraStructureNode = build_node(
    get_xspectra_structures,
    outputs=[["General", "output_parameters"], ["General", "marked_structures"]],
)
GetMarkedStructuresNode = build_node(
    get_marked_structures,
    outputs=[["General", "output_parameters"], ["General", "marked_structures"]],
)


@task.graph_builder(outputs=[["context.scf", "result"]])
def run_scf(
    structure,
    code: Code = None,
    parameters: dict = None,
    basis_pseudo_files: dict = None,
    core_hole_pseudos: dict = None,
    core_hole_treatment: str = "xch",
    is_molecule: bool = None,
    metadata: dict = None,
    **marked_structures,
):
    from aiida_cp2k.calculations import Cp2kCalculation
    from copy import deepcopy

    output_parameters = marked_structures.pop("output_parameters", Dict({})).get_dict()
    sites_info = output_parameters["equivalent_sites_data"]
    wg = WorkGraph("run_scf")
    if is_molecule:
        parameters["FORCE_EVAL"]["DFT"].setdefault("POISSON", {})
        parameters["FORCE_EVAL"]["DFT"]["POISSON"].update(
            {"PERIODIC": None, "PSOLVER": "MT"}
        )
        core_hole_treatment = "FULL"
    # ground state
    supercell = marked_structures.pop("supercell", structure)
    scf_ground = wg.tasks.new(Cp2kCalculation, name="scf_ground")
    scf_ground.set(
        {
            "code": code,
            "parameters": Dict(parameters),
            "metadata": metadata,
            "file": basis_pseudo_files,
            "structure": supercell,
        }
    )
    scf_ground.to_context = [["output_parameters", "scf.ground"]]
    marked_structures = marked_structures["marked_structures"]
    # excited state node
    for key, marked_structure in marked_structures.items():
        ch_parameters = deepcopy(parameters)
        symbol = sites_info[key]["symbol"]
        ch_parameters["FORCE_EVAL"]["SUBSYS"]["KIND"].append(core_hole_pseudos[symbol])
        if core_hole_treatment.upper() == "XCH":
            ch_parameters["FORCE_EVAL"]["DFT"].update(
                {"UKS": True, "MULTIPLICITY": 2, "CHARGE": -1}
            )
        else:
            ch_parameters["FORCE_EVAL"]["DFT"].update(
                {"UKS": False, "MULTIPLICITY": 1, "CHARGE": 0}
            )
        scf_ch = wg.tasks.new(Cp2kCalculation, name=f"scf_{key}")
        scf_ch.set(
            {
                "code": code,
                "parameters": Dict(ch_parameters),
                "metadata": metadata,
                "file": basis_pseudo_files,
                "structure": marked_structure,
            }
        )
        scf_ch.to_context = [["output_parameters", f"scf.{key}"]]
    return wg


@task.graph_builder(outputs=[["binding_energy.result", "result"]])
def xps_workgraph(
    structure: StructureData = None,
    code: Code = None,
    atoms_list: list = None,
    element_list: list = None,
    parameters: dict = None,
    basis_pseudo_files: dict = None,
    core_hole_pseudos: dict = None,
    core_hole_treatment: str = "xch",
    correction_energies: dict = None,
    is_molecule: bool = None,
    metadata: dict = None,
):
    """Workgraph for XPS calculation.
    1. Get the marked structures for each atom.
    2. Run the SCF calculation for ground state, and each marked structure with core hole.
    3. Calculate the binding energy.
    """

    wg = WorkGraph("XPS")
    if atoms_list:
        structures_node = wg.tasks.new(
            GetMarkedStructuresNode,
            name="marked_structures",
            structure=structure,
            atoms_list=atoms_list,
        )
    else:
        structures_node = wg.tasks.new(
            GetXspectraStructureNode,
            name="marked_structures",
            structure=structure,
            kwargs={
                "absorbing_elements_list": List(element_list),
                "is_molecule_input": Bool(is_molecule),
            },
        )
    scf_task = wg.tasks.new(
        run_scf,
        name="run_scf",
        structure=structure,
        code=code,
        parameters=parameters,
        basis_pseudo_files=basis_pseudo_files,
        core_hole_pseudos=core_hole_pseudos,
        core_hole_treatment=core_hole_treatment,
        metadata=metadata,
        is_molecule=is_molecule,
        marked_structures=structures_node.outputs["_outputs"],
    )
    wg.tasks.new(
        binding_energy,
        name="binding_energy",
        sites_info=structures_node.outputs["output_parameters"],
        scf_outputs=scf_task.outputs["result"],
        corrections=correction_energies,
        energy_units="a.u",
    )
    return wg
