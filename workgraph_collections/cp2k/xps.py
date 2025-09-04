from aiida_workgraph import task
from workgraph_collections.cp2k import Cp2kTask
from aiida.orm import List, Dict, StructureData, Code, Bool
from workgraph_collections.common.xps import binding_energy
from aiida_quantumespresso.workflows.functions.get_xspectra_structures import (
    get_xspectra_structures,
)
from aiida_quantumespresso.workflows.functions.get_marked_structures import (
    get_marked_structures,
)

# add a output socket manually
GetXspectraStructureTask = task(outputs=["output_parameters", "marked_structures"])(
    get_xspectra_structures
)
GetMarkedStructuresTask = task(outputs=["output_parameters", "marked_structures"])(
    get_marked_structures
)


@task.graph(outputs=[{"name": "result", "form": "context.scf"}])
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
    from copy import deepcopy

    results = {}

    output_parameters = marked_structures.pop("output_parameters", Dict({})).get_dict()
    sites_info = output_parameters["equivalent_sites_data"]
    if is_molecule:
        parameters["FORCE_EVAL"]["DFT"].setdefault("POISSON", {})
        parameters["FORCE_EVAL"]["DFT"]["POISSON"].update(
            {"PERIODIC": None, "PSOLVER": "MT"}
        )
        core_hole_treatment = "FULL"
    # ground state
    supercell = marked_structures.pop("supercell", structure)
    scf_ground_inputs = {
        "code": code,
        "parameters": Dict(parameters),
        "metadata": metadata,
        "file": basis_pseudo_files,
        "structure": supercell,
    }
    scf_ground_out = Cp2kTask(**scf_ground_inputs)
    results["ground"] = scf_ground_out.output_parameters
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
        scf_ch_inputs = {
            "code": code,
            "parameters": Dict(ch_parameters),
            "metadata": metadata,
            "file": basis_pseudo_files,
            "structure": marked_structure,
        }
        scf_ch = Cp2kTask(**scf_ch_inputs)
        results[key] = scf_ch.output_parameters
    return results


@task.graph()
def XpsWorkgraph(
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

    if atoms_list:
        structures_out = GetMarkedStructuresTask(
            structure=structure,
            atoms_list=atoms_list,
        )
    else:
        structures_out = GetXspectraStructureTask(
            structure=structure,
            absorbing_elements_list=List(element_list),
            is_molecule_input=Bool(is_molecule),
        )
    scf_out = run_scf(
        structure=structure,
        code=code,
        parameters=parameters,
        basis_pseudo_files=basis_pseudo_files,
        core_hole_pseudos=core_hole_pseudos,
        core_hole_treatment=core_hole_treatment,
        metadata=metadata,
        is_molecule=is_molecule,
        marked_structures=structures_out,
    )
    return binding_energy(
        name="binding_energy",
        sites_info=structures_out.output_parameters,
        scf_outputs=scf_out.result,
        corrections=correction_energies,
        energy_units="a.u",
    ).result
