from aiida.orm import StructureData, Dict, KpointsData, Code, List, Bool, UpfData
from workgraph_collections.common.xps import binding_energy
from aiida_qe_xspec.workflows.functions.get_xspectra_structures import (
    get_xspectra_structures,
)
from aiida_qe_xspec.workflows.functions.get_marked_structures import (
    get_marked_structures,
)
from typing import Annotated
from aiida_workgraph import task, spec
from workgraph_collections.qe import PwTask

# add a output socket manually
GetXspectraStructureTask = task(outputs=["output_parameters", "marked_structures"])(
    get_xspectra_structures
)
GetMarkedStructuresTask = task(outputs=["output_parameters", "marked_structures"])(
    get_marked_structures
)


@task.graph(outputs=spec.dynamic(PwTask.outputs.output_parameters))
def run_scf(
    structure: StructureData = None,
    code: Code = None,
    parameters: dict = None,
    kpoints: KpointsData = None,
    pseudos: Annotated[dict, spec.dynamic(UpfData)] = None,
    core_hole_pseudos: Annotated[dict, spec.dynamic(UpfData)] = None,
    core_hole_treatment: str = "xch",
    is_molecule: bool = None,
    metadata: dict = None,
    marked_structures: Annotated[dict, spec.dynamic(StructureData)] = None,
):
    from copy import deepcopy

    results = {}
    #
    output_parameters = marked_structures.pop("output_parameters", Dict({})).get_dict()
    sites_info = output_parameters["equivalent_sites_data"]

    for site in sites_info:
        abs_element = sites_info[site]["symbol"]
        pseudos[abs_element] = core_hole_pseudos["gipaw"][abs_element]
    # ground state
    supercell = marked_structures.pop("supercell", structure)
    parameters = parameters.get_dict() if isinstance(parameters, Dict) else parameters
    pw_inputs = {
        "code": code,
        "parameters": Dict(parameters),
        "kpoints": kpoints,
        "pseudos": pseudos,
        "metadata": metadata,
        "structure": supercell,
    }
    pw_ground_out = PwTask(**pw_inputs)
    results["ground"] = pw_ground_out.output_parameters
    # excited state node
    for key, data in sites_info.items():
        symbol = data["symbol"]
        marked_structure = marked_structures[f"{key}_{symbol}"]
        pseudos1 = pseudos.copy()
        pseudos1["X"] = core_hole_pseudos["core_hole"][symbol]
        # remove pseudo of non-exist element
        pseudos1 = {kind.name: pseudos1[kind.name] for kind in marked_structure.kinds}
        # update parameters
        ch_parameters = deepcopy(parameters)
        if is_molecule:
            ch_parameters["SYSTEM"]["assume_isolated"] = "mt"
            settings = Dict(dict={"gamma_only": True})
            kpoints = KpointsData()
            kpoints.set_kpoints_mesh([1, 1, 1])
            core_hole_treatment = "FULL"
        else:
            settings = None
        if core_hole_treatment.upper() == "XCH_SMEAR":
            ch_parameters["SYSTEM"].update(
                {
                    "occupations": "smearing",
                    "tot_charge": 0,
                    "nspin": 2,
                    "starting_magnetization(1)": 0,
                }
            )
        elif core_hole_treatment.upper() == "XCH_FIXED":
            ch_parameters["SYSTEM"].update(
                {
                    "occupations": "fixed",
                    "tot_charge": 0,
                    "nspin": 2,
                    "tot_magnetization": 1,
                }
            )
        elif core_hole_treatment.upper() == "FULL":
            ch_parameters["SYSTEM"].update(
                {
                    "tot_charge": 1,
                }
            )
        pw_inputs = {
            "code": code,
            "parameters": Dict(ch_parameters),
            "kpoints": kpoints,
            "pseudos": pseudos1,
            "metadata": metadata,
            "structure": marked_structure,
            "settings": settings,
        }
        pw_excited_out = PwTask(**pw_inputs)
        results[key] = pw_excited_out.output_parameters
    return results


@task.graph()
def XpsWorkgraph(
    structure: StructureData = None,
    code: Code = None,
    atoms_list: list = None,
    element_list: list = None,
    parameters: dict = None,
    kpoints: KpointsData = None,
    pseudos: Annotated[dict, spec.dynamic(UpfData)] = None,
    is_molecule: bool = False,
    core_hole_treatment: str = "xch",
    core_hole_pseudos: Annotated[dict, spec.dynamic(UpfData)] = None,
    correction_energies: dict = None,
    metadata: dict = None,
):
    """Workgraph for XPS calculation.
    1. Get the marked structures for each atom.
    2. Run the SCF calculation for ground state, and each marked structure with core hole.
    3. Calculate the binding energy.
    """
    if atoms_list:
        structures_task_out = GetMarkedStructuresTask(
            structure=structure,
            atoms_list=atoms_list,
        )
    else:
        structures_task_out = GetXspectraStructureTask(
            structure=structure,
            absorbing_elements_list=List(element_list),
            is_molecule_input=Bool(is_molecule),
        )
    run_scf1_out = run_scf(
        structure=structure,
        code=code,
        parameters=parameters,
        kpoints=kpoints,
        pseudos=pseudos,
        core_hole_pseudos=core_hole_pseudos,
        is_molecule=is_molecule,
        core_hole_treatment=core_hole_treatment,
        metadata=metadata,
        marked_structures=structures_task_out,
    )
    binding_energy_out = binding_energy(
        sites_info=structures_task_out.output_parameters,
        corrections=correction_energies,
        scf_outputs=run_scf1_out,
    )
    return binding_energy_out
