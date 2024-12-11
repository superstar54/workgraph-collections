from aiida_workgraph import WorkGraph, task, build_task
from aiida.orm import StructureData, Dict, KpointsData, Code, List, Bool
from workgraph_collections.common.xps import binding_energy
from aiida_quantumespresso.workflows.functions.get_xspectra_structures import (
    get_xspectra_structures,
)
from aiida_quantumespresso.workflows.functions.get_marked_structures import (
    get_marked_structures,
)

# add a output socket manually
GetXspectraStructureTask = build_task(
    get_xspectra_structures,
    outputs=[
        {"name": "output_parameters"},
        {"name": "marked_structures"},
    ],
)
GetMarkedStructuresTask = build_task(
    get_marked_structures,
    outputs=[
        {"name": "output_parameters"},
        {"name": "marked_structures"},
    ],
)


@task.graph_builder(outputs=[{"name": "result", "from": "context.scf"}])
def run_scf(
    structure: StructureData = None,
    code: Code = None,
    parameters: dict = None,
    kpoints: KpointsData = None,
    pseudos: dict = None,
    core_hole_pseudos: dict = None,
    core_hole_treatment: str = "xch",
    is_molecule: bool = None,
    metadata: dict = None,
    **marked_structures,
):
    from aiida_workgraph import WorkGraph
    from aiida_quantumespresso.calculations.pw import PwCalculation
    from copy import deepcopy

    #
    output_parameters = marked_structures.pop("output_parameters", Dict({})).get_dict()
    sites_info = output_parameters["equivalent_sites_data"]
    print("sites_info", sites_info)

    for site in sites_info:
        abs_element = sites_info[site]["symbol"]
        pseudos[abs_element] = core_hole_pseudos["gipaw"][abs_element]
    # ground state
    wg = WorkGraph("run_scf")
    supercell = marked_structures.pop("supercell", structure)
    pw_ground = wg.add_task(PwCalculation, name="ground")
    pw_ground.set(
        {
            "code": code,
            "parameters": parameters,
            "kpoints": kpoints,
            "pseudos": pseudos,
            "metadata": metadata,
            "structure": supercell,
        }
    )
    pw_ground.set_context({"scf.ground": "output_parameters"})
    # remove unwanted data
    marked_structures = marked_structures["marked_structures"]
    # excited state node
    for key, marked_structure in marked_structures.items():
        pseudos1 = pseudos.copy()
        symbol = sites_info[key]["symbol"]
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
        pw_excited = wg.add_task(PwCalculation, name=f"pw_excited_{key}")
        pw_excited.set(
            {
                "code": code,
                "parameters": ch_parameters,
                "kpoints": kpoints,
                "pseudos": pseudos1,
                "metadata": metadata,
                "structure": marked_structure,
                "settings": settings,
            }
        )
        pw_excited.set_context({f"scf.{key}": "output_parameters"})
    return wg


@task.graph_builder(outputs=[{"name": "result", "from": "binding_energy.result"}])
def xps_workgraph(
    structure: StructureData = None,
    code: Code = None,
    atoms_list: list = None,
    element_list: list = None,
    parameters: dict = None,
    kpoints: KpointsData = None,
    pseudos: dict = None,
    is_molecule: bool = False,
    core_hole_treatment: str = "xch",
    core_hole_pseudos: dict = None,
    correction_energies: dict = None,
    metadata: dict = None,
):
    """Workgraph for XPS calculation.
    1. Get the marked structures for each atom.
    2. Run the SCF calculation for ground state, and each marked structure with core hole.
    3. Calculate the binding energy.
    """
    wg = WorkGraph()
    if atoms_list:
        structures_task = wg.add_task(
            GetMarkedStructuresTask,
            name="marked_structures",
            structure=structure,
            atoms_list=atoms_list,
        )
    else:
        structures_task = wg.add_task(
            GetXspectraStructureTask,
            name="marked_structures",
            structure=structure,
            kwargs={
                "absorbing_elements_list": List(element_list),
                "is_molecule_input": Bool(is_molecule),
            },
        )
    run_scf1 = wg.add_task(
        run_scf,
        name="run_scf",
        structure=structure,
        code=code,
        parameters=parameters,
        kpoints=kpoints,
        pseudos=pseudos,
        core_hole_pseudos=core_hole_pseudos,
        is_molecule=is_molecule,
        core_hole_treatment=core_hole_treatment,
        metadata=metadata,
        marked_structures=structures_task.outputs["_outputs"],
    )
    wg.add_task(
        binding_energy,
        name="binding_energy",
        sites_info=structures_task.outputs["output_parameters"],
        scf_outputs=run_scf1.outputs["result"],
        corrections=correction_energies,
        energy_units="a.u",
    )
    return wg
