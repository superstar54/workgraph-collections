from aiida_workgraph import WorkGraph, node
from aiida.orm import StructureData, Dict, KpointsData, Code
from workgraph_collections.common.xps import get_marked_structures, binding_energy


@node.graph_builder(outputs=[["context.scf", "result"]])
def run_scf(
    structures: StructureData = None,
    code: Code = None,
    parameters: dict = None,
    kpoints: KpointsData = None,
    pseudos: dict = None,
    core_hole_pseudos: dict = None,
    core_hole_treatment: str = "xch",
    is_molecule: bool = None,
    metadata: dict = None,
):
    from aiida_workgraph import WorkGraph
    from aiida_quantumespresso.calculations.pw import PwCalculation
    from copy import deepcopy

    #
    wg = WorkGraph("run_scf")
    # ground state
    pw_ground = wg.nodes.new(PwCalculation, name="ground")
    pw_ground.set(
        {
            "code": code,
            "parameters": parameters,
            "kpoints": kpoints,
            "pseudos": pseudos,
            "metadata": metadata,
            "structure": structures.pop("ground"),
        }
    )
    pw_ground.to_context = [["output_parameters", "scf.ground"]]
    # excited state node
    for key, structure in structures.items():
        pseudos1 = pseudos.copy()
        peak = structure.base.extras.get("info")["peak"]
        pseudos1["X"] = core_hole_pseudos[peak]
        # remove pseudo of non-exist element
        pseudos1 = {kind.name: pseudos1[kind.name] for kind in structure.kinds}
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
        pw_excited = wg.nodes.new(PwCalculation, name=f"pw_excited_{key}")
        pw_excited.set(
            {
                "code": code,
                "parameters": ch_parameters,
                "kpoints": kpoints,
                "pseudos": pseudos1,
                "metadata": metadata,
                "structure": structure,
                "settings": settings,
            }
        )
        pw_excited.to_context = [["output_parameters", f"scf.{key}"]]
    return wg


@node.graph_builder(outputs=[["binding_energy.result", "result"]])
def xps_workgraph(
    structure: StructureData = None,
    code: Code = None,
    atoms_list: list = None,
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
    get_marked_structures1 = wg.nodes.new(
        get_marked_structures,
        name="get_marked_structures",
        marker="X",
        structure=structure,
        atoms_list=atoms_list,
    )
    run_scf1 = wg.nodes.new(
        run_scf,
        name="run_scf",
        code=code,
        parameters=parameters,
        kpoints=kpoints,
        pseudos=pseudos,
        core_hole_pseudos=core_hole_pseudos,
        is_molecule=is_molecule,
        core_hole_treatment=core_hole_treatment,
        metadata=metadata,
        structures=get_marked_structures1.outputs["structures"],
    )
    wg.nodes.new(
        binding_energy,
        name="binding_energy",
        corrections=correction_energies,
        scf_outputs=run_scf1.outputs["result"],
        energy_units="eV",
    )
    return wg
