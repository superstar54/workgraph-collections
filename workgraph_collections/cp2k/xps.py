from aiida_workgraph import WorkGraph, node
from aiida.orm import Dict, StructureData, Code
from workgraph_collections.common.xps import get_marked_structures, binding_energy


@node.graph_builder(outputs=[["context.scf", "result"]])
def run_scf(
    structures: StructureData = None,
    code: Code = None,
    parameters: dict = None,
    basis_pseudo_files: dict = None,
    core_hole_pseudos: dict = None,
    core_hole_treatment: str = "xch",
    metadata: dict = None,
):
    from aiida_cp2k.calculations import Cp2kCalculation
    from copy import deepcopy

    wg = WorkGraph("run_scf")
    # ground state
    scf_ground = wg.nodes.new(Cp2kCalculation, name="scf_ground")
    scf_ground.set(
        {
            "code": code,
            "parameters": Dict(parameters),
            "metadata": metadata,
            "file": basis_pseudo_files,
            "structure": structures.pop("ground"),
        }
    )
    scf_ground.to_context = [["output_parameters", "scf.ground"]]
    # excited state node
    for key, structure in structures.items():
        ch_parameters = deepcopy(parameters)
        symbol = key.split("_")[0]
        ch_parameters["FORCE_EVAL"]["SUBSYS"]["KIND"].append(core_hole_pseudos[symbol])
        if core_hole_treatment.upper == "XCH":
            ch_parameters["FORCE_EVAL"]["DFT"].update(
                {"UKS": True, "MULTIPLICITY": 2, "CHARGE": -1}
            )
        else:
            ch_parameters["FORCE_EVAL"]["DFT"].update(
                {"UKS": False, "MULTIPLICITY": 1, "CHARGE": 0}
            )
        scf_ch = wg.nodes.new(Cp2kCalculation, name=f"scf_{key}")
        scf_ch.set(
            {
                "code": code,
                "parameters": Dict(ch_parameters),
                "metadata": metadata,
                "file": basis_pseudo_files,
                "structure": structure,
            }
        )
        scf_ch.to_context = [["output_parameters", f"scf.{key}"]]
    return wg


@node.graph_builder(outputs=[["binding_energy.result", "result"]])
def xps_workgraph(
    structure: StructureData = None,
    code: Code = None,
    atoms_list: list = None,
    parameters: dict = None,
    basis_pseudo_files: dict = None,
    core_hole_pseudos: dict = None,
    core_hole_treatment: str = "xch",
    correction_energies: dict = None,
    metadata: dict = None,
):
    """Workgraph for XPS calculation.
    1. Get the marked structures for each atom.
    2. Run the SCF calculation for ground state, and each marked structure with core hole.
    3. Calculate the binding energy.
    """
    wg = WorkGraph()
    structures_node = wg.nodes.new(
        get_marked_structures,
        name="get_marked_structures",
        structure=structure,
        atoms_list=atoms_list,
    )
    scf_node = wg.nodes.new(
        run_scf,
        name="run_scf",
        code=code,
        parameters=parameters,
        basis_pseudo_files=basis_pseudo_files,
        core_hole_pseudos=core_hole_pseudos,
        core_hole_treatment=core_hole_treatment,
        metadata=metadata,
        structures=structures_node.outputs["structures"],
    )
    wg.nodes.new(
        binding_energy,
        name="binding_energy",
        scf_outputs=scf_node.outputs["result"],
        corrections=correction_energies,
        energy_units="a.u",
    )
    return wg
