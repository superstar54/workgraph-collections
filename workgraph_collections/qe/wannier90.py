from aiida import orm
from aiida_workgraph import WorkGraph, node, build_node
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_wannier90_workflows.workflows.base.wannier90 import Wannier90BaseWorkChain
from aiida_wannier90_workflows.workflows.base.projwfc import ProjwfcBaseWorkChain
from aiida_wannier90_workflows.workflows.base.pw2wannier90 import (
    Pw2wannier90BaseWorkChain,
)
from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import (
    seekpath_structure_analysis,
)


SeekpathNode = build_node(
    seekpath_structure_analysis,
    outputs=[
        ["General", "primitive_structure"],
        ["General", "explicit_kpoints"],
        ["General", "parameters"],
        ["General", "conv_structure"],
    ],
)


@node.calcfunction(outputs=[["General", "kpoint_path"]])
def inspect_seekpath(parameters):
    """Inspect seekpath calculation."""
    parameters = parameters.get_dict()
    kpoint_path = orm.Dict(
        dict={
            "path": parameters["path"],
            "point_coords": parameters["point_coords"],
        }
    )
    return kpoint_path


@node.calcfunction()
def prepare_wannier90_pp_inputs(parameters, nscf_output):
    """Prepare the inputs of wannier90 calculation before submission."""
    parameters = parameters.get_dict()
    parameters["fermi_energy"] = nscf_output.get_dict().get("fermi_energy")
    return orm.Dict(parameters)


@node.graph_builder()
def wannier_workgraph(structure=None, inputs=None):
    """Generate PdosWorkGraph."""
    inputs = {} if inputs is None else inputs
    # create workgraph
    wg = WorkGraph()
    wg.ctx = {}
    # -------- seekpath -----------
    seekpath_node = wg.nodes.new(
        SeekpathNode,
        name="seekpath",
        structure=structure,
    )
    inspect_seekpath_node = wg.nodes.new(
        inspect_seekpath,
        name="inspect_seekpath",
        parameters=seekpath_node.outputs["parameters"],
    )
    inspect_seekpath
    # -------- scf -----------
    scf_node = wg.nodes.new(PwBaseWorkChain, name="scf")
    scf_inputs = inputs.get("scf", {})
    scf_inputs["pw.structure"] = seekpath_node.outputs["primitive_structure"]
    scf_node.set(scf_inputs)
    # -------- nscf -----------
    nscf_node = wg.nodes.new(PwBaseWorkChain, name="nscf")
    nscf_inputs = inputs.get("nscf", {})
    nscf_inputs.update(
        {
            "pw.structure": seekpath_node.outputs["primitive_structure"],
            "pw.parent_folder": scf_node.outputs["remote_folder"],
        }
    )
    nscf_node.set(nscf_inputs)
    # -------- projwfc -----------
    projwfc = wg.nodes.new(
        ProjwfcBaseWorkChain,
        name="projwfc",
    )
    projwfc_inputs = inputs.get("projwfc", {})
    projwfc_inputs.update({"projwfc.parent_folder": nscf_node.outputs["remote_folder"]})
    projwfc.set(projwfc_inputs)
    # -------- wannier90_pp -----------
    wannier90_pp = wg.nodes.new(
        Wannier90BaseWorkChain,
        name="wannier90_pp",
        bands=nscf_node.outputs["output_band"],
    )
    wannier90_pp_inputs = inputs.get("wannier90_pp", {"wannier90": {"parameters": {}}})
    wannier90_pp_inputs.update(
        {
            "wannier90.structure": seekpath_node.outputs["primitive_structure"],
            "wannier90.kpoint_path": inspect_seekpath_node.outputs["kpoint_path"],
        }
    )
    wannier90_pp.set(wannier90_pp_inputs)
    wannier90_pp_parameters = wg.nodes.new(
        prepare_wannier90_pp_inputs,
        name="wannier90_pp_parameters",
        parameters=wannier90_pp_inputs["wannier90"].get("parameters"),
    )
    wg.links.new(
        nscf_node.outputs["output_parameters"],
        wannier90_pp_parameters.inputs["nscf_output"],
    )
    wg.links.new(
        wannier90_pp_parameters.outputs[0], wannier90_pp.inputs["wannier90.parameters"]
    )
    # -------- pw2wannier90 -----------
    pw2wannier90 = wg.nodes.new(
        Pw2wannier90BaseWorkChain,
        name="pw2wannier90",
        bands=projwfc.outputs["bands"],
        bands_projections=projwfc.outputs["projections"],
    )
    pw2wannier90.set(
        {
            "pw2wannier90.nnkp_file": wannier90_pp.outputs["nnkp_file"],
            "pw2wannier90.parent_folder": nscf_node.outputs["remote_folder"],
        }
    )
    pw2wannier90_inputs = inputs.get("pw2wannier90", {})
    pw2wannier90.set(pw2wannier90_inputs)
    # -------- wannier90 -----------
    wannier90 = wg.nodes.new(Wannier90BaseWorkChain, name="wannier90")
    wannier90_inputs = inputs.get("wannier90", {})
    wannier90_inputs.update(
        {
            "wannier90.structure": seekpath_node.outputs["primitive_structure"],
            "wannier90.remote_input_folder": pw2wannier90.outputs["remote_folder"],
            "wannier90.kpoint_path": inspect_seekpath_node.outputs["kpoint_path"],
        }
    )
    wannier90.set(wannier90_inputs)
    # export workgraph
    return wg
