from aiida_workgraph import WorkGraph, node
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_wannier90_workflows.workflows.base.pw2wannier90 import (
    Pw2wannier90BaseWorkChain,
)
from aiida_wannier90_workflows.workflows.base.wannier90 import Wannier90BaseWorkChain


@node.graph_builder()
def wannier90_minimal_base_workgraph(structure=None, inputs=None):
    """Generate PdosWorkGraph."""
    inputs = {} if inputs is None else inputs
    # create workgraph
    wg = WorkGraph()
    wg.context = {}
    # -------- scf -----------
    scf_node = wg.nodes.new(PwBaseWorkChain, name="scf")
    scf_inputs = inputs.get("scf", {})
    scf_inputs["pw.structure"] = structure
    scf_node.set(scf_inputs)
    # -------- nscf -----------
    nscf_node = wg.nodes.new(PwBaseWorkChain, name="nscf")
    nscf_inputs = inputs.get("nscf", {})
    nscf_inputs.update(
        {
            "pw.structure": structure,
            "pw.parent_folder": scf_node.outputs["remote_folder"],
        }
    )
    nscf_node.set(nscf_inputs)
    # -------- wannier90_pp -----------
    wannier90_pp = wg.nodes.new(Wannier90BaseWorkChain, name="wannier90_pp")
    wannier90_pp_inputs = inputs.get("wannier90_pp", {})
    wannier90_pp_inputs.update(
        {
            "wannier90.structure": structure,
        }
    )
    wannier90_pp.set(wannier90_pp_inputs)
    # -------- pw2wannier90 -----------
    pw2wannier90 = wg.nodes.new(Pw2wannier90BaseWorkChain, name="pw2wannier90")
    pw2wannier90_inputs = inputs.get("pw2wannier90", {})
    pw2wannier90.set(
        {
            "pw2wannier90.nnkp_file": wannier90_pp.outputs["nnkp_file"],
            "pw2wannier90.parent_folder": nscf_node.outputs["remote_folder"],
        }
    )
    pw2wannier90.set(pw2wannier90_inputs)
    # -------- wannier90 -----------
    wannier90 = wg.nodes.new(Wannier90BaseWorkChain, name="wannier90")
    wannier90_inputs = inputs.get("wannier90", {})
    wannier90_inputs.update(
        {
            "wannier90.structure": structure,
            "wannier90.remote_input_folder": pw2wannier90.outputs["remote_folder"],
        }
    )
    wannier90.set(wannier90_inputs)
    return wg
