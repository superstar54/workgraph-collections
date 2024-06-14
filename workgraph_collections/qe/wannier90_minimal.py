from aiida_workgraph import WorkGraph, task
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.calculations.pw2wannier90 import Pw2wannier90Calculation
from aiida_wannier90.calculations.wannier90 import Wannier90Calculation


@task.graph_builder()
def wannier90_minimal_workgraph(structure=None, inputs=None):
    """Generate PdosWorkGraph."""
    inputs = {} if inputs is None else inputs
    # create workgraph
    wg = WorkGraph("Wannier90_Minimal")
    wg.context = {}
    # -------- scf -----------
    scf_task = wg.tasks.new(PwCalculation, name="scf", structure=structure)
    scf_inputs = inputs.get("scf", {})
    scf_task.set(scf_inputs)
    # -------- nscf -----------
    nscf_task = wg.tasks.new(
        PwCalculation,
        name="nscf",
        structure=structure,
        parent_folder=scf_task.outputs["remote_folder"],
    )
    nscf_inputs = inputs.get("nscf", {})
    nscf_task.set(nscf_inputs)
    # -------- wannier90_pp -----------
    wannier90_pp = wg.tasks.new(
        Wannier90Calculation, name="wannier90_pp", structure=structure
    )
    wannier90_pp_inputs = inputs.get("wannier90_pp", {})
    wannier90_pp.set(wannier90_pp_inputs)
    # -------- pw2wannier90 -----------
    pw2wannier90 = wg.tasks.new(
        Pw2wannier90Calculation,
        name="pw2wannier90",
        nnkp_file=wannier90_pp.outputs["nnkp_file"],
        parent_folder=nscf_task.outputs["remote_folder"],
    )
    pw2wannier90_inputs = inputs.get("pw2wannier90", {})
    pw2wannier90.set(pw2wannier90_inputs)
    # -------- wannier90 -----------
    wannier90 = wg.tasks.new(
        Wannier90Calculation,
        name="wannier90",
        structure=structure,
        remote_input_folder=pw2wannier90.outputs["remote_folder"],
    )
    wannier90_inputs = inputs.get("wannier90", {})
    wannier90.set(wannier90_inputs)
    return wg
