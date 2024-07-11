# -*- coding: utf-8 -*-
"""QeBaderWorkGraph of the AiiDA bader plugin"""
from aiida import orm
from aiida_workgraph import WorkGraph, task


@task.graph_builder(outputs=[{"name": "charge", "from": "bader.charge"}])
def bader_workgraph(
    structure: orm.StructureData = None,
    pw_code: orm.Code = None,
    pp_code: orm.Code = None,
    bader_code: orm.Code = None,
    inputs: dict = None,
):
    """Workgraph for Bader charge analysis.
    1. Run the SCF calculation.
    2. Run the PP calculation for valence charge density.
    3. Run the PP calculation for all-electron charge density.
    4. Run the Bader charge analysis.
    """

    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
    from aiida_quantumespresso.calculations.pp import PpCalculation
    from aiida_bader.calculations import BaderCalculation

    inputs = {} if inputs is None else inputs
    wg = WorkGraph("BaderCharge")
    # -------- scf -----------
    scf_task = wg.tasks.new(PwBaseWorkChain, name="scf")
    scf_inputs = inputs.get("scf", {})
    scf_inputs.update({"pw.structure": structure, "pw.code": pw_code})
    scf_task.set(scf_inputs)
    # -------- pp valence -----------
    pp_valence = wg.tasks.new(
        PpCalculation,
        name="pp_valence",
        code=pp_code,
        parent_folder=scf_task.outputs["remote_folder"],
    )
    pp_valence_inputs = inputs.get("pp_valence", {})
    pp_valence.set(pp_valence_inputs)
    # -------- pp all -----------
    pp_all = wg.tasks.new(
        PpCalculation,
        name="pp_all",
        code=pp_code,
        parent_folder=scf_task.outputs["remote_folder"],
    )
    pp_all_inputs = inputs.get("pp_all", {})
    pp_all.set(pp_all_inputs)
    # -------- bader -----------
    bader_task = wg.tasks.new(
        BaderCalculation,
        name="bader",
        code=bader_code,
        charge_density_folder=pp_valence.outputs["remote_folder"],
        reference_charge_density_folder=pp_all.outputs["remote_folder"],
    )
    bader_inputs = inputs.get("bader", {})
    bader_task.set(bader_inputs)
    return wg
