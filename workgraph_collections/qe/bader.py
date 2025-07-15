# -*- coding: utf-8 -*-
"""QeBaderWorkGraph of the AiiDA bader plugin"""
from aiida import orm
from aiida_workgraph import task
from workgraph_collections.qe import PwBaseTask, PpTask
from aiida_bader.calculations import BaderCalculation

BaderTask = task()(BaderCalculation)


@task.graph(outputs=["bader_charge"])
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

    inputs = {} if inputs is None else inputs
    # -------- scf -----------
    scf_inputs = inputs.get("scf", {})
    scf_inputs.update({"pw.structure": structure, "pw.code": pw_code})
    scf_outs = PwBaseTask(**scf_inputs)
    # -------- pp valence -----------
    pp_valence_outs = PpTask(
        code=pp_code,
        parent_folder=scf_outs.remote_folder,
        **inputs.get("pp_valence", {}),
    )
    # -------- pp all -----------
    pp_all_outs = PpTask(
        code=pp_code,
        parent_folder=scf_outs.remote_folder,
        **inputs.get("pp_all", {}),
    )

    # -------- bader -----------
    bader_outs = BaderTask(
        code=bader_code,
        charge_density_folder=pp_valence_outs.remote_folder,
        reference_charge_density_folder=pp_all_outs.remote_folder,
        **inputs.get("bader", {}),
    )
    return {"bader_charge": bader_outs.bader_charge}
