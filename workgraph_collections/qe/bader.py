# -*- coding: utf-8 -*-
"""QeBaderWorkGraph of the AiiDA bader plugin"""
from aiida import orm
from aiida_workgraph import WorkGraph, node

@node.graph_builder(outputs=[["bader.charge", "charge"]])
def bader_workgraph(
    name: str = "bader",
    structure: orm.StructureData = None,
    pw_code: orm.Code = None,
    pp_code: orm.Code = None,
    bader_code: orm.Code = None,
    parameters: dict = None,
    kpoints: orm.KpointsData = None,
    pseudos: dict = None,
    metadata: dict = None,
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

    wg = WorkGraph(name)
    pw_node = wg.nodes.new(PwBaseWorkChain, name="scf")
    pw_code.set({
                "pw": {
                    "structure": structure,
                    "code": pw_code,
                    "parameters": parameters.get("scf", {}),
                    "pseudos": pseudos,
                    "metadata": metadata
                    },
                "kpoints": kpoints,
                })
    pp_valence = wg.nodes.new(
        PpCalculation, name="pp_valence", code=pp_code,
        parameters=parameters.get("pp_valence", {}),
        parent_folder=pw_node.outputs["remote_folder"]
    )
    pp_all = wg.nodes.new(
        PpCalculation, name="pp_all", code=pp_code,
        parameters=parameters.get("pp_all", {}),
        parent_folder=pw_node.outputs["remote_folder"]
    )
    wg.nodes.new(
        BaderCalculation,
        name="bader",
        code=bader_code,
        charge_density_folder=pp_valence.outputs["remote_folder"],
        reference_charge_density_folder=pp_all.outputs["remote_folder"],
    )
    return wg
