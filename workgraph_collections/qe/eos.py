from aiida import orm
from aiida_workgraph import task, WorkGraph
from workgraph_collections.common.eos import scale_structure, fit_eos


# Output result from context to the output socket
@task.graph_builder(outputs=[{"name": "result", "from": "context.result"}])
def all_scf(structures, scf_inputs):
    """Run the scf calculation for each structure."""
    from aiida_workgraph import WorkGraph
    from aiida_quantumespresso.calculations.pw import PwCalculation

    wg = WorkGraph()
    for key, structure in structures.items():
        scf = wg.add_task(PwCalculation, name=f"scf_{key}", structure=structure)
        scf.set(scf_inputs)
        # save the output parameters to the context
        scf.set_context({f"result.{key}": "output_parameters"})
    return wg


@task.graph_builder(outputs=[{"name": "result", "from": "fit_eos.result"}])
def eos_workgraph(
    structure: orm.StructureData = None,
    code: orm.Code = None,
    scales: list = None,
    parameters: dict = None,
    kpoints: orm.KpointsData = None,
    pseudos: dict = None,
    metadata: dict = None,
):
    """Workgraph for EOS calculation.
    1. Get the scaled structures.
    2. Run the SCF calculation for each scaled structure.
    3. Fit the EOS.
    """
    wg = WorkGraph("EOS")
    scale_structure1 = wg.add_task(
        scale_structure, name="scale_structure", structure=structure, scales=scales
    )
    all_scf1 = wg.add_task(
        all_scf,
        name="all_scf",
        structures=scale_structure1.outputs.structures,
        scf_inputs={
            "code": code,
            "parameters": orm.Dict(parameters),
            "kpoints": kpoints,
            "pseudos": pseudos,
            "metadata": metadata,
        },
    )
    wg.add_task(
        fit_eos,
        name="fit_eos",
        volumes=scale_structure1.outputs["volumes"],
        scf_outputs=all_scf1.outputs.result,
    )
    return wg
