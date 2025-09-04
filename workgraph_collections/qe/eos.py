from aiida import orm
from aiida_workgraph import task, spec
from workgraph_collections.common.eos import scale_structure, fit_eos
from workgraph_collections.qe import PwTask
from typing import Annotated, Any


# Output result from context to the output socket
@task.graph(outputs=spec.namespace(result=spec.dynamic(Any)))
def all_scf(
    structures: Annotated[dict, spec.dynamic(orm.StructureData)],
    scf_inputs: Annotated[dict, spec.dynamic(Any)],
) -> dict:
    """Run the scf calculation for each structure."""
    result = {}
    for key, structure in structures.items():
        scf_out = PwTask(structure=structure, **scf_inputs)
        # save the output parameters to the context
        result[key] = scf_out.output_parameters
    return {"result": result}


@task.graph
def EosWorkgraph(
    structure: orm.StructureData = None,
    code: orm.Code = None,
    scales: list = None,
    parameters: dict = None,
    kpoints: orm.KpointsData = None,
    pseudos: Annotated[dict, spec.dynamic(orm.UpfData)] = None,
    metadata: Annotated[dict, spec.dynamic(Any)] = None,
):
    """Workgraph for EOS calculation.
    1. Get the scaled structures.
    2. Run the SCF calculation for each scaled structure.
    3. Fit the EOS.
    """
    scale_structure_out = scale_structure(structure=structure, scales=scales)
    all_scf_out = all_scf(
        structures=scale_structure_out.structures,
        scf_inputs={
            "code": code,
            "parameters": orm.Dict(parameters),
            "kpoints": kpoints,
            "pseudos": pseudos,
            "metadata": metadata,
        },
    )
    return fit_eos(
        volumes=scale_structure_out.volumes,
        scf_outputs=all_scf_out.result,
    ).result
