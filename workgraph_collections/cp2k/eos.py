from aiida import orm
from aiida_workgraph import task, dynamic
from workgraph_collections.common.eos import scale_structure, fit_eos
from workgraph_collections.cp2k import Cp2kTask
from typing import Annotated


@task.graph(outputs=dynamic(dict))
def all_scf(
    structures: Annotated[dict, dynamic(orm.StructureData)],
    code: orm.Code,
    parameters: orm.Dict,
    file: Annotated[dict, dynamic(orm.SinglefileData)] = None,
    metadata: dict = None,
):
    """Run the scf calculation for each structure."""

    results = {}
    for key, structure in structures.items():
        scf_out = Cp2kTask(
            structure=structure,
            code=code,
            parameters=parameters,
            file=file,
            metadata=metadata,
        )
        results[key] = scf_out.output_parameters
    return results


@task.graph()
def EosWorkgraph(
    structure: orm.StructureData = None,
    scales: list = None,
    scf_inputs: Annotated[dict, all_scf.inputs] = None,
):
    """Workgraph for EOS calculation.
    1. Get the scaled structures.
    2. Run the SCF calculation for each scaled structure.
    3. Fit the EOS.
    """
    scale_out = scale_structure(structure=structure, scales=scales)
    all_scf1 = all_scf(structures=scale_out.structures, **scf_inputs)
    return fit_eos(
        volumes=scale_out.volumes,
        scf_outputs=all_scf1,
    ).result
