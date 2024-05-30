from aiida import orm
from aiida_workgraph import node, WorkGraph
from workgraph_collections.common.eos import scale_structure, fit_eos


# Output result from context to the output socket
@node.graph_builder(outputs=[["context.result", "result"]])
def all_scf(structures, scf_inputs):
    """Run the scf calculation for each structure."""
    from aiida_workgraph import WorkGraph
    from aiida_cp2k.calculations import Cp2kCalculation

    wg = WorkGraph()
    for key, structure in structures.items():
        scf = wg.nodes.new(Cp2kCalculation, name=f"scf_{key}", structure=structure)
        scf.set(scf_inputs)
        # save the output parameters to the context
        scf.to_context = [["output_parameters", f"result.{key}"]]
    return wg


@node.graph_builder(outputs=[["fit_eos.result", "result"]])
def eos_workgraph(
    structure: orm.StructureData = None,
    code: orm.Code = None,
    scales: list = None,
    parameters: dict = None,
    basis_pseudo_files: dict = None,
    metadata: dict = None,
):
    """Workgraph for EOS calculation.
    1. Get the scaled structures.
    2. Run the SCF calculation for each scaled structure.
    3. Fit the EOS.
    """
    wg = WorkGraph("EOS")
    scale_structure1 = wg.nodes.new(
        scale_structure, name="scale_structure", structure=structure, scales=scales
    )
    all_scf1 = wg.nodes.new(
        all_scf,
        name="all_scf",
        structures=scale_structure1.outputs["structures"],
        scf_inputs={
            "code": code,
            "parameters": orm.Dict(parameters),
            "file": basis_pseudo_files,
            "metadata": metadata,
        },
    )
    wg.nodes.new(
        fit_eos,
        name="fit_eos",
        volumes=scale_structure1.outputs["volumes"],
        scf_outputs=all_scf1.outputs["result"],
    )
    return wg
