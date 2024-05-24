from aiida_workgraph import node, WorkGraph
from workgraph_collections.ase.common.eos import generate_scaled_atoms, fit_eos


@node.graph_builder(outputs=[["context.result", "scf_results"]])
def all_scf(scaled_atoms, scf_inputs):
    """Run the scf calculation for each atoms."""
    from aiida_workgraph import WorkGraph
    from .base import pw_calculator

    wg = WorkGraph()
    for key, atoms in scaled_atoms.items():
        scf = wg.nodes.new(pw_calculator, name=f"scf_{key}", atoms=atoms)
        scf.set(scf_inputs)
        # save the output parameters to the context
        scf.to_context = [["result", f"result.{key}"]]
    return wg


@node.graph_builder(outputs=[["fit_eos.result", "result"]])
def eos_workgraph(
    name="eos",
    atoms=None,
    scales=None,
):
    """Workgraph for EOS calculation.
    1. Get the scaled atoms.
    2. Run the SCF calculation for each scaled atoms.
    3. Fit the EOS.
    """
    wg = WorkGraph(name)
    scale_atoms_node = wg.nodes.new(
        generate_scaled_atoms, name="scale_atoms", atoms=atoms, scales=scales
    )
    all_scf1 = wg.nodes.new(
        all_scf,
        name="all_scf",
        scaled_atoms=scale_atoms_node.outputs["scaled_atoms"],
    )
    wg.nodes.new(
        fit_eos,
        name="fit_eos",
        volumes=scale_atoms_node.outputs["volumes"],
        scf_results=all_scf1.outputs["scf_results"],
    )
    return wg
