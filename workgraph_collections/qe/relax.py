# -*- coding: utf-8 -*-
"""PdosWorkGraph."""

from aiida import orm
from aiida_workgraph import WorkGraph
from aiida_workgraph.decorator import node
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain


@node()
def should_run_relax(is_converged=False, iteration=0, max_iterations=5):
    """Check if relax should be run."""
    return not is_converged and iteration < max_iterations


@node.calcfunction()
def prepare_relax_inputs(parameters, current_number_of_bands=None):
    """Generate scf parameters from relax calculation."""
    parameters = parameters.get_dict()
    parameters.setdefault("SYSTEM", {})
    if not current_number_of_bands:
        current_number_of_bands = parameters["SYSTEM"].get("nbnd")
    return orm.Dict(parameters)


@node()
def inspect_relax(outputs=None, prev_cell_volume=None, volume_threshold=0.1):
    """Inspect relax calculation."""
    structure = outputs.output_structure
    curr_cell_volume = structure.get_cell_volume()
    current_number_of_bands = outputs.output_parameters.get_dict()["number_of_bands"]
    is_converged = False
    if prev_cell_volume is not None:
        volume_difference = abs(prev_cell_volume - curr_cell_volume) / prev_cell_volume
        if volume_difference < volume_threshold:
            is_converged = True

    return {
        "prev_cell_volume": curr_cell_volume,
        "current_number_of_bands": orm.Int(current_number_of_bands),
        "is_converged": is_converged,
    }


@node.calcfunction()
def prepare_scf_inputs(parameters, current_number_of_bands=None):
    """Generate scf parameters from relax calculation."""
    parameters = parameters.get_dict()
    parameters.setdefault("SYSTEM", {})
    if not current_number_of_bands:
        current_number_of_bands = parameters["SYSTEM"].get("nbnd")
    return orm.Dict(parameters)


@node.group(
    outputs=[
        ["relax.output_structure", "output_structure"],
        ["inspect_relax.current_number_of_bands", "current_number_of_bands"],
    ]
)
def relax_workgraph(
    structure=None, inputs=None, max_iterations=5, volume_threshold=0.1
):
    """Generate RelaxWorkGraph."""
    inputs = {} if inputs is None else inputs
    # create workgraph
    tree = WorkGraph()
    tree.workgraph_type = "WHILE"
    tree.max_iterations = max_iterations
    tree.conditions = ["should_run_relax.result"]
    tree.context = {
        "current_structure": structure,
        "current_number_of_bands": None,
        "prev_cell_volume": None,
        "is_converged": False,
        "iteration": 0,
    }
    # -------- should run relax -----------
    tree.nodes.new(
        should_run_relax,
        name="should_run_relax",
        is_converged="{{is_converged}}",
        iteration="{{iteration}}",
        max_iterations=max_iterations,
    )
    # -------- prepare relax input -----------
    prepare_relax_inputs_node = tree.nodes.new(
        prepare_relax_inputs,
        name="prepare_relax_inputs",
        parameters=inputs["pw"].get("parameters", {}),
        current_number_of_bands="{{current_number_of_bands}}",
    )
    # -------- relax -----------
    relax_node = tree.nodes.new(PwBaseWorkChain, name="relax")
    inputs["pw.structure"] = "{{current_structure}}"
    relax_node.set(inputs)
    relax_node.to_context = [["output_structure", "current_structure"]]
    tree.links.new(
        prepare_relax_inputs_node.outputs[0], relax_node.inputs["pw.parameters"]
    )
    # -------- inspect relax -----------
    inspect_relax_node = tree.nodes.new(
        inspect_relax,
        name="inspect_relax",
        prev_cell_volume="{{prev_cell_volume}}",
        volume_threshold=volume_threshold,
    )
    tree.links.new(relax_node.outputs["_outputs"], inspect_relax_node.inputs["outputs"])
    inspect_relax_node.to_context = [
        ["current_number_of_bands", "current_number_of_bands"],
        ["prev_cell_volume", "prev_cell_volume"],
        ["is_converged", "is_converged"],
    ]
    # export workgraph
    return tree


@node.graph_builder()
def relax_scf_workgraph(
    structure=None, inputs=None, max_iterations=5, volume_threshold=0.1
):
    """Generate RelaxSCFWorkGraph."""
    inputs = {} if inputs is None else inputs
    # create workgraph
    tree = WorkGraph()
    # -------- relax -----------
    relax_inputs = inputs.get("relax", {})
    relax_node = tree.nodes.new(
        relax_workgraph,
        name="relax",
        structure=structure,
        inputs=relax_inputs,
        max_iterations=max_iterations,
        volume_threshold=volume_threshold,
    )
    # -------- prepare scf inputs -----------
    scf_inputs = inputs.get("scf", {})
    prepare_scf_inputs_node = tree.nodes.new(
        prepare_scf_inputs,
        name="prepare_scf_inputs",
        parameters=scf_inputs["pw"].get("parameters", {}),
    )
    tree.links.new(
        relax_node.outputs["current_number_of_bands"],
        prepare_scf_inputs_node.inputs["current_number_of_bands"],
    )
    # -------- scf -----------
    scf_node = tree.nodes.new(PwBaseWorkChain, name="scf")
    scf_node.set(scf_inputs)
    tree.links.new(prepare_scf_inputs_node.outputs[0], scf_node.inputs["pw.parameters"])
    tree.links.new(
        relax_node.outputs["output_structure"], scf_node.inputs["pw.structure"]
    )
    return tree
