"""BandsWorkGraph."""

from aiida import orm
from aiida_workgraph import WorkGraph, node, build_node
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import (
    seekpath_structure_analysis,
)

# we build a SeekpathNode Node
# Add only two outputs port here, because we only use these outputs in the following.
SeekpathNode = build_node(
    seekpath_structure_analysis,
    outputs=[["General", "primitive_structure"], ["General", "explicit_kpoints"]],
)


@node()
def inspect_relax(parameters):
    """Inspect relax calculation."""
    return orm.Int(parameters.get_dict()["number_of_bands"])


@node.calcfunction()
def update_scf_parameters(parameters, current_number_of_bands=None):
    """Update scf parameters."""
    parameters = parameters.get_dict()
    parameters.setdefault("SYSTEM", {}).setdefault("nbnd", current_number_of_bands)
    return orm.Dict(parameters)


@node.calcfunction()
def update_bands_parameters(parameters, scf_parameters, nbands_factor=None):
    """Update bands parameters."""
    parameters = parameters.get_dict()
    parameters.setdefault("SYSTEM", {})
    scf_parameters = scf_parameters.get_dict()
    if nbands_factor:
        factor = nbands_factor.value
        nbands = int(scf_parameters["number_of_bands"])
        nelectron = int(scf_parameters["number_of_electrons"])
        nbnd = max(int(0.5 * nelectron * factor), int(0.5 * nelectron) + 4, nbands)
        parameters["SYSTEM"]["nbnd"] = nbnd
    # Otherwise set the current number of bands, unless explicitly set in the inputs
    else:
        parameters["SYSTEM"].setdefault("nbnd", scf_parameters["number_of_bands"])
    return orm.Dict(parameters)


@node.graph_builder()
def bands_workgraph(
    structure: orm.StructureData = None,
    code: orm.Code = None,
    pseudo_family: str = None,
    pseudos: dict = None,
    inputs: dict = None,
    run_relax: bool = False,
    bands_kpoints_distance: float = None,
    nbands_factor: float = None,
) -> WorkGraph:
    """BandsWorkGraph."""
    inputs = {} if inputs is None else inputs
    # Initialize some variables which can be overridden in the following
    bands_kpoints = None
    current_number_of_bands = None
    # Load the pseudopotential family.
    if pseudo_family is not None:
        pseudo_family = orm.load_group(pseudo_family)
        pseudos = pseudo_family.get_pseudos(structure=structure)
    # Initialize the workgraph
    wg = WorkGraph("BandsStructure")
    # ------- relax -----------
    if run_relax:
        relax_node = wg.nodes.new(PwRelaxWorkChain, name="relax", structure=structure)
        # retrieve the relax inputs from the inputs, and set the relax inputs
        relax_inputs = inputs.get("relax", {})
        relax_inputs.update(
            {
                "base.pw.code": code,
                "base.pw.pseudos": pseudos,
            }
        )
        relax_node.set(relax_inputs)
        # override the input structure with the relaxed structure
        structure = relax_node.outputs["output_structure"]
        # -------- inspect_relax -----------
        inspect_relax_node = wg.nodes.new(
            inspect_relax,
            name="inspect_relax",
            parameters=relax_node.outputs["output_parameters"],
        )
        current_number_of_bands = inspect_relax_node.outputs["result"]
    # -------- seekpath -----------
    if bands_kpoints_distance is not None:
        seekpath_node = wg.nodes.new(
            SeekpathNode,
            name="seekpath",
            structure=structure,
            kwargs={"reference_distance": orm.Float(bands_kpoints_distance)},
        )
        structure = seekpath_node.outputs["primitive_structure"]
        # override the bands_kpoints
        bands_kpoints = seekpath_node.outputs["explicit_kpoints"]
    # -------- scf -----------
    # retrieve the scf inputs from the inputs, and update the scf parameters
    scf_inputs = inputs.get("scf", {"pw": {}})
    scf_parameters = wg.nodes.new(
        update_scf_parameters,
        name="scf_parameters",
        parameters=scf_inputs["pw"].get("parameters", {}),
        current_number_of_bands=current_number_of_bands,
    )
    scf_node = wg.nodes.new(PwBaseWorkChain, name="scf")
    # update inputs
    scf_inputs.update(
        {
            "pw.code": code,
            "pw.structure": structure,
            "pw.pseudos": pseudos,
            "pw.parameters": scf_parameters.outputs[0],
        }
    )
    scf_node.set(scf_inputs)
    # -------- bands -----------
    bands_inputs = inputs.get("bands", {"pw": {}})
    bands_parameters = wg.nodes.new(
        update_bands_parameters,
        name="bands_parameters",
        parameters=bands_inputs["pw"].get("parameters", {}),
        nbands_factor=nbands_factor,
        scf_parameters=scf_node.outputs["output_parameters"],
    )
    bands_node = wg.nodes.new(PwBaseWorkChain, name="bands", kpoints=bands_kpoints)
    bands_inputs.update(
        {
            "pw.code": code,
            "pw.structure": structure,
            "pw.pseudos": pseudos,
            "pw.parent_folder": scf_node.outputs["remote_folder"],
            "pw.parameters": bands_parameters.outputs[0],
        }
    )
    bands_node.set(bands_inputs)
    return wg
