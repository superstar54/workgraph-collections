"""BandsWorkGraph."""

from aiida import orm
from aiida_workgraph import WorkGraph
from aiida_workgraph.decorator import node
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import (
    seekpath_structure_analysis,
)


@node()
def inspect_relax(outputs):
    """Inspect relax calculation."""
    current_number_of_bands = outputs.output_parameters.get_dict()["number_of_bands"]
    return {
        "current_number_of_bands": orm.Int(current_number_of_bands),
        "current_structure": outputs.output_structure,
    }


@node.calcfunction()
def generate_scf_parameters(parameters, current_number_of_bands=None):
    """Generate scf parameters from relax calculation."""
    parameters = parameters.get_dict()
    parameters.setdefault("SYSTEM", {})
    if not current_number_of_bands:
        current_number_of_bands = parameters["SYSTEM"].get("nbnd")
    return orm.Dict(parameters)


@node()
def inspect_scf(outputs):
    """Inspect scf calculation.
    outputs is the outputs of the scf calculation."""
    current_number_of_bands = outputs.output_parameters.get_dict()["number_of_bands"]
    return {"current_number_of_bands": orm.Int(current_number_of_bands)}


@node.calcfunction()
def generate_bands_parameters(parameters, output_parameters, nbands_factor=None):
    """Generate bands parameters from SCF calculation."""
    parameters = parameters.get_dict()
    parameters.setdefault("SYSTEM", {})
    if nbands_factor:
        factor = nbands_factor.value
        parameters = output_parameters.get_dict()
        nbands = int(parameters["number_of_bands"])
        nelectron = int(parameters["number_of_electrons"])
        nbnd = max(int(0.5 * nelectron * factor), int(0.5 * nelectron) + 4, nbands)
        parameters["SYSTEM"]["nbnd"] = nbnd
    # Otherwise set the current number of bands, unless explicitly set in the inputs
    else:
        parameters["SYSTEM"].setdefault(
            "nbnd", output_parameters.base.attributes.get("number_of_bands")
        )
    return orm.Dict(parameters)


@node.graph_builder()
def bands_workgraph(
    structure: orm.StructureData = None,
    code: orm.Code = None,
    protocol: str = "fast",
    inputs: dict = None,
    pseudo_family: str = None,
    pseudos: dict = None,
    run_relax: bool = False,
    bands_kpoints_distance: float = 0.1,
    nbands_factor: orm.Float = None,
) -> WorkGraph:
    """BandsWorkGraph."""
    inputs = {} if inputs is None else inputs
    # Load the pseudopotential family.
    if pseudo_family is not None:
        pseudo_family = orm.load_group(pseudo_family)
        pseudos = pseudo_family.get_pseudos(structure=structure)
    # create workgraph
    wg = WorkGraph("BandsStructure")
    wg.context = {
        "current_structure": structure,
        "current_number_of_bands": None,
        "bands_kpoints": None,
    }
    # ------- relax -----------
    relax_node = wg.nodes.new(PwRelaxWorkChain, name="relax")
    relax_inputs = PwRelaxWorkChain.get_protocol_inputs(protocol)
    relax_inputs = inputs.get("relax", {})
    relax_inputs.update(
        {
            "structure": "{{current_structure}}",
            "base.pw.code": code,
            "base.pw.pseudos": pseudos,
        }
    )
    relax_node.set(relax_inputs)
    inspect_relax_node = wg.nodes.new(inspect_relax, name="inspect_relax")
    wg.links.new(relax_node.outputs["_outputs"], inspect_relax_node.inputs["outputs"])
    inspect_relax_node.to_context = [
        ["current_number_of_bands", "current_number_of_bands"],
        ["current_structure", "current_structure"],
    ]
    # -------- seekpath -----------
    seekpath_node = wg.nodes.new(
        seekpath_structure_analysis,
        name="seekpath",
        structure="{{current_structure}}",
        kwargs={"reference_distance": orm.Float(bands_kpoints_distance)},
    )
    seekpath_node.to_context = [
        ["primitive_structure", "current_structure"],
        ["explicit_kpoints", "bands_kpoints"],
    ]
    # -------- scf -----------
    scf_inputs = inputs.get("scf", {"pw": {}})
    scf_inputs.update(
        {
            "pw.code": code,
            "pw.structure": "{{current_structure}}",
            "pw.pseudos": pseudos,
        }
    )
    scf_node = wg.nodes.new(PwBaseWorkChain, name="scf")
    scf_node.set(scf_inputs)
    scf_parameters = wg.nodes.new(
        generate_scf_parameters,
        name="scf_parameters",
        parameters=scf_inputs["pw"].get("parameters", {}),
        current_number_of_bands="{{current_number_of_bands}}",
    )
    wg.links.new(scf_parameters.outputs[0], scf_node.inputs["pw.parameters"])
    inspect_scf_node = wg.nodes.new(inspect_scf, name="inspect_scf")
    wg.links.new(scf_node.outputs["_outputs"], inspect_scf_node.inputs["outputs"])
    # -------- bands -----------
    bands_node = wg.nodes.new(PwBaseWorkChain, name="bands")
    bands_inputs = inputs.get("bands", {"pw": {}})
    bands_inputs.update(
        {
            "pw.code": code,
            "pw.structure": "{{current_structure}}",
            "pw.pseudos": pseudos,
        }
    )
    bands_node.set(bands_inputs)
    bands_parameters = wg.nodes.new(
        generate_bands_parameters,
        name="bands_parameters",
        parameters=bands_inputs["pw"].get("parameters", {}),
        nbands_factor=nbands_factor,
    )
    wg.links.new(
        scf_node.outputs["remote_folder"], bands_node.inputs["pw.parent_folder"]
    )
    wg.links.new(
        scf_node.outputs["output_parameters"],
        bands_parameters.inputs["output_parameters"],
    )
    wg.links.new(bands_parameters.outputs[0], bands_node.inputs["pw.parameters"])
    # -------- dependences -----------
    seekpath_node.wait = ["inspect_relax"]
    scf_parameters.wait = ["inspect_relax", "seekpath"]
    bands_parameters.wait = ["inspect_scf"]
    # delete nodes
    if not run_relax:
        wg.nodes.delete("relax")
        wg.nodes.delete("inspect_relax")
    if not bands_kpoints_distance:
        wg.nodes.delete("seekpath")
    # export workgraph
    return wg
