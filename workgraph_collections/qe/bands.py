"""BandsWorkGraph."""

from aiida import orm
from aiida_workgraph import task, spec
from node_graph.socket_spec import select
from workgraph_collections.qe import PwBaseTask, PwRelaxTask, SeekpathTask
from typing import Annotated


@task.calcfunction()
def inspect_relax(parameters):
    """Inspect relax calculation."""
    return orm.Int(parameters.get_dict()["number_of_bands"])


@task.calcfunction()
def update_scf_parameters(parameters, current_number_of_bands=None):
    """Update scf parameters."""
    parameters = parameters.get_dict()
    parameters.setdefault("SYSTEM", {}).setdefault("nbnd", current_number_of_bands)
    return orm.Dict(parameters)


@task.calcfunction()
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


@task.graph()
def BandsWorkgraph(
    structure: orm.StructureData = None,
    code: orm.Code = None,
    pseudo_family: str = None,
    pseudos: Annotated[dict, spec.dynamic(orm.UpfData)] = None,
    inputs: Annotated[
        dict,
        spec.namespace(
            relax=Annotated[dict, PwRelaxTask.inputs, select(exclude="structure")],
            scf=Annotated[dict, PwBaseTask.inputs, select(exclude="pw.structure")],
            bands=Annotated[dict, PwBaseTask.inputs, select(exclude="pw.structure")],
        ),
    ] = None,
    run_relax: bool = False,
    bands_kpoints_distance: float = None,
    nbands_factor: float = None,
) -> Annotated[dict, PwBaseTask.outputs]:
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
    # ------- relax -----------
    if run_relax:
        # retrieve the relax inputs from the inputs, and set the relax inputs
        relax_inputs = inputs.get("relax", {})
        relax_inputs.update(
            {
                "structure": structure,
                "base.pw.code": code,
                "base.pw.pseudos": pseudos,
            }
        )
        relax_task_out = PwRelaxTask(**relax_inputs)
        # override the input structure with the relaxed structure
        structure = relax_task_out.output_structure
        # -------- inspect_relax -----------
        inspect_relax_out = inspect_relax(parameters=relax_task_out.output_parameters)
        current_number_of_bands = inspect_relax_out.result
    # -------- seekpath -----------
    if bands_kpoints_distance is not None:
        seekpath_task_out = SeekpathTask(
            structure=structure,
            kwargs={"reference_distance": orm.Float(bands_kpoints_distance)},
        )
        structure = seekpath_task_out.primitive_structure
        bands_kpoints = seekpath_task_out.explicit_kpoints
    # -------- scf -----------
    # retrieve the scf inputs from the inputs, and update the scf parameters
    scf_inputs = inputs.get("scf", {})
    scf_inputs.setdefault("pw", {})
    scf_parameters_out = update_scf_parameters(
        parameters=scf_inputs["pw"].get("parameters", {}),
        current_number_of_bands=current_number_of_bands,
    )
    scf_inputs["pw"].update(
        {
            "code": code,
            "structure": structure,
            "pseudos": pseudos,
            "parameters": scf_parameters_out.result,
        }
    )
    scf_task_out = PwBaseTask(**scf_inputs)
    # -------- bands -----------
    bands_inputs = inputs.get("bands", {"pw": {}})
    bands_inputs.setdefault("pw", {})
    bands_parameters_out = update_bands_parameters(
        parameters=bands_inputs["pw"].get("parameters", {}),
        nbands_factor=nbands_factor,
        scf_parameters=scf_task_out.output_parameters,
    )
    bands_inputs["kpoints"] = bands_kpoints
    bands_inputs["pw"].update(
        {
            "code": code,
            "structure": structure,
            "pseudos": pseudos,
            "parent_folder": scf_task_out.remote_folder,
            "parameters": bands_parameters_out.result,
        }
    )
    bands_task_out = PwBaseTask(**bands_inputs)
    return bands_task_out
