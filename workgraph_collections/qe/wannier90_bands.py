from aiida import orm
from aiida_workgraph import WorkGraph, task, build_node
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_wannier90_workflows.workflows.base.wannier90 import Wannier90BaseWorkChain
from aiida_wannier90_workflows.workflows.base.projwfc import ProjwfcBaseWorkChain
from aiida_wannier90_workflows.workflows.base.pw2wannier90 import (
    Pw2wannier90BaseWorkChain,
)
from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import (
    seekpath_structure_analysis,
)


SeekpathNode = build_node(
    seekpath_structure_analysis,
    outputs=[
        ["General", "primitive_structure"],
        ["General", "explicit_kpoints"],
        ["General", "parameters"],
        ["General", "conv_structure"],
    ],
)


@task.calcfunction(outputs=[["General", "kpoint_path"]])
def inspect_seekpath(parameters):
    """Inspect seekpath calculation."""
    parameters = parameters.get_dict()
    kpoint_path = orm.Dict(
        dict={
            "path": parameters["path"],
            "point_coords": parameters["point_coords"],
        }
    )
    return kpoint_path


@task.calcfunction()
def prepare_wannier90_pp_inputs(
    parameters, scf_output_parameters=None, nscf_output_parameters=None
):
    """Prepare the inputs of wannier90 calculation before submission."""
    parameters = parameters.get_dict()
    if scf_output_parameters is not None:
        parameters["fermi_energy"] = scf_output_parameters.get_dict().get(
            "fermi_energy"
        )
    elif nscf_output_parameters is not None:
        parameters["fermi_energy"] = nscf_output_parameters.get_dict().get(
            "fermi_energy"
        )
    else:
        if "fermi_energy" not in parameters:
            raise ValueError("Fermi energy not found in scf or nscf output")
    return orm.Dict(parameters)


@task.graph_builder()
def wannier90_bands_workgraph(
    structure: orm.StructureData = None,
    codes: dict = None,
    inputs: dict = None,
    projection_type: str = "SCDM",
    frozen_type: str = None,
    bands_kpoints_distance: float = None,
    kpoint_path: orm.KpointsData = None,
    bands_kpoints: orm.KpointsData = None,
    bands: orm.BandsData = None,
    bands_projections: orm.ProjectionData = None,
    scf_parent_folder: orm.RemoteData = None,
    nscf_parent_folder: orm.RemoteData = None,
    run_scf: bool = True,
    run_nscf: bool = True,
    run_projwfc: bool = True,
):
    """Generate PdosWorkGraph."""

    inputs = {} if inputs is None else inputs
    codes = {} if codes is None else codes
    scf_inputs = inputs.get("scf", {})
    nscf_inputs = inputs.get("nscf", {})
    projwfc_inputs = inputs.get("projwfc", {})
    pw2wannier90_inputs = inputs.get("pw2wannier90", {})
    wannier90_pp_inputs = inputs.get("wannier90_pp", {"wannier90": {"parameters": {}}})
    wannier90_inputs = inputs.get("wannier90", {"wannier90": {"parameters": {}}})
    # initialize variables which will be overriden later
    scf_output_parameters = None
    nscf_output_parameters = None

    if projection_type.upper() == "SCDM":
        run_projwfc = True
    else:
        if frozen_type.upper() == "ENERGY_AUTO":
            run_projwfc = True
        else:
            run_projwfc = False

    # create workgraph
    wg = WorkGraph("Wannier90")
    wg.context = {}
    # -------- seekpath -----------
    if bands_kpoints_distance is not None:
        seekpath_node = wg.tasks.new(
            SeekpathNode,
            name="seekpath",
            structure=structure,
            kwargs={"reference_distance": orm.Float(bands_kpoints_distance)},
        )
        structure = seekpath_node.outputs["primitive_structure"]
        inspect_seekpath_node = wg.tasks.new(
            inspect_seekpath,
            name="inspect_seekpath",
            parameters=seekpath_node.outputs["parameters"],
        )
        kpoint_path = inspect_seekpath_node.outputs["kpoint_path"]
    # -------- scf -----------
    if run_scf:
        scf_task = wg.tasks.new(PwBaseWorkChain, name="scf")
        scf_inputs.update({"pw.structure": structure, "pw.code": codes.get("pw")})
        scf_task.set(scf_inputs)
        scf_parent_folder = scf_task.outputs["remote_folder"]
        scf_output_parameters = scf_task.outputs["output_parameters"]
        output_band = scf_task.outputs["output_band"]
    # -------- nscf -----------
    if run_nscf:
        nscf_task = wg.tasks.new(PwBaseWorkChain, name="nscf")
        nscf_inputs.update(
            {
                "pw.structure": structure,
                "pw.code": codes.get("pw"),
                "pw.parent_folder": scf_parent_folder,
            }
        )
        nscf_task.set(nscf_inputs)
        nscf_parent_folder = nscf_task.outputs["remote_folder"]
        nscf_output_parameters = nscf_task.outputs["output_parameters"]
        output_band = nscf_task.outputs["output_band"]
    # -------- projwfc -----------
    if run_projwfc:
        projwfc_task = wg.tasks.new(
            ProjwfcBaseWorkChain,
            name="projwfc",
        )
        projwfc_inputs.update(
            {
                "projwfc.code": codes.get("projwfc"),
                "projwfc.parent_folder": nscf_parent_folder,
            }
        )
        projwfc_task.set(projwfc_inputs)
        bands = projwfc_task.outputs["bands"]
        bands_projections = projwfc_task.outputs["projections"]
    # -------- wannier90_pp -----------
    wannier90_pp_parameters = wg.tasks.new(
        prepare_wannier90_pp_inputs,
        name="wannier90_pp_parameters",
        parameters=wannier90_pp_inputs["wannier90"].get("parameters"),
        scf_output_parameters=scf_output_parameters,
        nscf_output_parameters=nscf_output_parameters,
    )
    wannier90_pp = wg.tasks.new(
        Wannier90BaseWorkChain,
        name="wannier90_pp",
    )
    wannier90_pp_inputs.update(
        {
            "wannier90.structure": structure,
            "wannier90.code": codes.get("wannier90"),
            "wannier90.bands_kpoints": bands_kpoints,
            "wannier90.kpoint_path": kpoint_path,
            "wannier90.parameters": wannier90_pp_parameters.outputs[0],
        }
    )
    if (
        wannier90_pp_inputs.get("shift_energy_windows", False)
        and "bands" not in wannier90_pp_inputs
    ):
        wannier90_pp_inputs.update({"bands": output_band})
    if wannier90_pp_inputs.get("auto_energy_windows", False):
        wannier90_pp_inputs.update(
            {"bands": bands, "bands_projections": bands_projections}
        )
    wannier90_pp.set(wannier90_pp_inputs)
    # -------- pw2wannier90 -----------
    p2w_inputs = pw2wannier90_inputs.get("pw2wannier90", {})
    pw2wannier90_inputs.update(
        {
            "pw2wannier90.code": codes.get("pw2wannier90"),
            "pw2wannier90.nnkp_file": wannier90_pp.outputs["nnkp_file"],
            "pw2wannier90.parent_folder": nscf_parent_folder,
        }
    )
    parameters = (
        p2w_inputs.get("parameters", orm.Dict({})).get_dict().get("inputpp", {})
    )
    scdm_proj = parameters.get("scdm_proj", False)
    scdm_entanglement = parameters.get("scdm_entanglement", None)
    scdm_mu = parameters.get("scdm_mu", None)
    scdm_sigma = parameters.get("scdm_sigma", None)
    fit_scdm = (
        scdm_proj
        and scdm_entanglement == "erfc"
        and (scdm_mu is None or scdm_sigma is None)
    )
    if fit_scdm:
        if not run_projwfc:
            raise ValueError("Needs to run projwfc for SCDM projection")
        pw2wannier90_inputs.update(
            {
                "bands": bands,
                "bands_projections": bands_projections,
            }
        )

    pw2wannier90 = wg.tasks.new(
        Pw2wannier90BaseWorkChain,
        name="pw2wannier90",
    )
    pw2wannier90.set(pw2wannier90_inputs)
    # -------- wannier90 -----------
    wannier90 = wg.tasks.new(Wannier90BaseWorkChain, name="wannier90")
    wannier90_inputs.pop("shift_energy_windows", None)
    wannier90_inputs.pop("auto_energy_windows", None)
    wannier90_inputs.pop("auto_energy_windows_threshold", None)
    wannier90_inputs.pop("bands", None)
    wannier90_inputs.pop("bands_projections", None)
    wannier90_inputs.update(
        {
            "wannier90.structure": structure,
            "wannier90.code": codes.get("wannier90"),
            "wannier90.remote_input_folder": pw2wannier90.outputs["remote_folder"],
            "wannier90.kpoint_path": inspect_seekpath_node.outputs["kpoint_path"],
        }
    )
    wannier90.set(wannier90_inputs)
    return wg
