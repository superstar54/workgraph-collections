# -*- coding: utf-8 -*-
"""PdosWorkGraph."""

from aiida import orm
from aiida_workgraph import task
from workgraph_collections.qe import PwBaseTask, PwRelaxTask, DosTask, ProjwfcTask


@task()
def generate_dos_parameters(nscf_outputs, parameters=None):
    """Generate DOS parameters from NSCF calculation."""
    nscf_emin = nscf_outputs.output_band.get_array("bands").min()
    nscf_emax = nscf_outputs.output_band.get_array("bands").max()
    nscf_fermi = nscf_outputs.output_parameters.dict.fermi_energy
    paras = {} if parameters is None else parameters.get_dict()
    paras.setdefault("DOS", {})
    if paras.pop("align_to_fermi", False):
        paras["DOS"].setdefault("Emax", nscf_emax)
        paras["DOS"]["Emin"] = paras["DOS"].get("Emin", nscf_emin) + nscf_fermi
        paras["DOS"]["Emax"] = paras["DOS"].get("Emin", nscf_emin) + nscf_fermi
    return orm.Dict(paras)


@task()
def generate_projwfc_parameters(nscf_outputs, parameters=None):
    """Generate PROJWFC parameters from NSCF calculation."""
    nscf_emin = nscf_outputs.output_band.get_array("bands").min()
    nscf_emax = nscf_outputs.output_band.get_array("bands").max()
    nscf_fermi = nscf_outputs.output_parameters.dict.fermi_energy
    paras = {} if parameters is None else parameters.get_dict()
    paras.setdefault("PROJWFC", {})
    if paras.pop("align_to_fermi", False):
        paras["PROJWFC"]["Emin"] = paras["PROJWFC"].get("Emin", nscf_emin) + nscf_fermi
        paras["PROJWFC"]["Emax"] = paras["PROJWFC"].get("Emax", nscf_emax) + nscf_fermi
    return orm.Dict(paras)


@task.graph(
    outputs=[
        {"name": "dos", "identifier": "workgraph.namespace"},
        {"name": "projwfc", "identifier": "workgraph.namespace"},
    ]
)
def pdos_workgraph(
    structure: orm.StructureData = None,
    pw_code: orm.Code = None,
    dos_code: orm.Code = None,
    projwfc_code: orm.Code = None,
    inputs: dict = None,
    pseudo_family: str = None,
    pseudos: dict = None,
    scf_parent_folder: orm.RemoteData = None,
    run_scf: bool = False,
    run_relax: bool = False,
):
    """Workgraph to run a full PDOS calculation."""

    inputs = {} if inputs is None else inputs
    # Load the pseudopotential family.
    if pseudo_family is not None:
        pseudo_family = orm.load_group(pseudo_family)
        pseudos = pseudo_family.get_pseudos(structure=structure)
    # ------- relax -----------
    if run_relax:
        relax_inputs = inputs.get("relax", {})
        relax_inputs.update(
            {
                "base.pw.code": pw_code,
                "base.pw.pseudos": pseudos,
            }
        )
        relax_outs = PwRelaxTask(structure=structure, **relax_inputs)
        # override the structure
        structure = relax_outs.output_structure
    # -------- scf -----------
    if run_scf:
        scf_inputs = inputs.get("scf", {})
        scf_inputs.update(
            {"pw.structure": structure, "pw.code": pw_code, "pw.pseudos": pseudos}
        )
        scf_outs = PwBaseTask(**scf_inputs)
        scf_parent_folder = scf_outs.remote_folder
    # -------- nscf -----------
    nscf_inputs = inputs.get("nscf", {})
    nscf_inputs.update(
        {
            "pw.structure": structure,
            "pw.code": pw_code,
            "pw.parent_folder": scf_parent_folder,
            "pw.pseudos": pseudos,
        }
    )
    nscf_outs = PwBaseTask(**nscf_inputs)
    # -------- dos -----------
    dos_input = inputs.get("dos", {})
    dos_input.update({"code": dos_code})
    dos_parameters_outs = generate_dos_parameters(
        name="dos_parameters",
        parameters=dos_input.pop("parameters", {}),
    )
    dos_input.update(
        {
            "parent_folder": nscf_outs.remote_folder,
            "nscf_outputs": nscf_outs._outputs,
            "parameters": dos_parameters_outs.result,
        }
    )
    dos_outs = DosTask(**dos_input)
    # -------- projwfc -----------
    projwfc_inputs = inputs.get("projwfc", {})
    projwfc_inputs.update({"code": projwfc_code})
    projwfc_parameters_outs = generate_projwfc_parameters(
        name="projwfc_parameters",
        parameters=projwfc_inputs.pop("parameters", {}),
    )
    projwfc_inputs.update(
        {
            "parent_folder": nscf_outs.remote_folder,
            "nscf_outputs": nscf_outs._outputs,
            "parameters": projwfc_parameters_outs.result,
        }
    )
    projwfc_outs = ProjwfcTask(**projwfc_inputs)
    return {"dos": dos_outs, "projwfc": projwfc_outs}
