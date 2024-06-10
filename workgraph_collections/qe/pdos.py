# -*- coding: utf-8 -*-
"""PdosWorkGraph."""

from aiida import orm
from aiida_workgraph import WorkGraph
from aiida_workgraph.decorator import node
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.calculations.dos import DosCalculation
from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation


@node()
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


@node()
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


@node.graph_builder()
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
    """Generate PdosWorkGraph."""
    inputs = {} if inputs is None else inputs
    # Load the pseudopotential family.
    if pseudo_family is not None:
        pseudo_family = orm.load_group(pseudo_family)
        pseudos = pseudo_family.get_pseudos(structure=structure)
    # create workgraph
    wg = WorkGraph("PDOS")
    # ------- relax -----------
    if run_relax:
        relax_node = wg.nodes.new(PwRelaxWorkChain, name="relax", structure=structure)
        relax_inputs = inputs.get("relax", {})
        relax_inputs.update(
            {
                "base.pw.code": pw_code,
                "base.pw.pseudos": pseudos,
            }
        )
        relax_node.set(relax_inputs)
        # override the structure
        structure = relax_node.outputs["output_structure"]
    # -------- scf -----------
    if run_scf:
        scf_node = wg.nodes.new(PwBaseWorkChain, name="scf")
        scf_inputs = inputs.get("scf", {})
        scf_inputs.update(
            {"pw.structure": structure, "pw.code": pw_code, "pw.pseudos": pseudos}
        )
        scf_node.set(scf_inputs)
        scf_parent_folder = scf_node.outputs["remote_folder"]
    # -------- nscf -----------
    nscf_node = wg.nodes.new(PwBaseWorkChain, name="nscf")
    nscf_inputs = inputs.get("nscf", {})
    nscf_inputs.update(
        {
            "pw.structure": structure,
            "pw.code": pw_code,
            "pw.parent_folder": scf_parent_folder,
            "pw.pseudos": pseudos,
        }
    )
    nscf_node.set(nscf_inputs)
    # -------- dos -----------
    dos1 = wg.nodes.new(DosCalculation, name="dos")
    dos_input = inputs.get("dos", {})
    dos_input.update({"code": dos_code})
    dos1.set(dos_input)
    dos_parameters = wg.nodes.new(
        generate_dos_parameters,
        name="dos_parameters",
        parameters=dos_input.get("parameters"),
    )
    wg.links.new(nscf_node.outputs["remote_folder"], dos1.inputs["parent_folder"])
    wg.links.new(nscf_node.outputs["_outputs"], dos_parameters.inputs["nscf_outputs"])
    wg.links.new(dos_parameters.outputs[0], dos1.inputs["parameters"])
    # -------- projwfc -----------
    projwfc1 = wg.nodes.new(ProjwfcCalculation, name="projwfc")
    projwfc_inputs = inputs.get("projwfc", {})
    projwfc_inputs.update({"code": projwfc_code})
    projwfc1.set(projwfc_inputs)
    projwfc_parameters = wg.nodes.new(
        generate_projwfc_parameters,
        name="projwfc_parameters",
        parameters=projwfc_inputs.get("parameters"),
    )
    wg.links.new(nscf_node.outputs["remote_folder"], projwfc1.inputs["parent_folder"])
    wg.links.new(
        nscf_node.outputs["_outputs"], projwfc_parameters.inputs["nscf_outputs"]
    )
    wg.links.new(projwfc_parameters.outputs[0], projwfc1.inputs["parameters"])
    return wg
