# -*- coding: utf-8 -*-
"""PdosWorkGraph."""

from aiida import orm
from aiida_workgraph import WorkGraph
from aiida_workgraph.decorator import node
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.calculations.dos import DosCalculation
from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation

@node()
def generate_dos_parameters(nscf_outputs, parameters=None):
    """Generate DOS parameters from NSCF calculation."""
    nscf_emin = nscf_outputs.output_band.get_array('bands').min()
    nscf_emax = nscf_outputs.output_band.get_array('bands').max()
    nscf_fermi = nscf_outputs.output_parameters.dict.fermi_energy
    paras = {} if parameters is None else parameters.get_dict()
    paras.setdefault('DOS', {})
    if paras.pop('align_to_fermi', False):
        paras['DOS'].setdefault('Emax', nscf_emax)
        paras['DOS']['Emin'] = paras['DOS'].get('Emin', nscf_emin) + nscf_fermi
        paras['DOS']['Emax'] = paras['DOS'].get('Emin', nscf_emin) + nscf_fermi
    return orm.Dict(paras)


@node()
def generate_projwfc_parameters(nscf_outputs, parameters=None):
    """Generate PROJWFC parameters from NSCF calculation."""
    nscf_emin = nscf_outputs.output_band.get_array('bands').min()
    nscf_emax = nscf_outputs.output_band.get_array('bands').max()
    nscf_fermi = nscf_outputs.output_parameters.dict.fermi_energy
    paras = {} if parameters is None else parameters.get_dict()
    paras.setdefault('PROJWFC', {})
    if paras.pop('align_to_fermi', False):
        paras['PROJWFC']['Emin'] = paras['PROJWFC'].get('Emin', nscf_emin) + nscf_fermi
        paras['PROJWFC']['Emax'] = paras['PROJWFC'].get('Emax', nscf_emax) + nscf_fermi
    return orm.Dict(paras)


@node.graph_builder()
def pdos_workgraph(structure=None, inputs=None, run_scf=True):
    """Generate PdosWorkGraph."""
    inputs = {} if inputs is None else inputs
    # create workgraph
    tree = WorkGraph()
    tree.ctx = {'current_structure': structure, 'current_number_of_bands': None, 'parent_folder': None}
    # -------- scf -----------
    scf_node = tree.nodes.new(PwBaseWorkChain, name='scf')
    scf_inputs = inputs.get('scf', {})
    scf_inputs['pw.structure'] = structure
    scf_node.set(scf_inputs)
    scf_node.to_ctx = [['remote_folder', 'scf_parent_folder']]
    # -------- nscf -----------
    nscf_node = tree.nodes.new(PwBaseWorkChain, name='nscf')
    nscf_inputs = inputs.get('nscf', {})
    nscf_inputs['pw.structure'] = structure
    if run_scf:
        nscf_inputs['pw.parent_folder'] = '{{scf_parent_folder}}'
    nscf_node.set(nscf_inputs)
    # -------- dos -----------
    dos1 = tree.nodes.new(DosCalculation, name='dos')
    dos_input = inputs.get('dos', {})
    dos1.set(dos_input)
    dos_parameters = tree.nodes.new(
        generate_dos_parameters, name='dos_parameters', parameters=dos_input.get('parameters')
    )
    tree.links.new(nscf_node.outputs['remote_folder'], dos1.inputs['parent_folder'])
    tree.links.new(nscf_node.outputs['_outputs'], dos_parameters.inputs['nscf_outputs'])
    tree.links.new(dos_parameters.outputs[0], dos1.inputs['parameters'])
    # -------- projwfc -----------
    projwfc1 = tree.nodes.new(ProjwfcCalculation, name='projwfc')
    projwfc_inputs = inputs.get('projwfc', {})
    projwfc1.set(projwfc_inputs)
    projwfc_parameters = tree.nodes.new(
        generate_projwfc_parameters, name='projwfc_parameters', parameters=projwfc_inputs.get('parameters')
    )
    tree.links.new(nscf_node.outputs['remote_folder'], projwfc1.inputs['parent_folder'])
    tree.links.new(nscf_node.outputs['_outputs'], projwfc_parameters.inputs['nscf_outputs'])
    tree.links.new(projwfc_parameters.outputs[0], projwfc1.inputs['parameters'])
    # -------- dependences -----------
    nscf_node.wait = ['scf']
    if not run_scf:
        tree.nodes.delete('scf')
    # export workgraph
    return tree
