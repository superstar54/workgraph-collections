"""BandsWorkGraph."""

from aiida import orm
from aiida_workgraph import WorkGraph
from aiida_workgraph.decorator import node
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import seekpath_structure_analysis

# define some utility nodes
@node()
def inspect_relax(outputs):
    """Inspect relax calculation."""
    current_number_of_bands = outputs.output_parameters.get_dict()['number_of_bands']
    return {'current_number_of_bands': orm.Int(current_number_of_bands), 'current_structure': outputs.output_structure}


@node.calcfunction()
def generate_scf_parameters(parameters, current_number_of_bands=None):
    """Generate scf parameters from relax calculation."""
    parameters = parameters.get_dict()
    parameters.setdefault('SYSTEM', {})
    if not current_number_of_bands:
        current_number_of_bands = parameters['SYSTEM'].get('nbnd')
    return orm.Dict(parameters)


@node()
def inspect_scf(outputs):
    """Inspect scf calculation.
    outputs is the outputs of the scf calculation."""
    current_number_of_bands = outputs.output_parameters.get_dict()['number_of_bands']
    return {'current_number_of_bands': orm.Int(current_number_of_bands)}


@node.calcfunction()
def generate_bands_parameters(parameters, output_parameters, nbands_factor=None):
    """Generate bands parameters from SCF calculation."""
    parameters = parameters.get_dict()
    parameters.setdefault('SYSTEM', {})
    if nbands_factor:
        factor = nbands_factor.value
        parameters = output_parameters.get_dict()
        nbands = int(parameters['number_of_bands'])
        nelectron = int(parameters['number_of_electrons'])
        nbnd = max(int(0.5 * nelectron * factor), int(0.5 * nelectron) + 4, nbands)
        parameters['SYSTEM']['nbnd'] = nbnd
    # Otherwise set the current number of bands, unless explicitly set in the inputs
    else:
        parameters['SYSTEM'].setdefault('nbnd', output_parameters.base.attributes.get('number_of_bands'))
    return orm.Dict(parameters)


@node.graph_builder()
def bands_workgraph(structure=None, inputs=None, run_relax=False, bands_kpoints_distance=None, nbands_factor=None):
    """BandsWorkGraph."""
    inputs = {} if inputs is None else inputs
    # create workgraph
    tree = WorkGraph()
    tree.ctx = {'current_structure': structure, 'current_number_of_bands': None, 'bands_kpoints': None}
    # ------- relax -----------
    relax_node = tree.nodes.new(PwRelaxWorkChain, name='relax')
    relax_inputs = inputs.get('relax', {})
    relax_inputs['structure'] = '{{current_structure}}'
    relax_node.set(relax_inputs)
    inspect_relax_node = tree.nodes.new(inspect_relax, name='inspect_relax')
    tree.links.new(relax_node.outputs['_outputs'], inspect_relax_node.inputs['outputs'])
    inspect_relax_node.to_ctx = [['current_number_of_bands', 'current_number_of_bands'],
                                 ['current_structure', 'current_structure']]
    # -------- seekpath -----------
    seekpath_node = tree.nodes.new(
        seekpath_structure_analysis,
        name='seekpath',
        structure='{{current_structure}}',
        kwargs={'reference_distance': orm.Float(bands_kpoints_distance)},
    )
    seekpath_node.to_ctx = [['primitive_structure', 'current_structure'], ['explicit_kpoints', 'bands_kpoints']]
    # -------- scf -----------
    scf_inputs = inputs.get('scf', {"pw": {}})
    scf_inputs['pw.structure'] = '{{current_structure}}'
    scf_node = tree.nodes.new(PwBaseWorkChain, name='scf')
    scf_node.set(scf_inputs)
    scf_parameters = tree.nodes.new(
        generate_scf_parameters,
        name='scf_parameters',
        parameters=scf_inputs['pw'].get('parameters', {}),
        current_number_of_bands='{{current_number_of_bands}}'
    )
    tree.links.new(scf_parameters.outputs[0], scf_node.inputs['pw.parameters'])
    inspect_scf_node = tree.nodes.new(inspect_scf, name='inspect_scf')
    tree.links.new(scf_node.outputs['_outputs'], inspect_scf_node.inputs['outputs'])
    # -------- bands -----------
    bands_node = tree.nodes.new(PwBaseWorkChain, name='bands')
    bands_inputs = inputs.get('bands', {"pw": {}})
    bands_inputs['pw.structure'] = '{{current_structure}}'
    bands_node.set(bands_inputs)
    bands_parameters = tree.nodes.new(
        generate_bands_parameters,
        name='bands_parameters',
        parameters=bands_inputs['pw'].get('parameters', {}),
        nbands_factor=nbands_factor,
    )
    tree.links.new(scf_node.outputs['remote_folder'], bands_node.inputs['pw.parent_folder'])
    tree.links.new(scf_node.outputs['output_parameters'], bands_parameters.inputs['output_parameters'])
    tree.links.new(bands_parameters.outputs[0], bands_node.inputs['pw.parameters'])
    # -------- dependences -----------
    seekpath_node.wait = ['inspect_relax']
    scf_parameters.wait = ['inspect_relax', 'seekpath']
    bands_parameters.wait = ['inspect_scf']
    # delete nodes
    if not run_relax:
        tree.nodes.delete('relax')
        tree.nodes.delete('inspect_relax')
    if not bands_kpoints_distance:
        tree.nodes.delete('seekpath')
    # export workgraph
    return tree
