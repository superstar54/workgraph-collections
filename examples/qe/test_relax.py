# -*- coding: utf-8 -*-
"""Test."""
from copy import deepcopy

from aiida import load_profile
from aiida.orm import Dict, KpointsData, StructureData, load_code, load_group
from ase.build import bulk

from workgraph_collections.qe.relax import relax_workgraph

load_profile()

atoms = bulk('Si')
structure_si = StructureData(ase=atoms)

code = load_code('qe-7.2-pw@localhost')
paras = Dict({
    'CONTROL': {
        'calculation': 'scf',
    },
    'SYSTEM': {
        'ecutwfc': 30,
        'ecutrho': 240,
        'occupations': 'smearing',
        'smearing': 'gaussian',
        'degauss': 0.1,
    },
})
relax_paras = deepcopy(paras)
relax_paras.get_dict()['CONTROL']['calculation'] = 'vc-relax'

kpoints = KpointsData()
kpoints.set_kpoints_mesh([3, 3, 3])
# Load the pseudopotential family.
pseudo_family = load_group('SSSP/1.3/PBEsol/efficiency')
pseudos = pseudo_family.get_pseudos(structure=structure_si)
#
metadata = {
    'options': {
        'resources': {
            'num_machines': 1,
            'num_mpiprocs_per_machine': 1,
        },
    }
}

relax_inputs = {
    'pw': {
        'code': code,
        'pseudos': pseudos,
        'parameters': relax_paras,
        'metadata': metadata,
    },
    'kpoints': kpoints,
}

wg = relax_workgraph(
    structure=structure_si,
    inputs=relax_inputs,
    max_iterations=5,
)
wg.run()
