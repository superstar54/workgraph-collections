# -*- coding: utf-8 -*-
"""Test."""
from copy import deepcopy

from aiida import load_profile
from aiida.orm import Dict, Float, KpointsData, StructureData, load_code, load_group
from aiida_workgraph import WorkGraph
from ase.build import bulk

from workgraph_collections.qe.bands import bands_workgraph

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
bands_paras = deepcopy(paras)
bands_paras.get_dict()['CONTROL']['calculation'] = 'bands'

kpoints = KpointsData()
kpoints.set_kpoints_mesh([1, 1, 1])
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

bands_inputs = {
    'relax': {
        'base': {
            'pw': {
                'code': code,
                'pseudos': pseudos,
                'parameters': relax_paras,
                'metadata': metadata,
            },
            'kpoints': kpoints,
        },
    },
    'scf': {
        'pw': {
            'code': code,
            'pseudos': pseudos,
            'parameters': paras,
            'metadata': metadata,
        },
        'kpoints': kpoints,
    },
    'bands': {
        'pw': {
            'code': code,
            'pseudos': pseudos,
            'parameters': bands_paras,
            'metadata': metadata,
        },
        'kpoints': kpoints,
    },
}

wg = WorkGraph('Bands')
wg = bands_workgraph(
    structure=structure_si,
    inputs=bands_inputs,
    run_relax=True,
    bands_kpoints_distance=Float(0.2),
)
wg.run()
