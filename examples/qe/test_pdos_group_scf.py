# -*- coding: utf-8 -*-
"""Test."""
from copy import deepcopy

from aiida import load_profile
from aiida.orm import Dict, KpointsData, StructureData, load_code, load_group
from ase.build import bulk

from workgraph_collections.qe.pdos import pdos_workgraph

load_profile()

atoms = bulk('Si')
structure_si = StructureData(ase=atoms)

code = load_code('qe-7.2-pw@localhost')
dos_code = load_code('qe-7.2-dos@localhost')
projwfc_code = load_code('qe-7.2-projwfc@localhost')
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
nscf_paras = deepcopy(paras)
nscf_paras.get_dict()['CONTROL']['calculation'] = 'nscf'

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

pdos_inputs = {
    'scf': {
        'pw': {
            'code': code,
            'pseudos': pseudos,
            'parameters': paras,
            'metadata': metadata,
        },
        'kpoints': kpoints,
    },
    'nscf': {
        'pw': {
            'code': code,
            'pseudos': pseudos,
            'parameters': nscf_paras,
            'metadata': metadata,
        },
        'kpoints': kpoints,
    },
    'dos': {
        'code': dos_code,
        'metadata': metadata,
    },
    'projwfc': {
        'code': projwfc_code,
        'metadata': metadata,
    },
}

wg = pdos_workgraph(structure=structure_si, inputs=pdos_inputs, run_scf=True)
wg.name = 'scf_and_pdos'
wg.run()
