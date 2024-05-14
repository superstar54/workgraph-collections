from aiida import load_profile, orm
from ase.build import bulk
from workgraph_collections.qe.eos import eos_workgraph


load_profile()

structure = orm.StructureData(ase=bulk("Si"))
code = orm.load_code("qe-7.2-pw@localhost")
parameters = {
    "CONTROL": {
        "calculation": "scf",
    },
    "SYSTEM": {
        "ecutwfc": 30,
        "ecutrho": 240,
        "occupations": "smearing",
        "smearing": "gaussian",
        "degauss": 0.1,
    },
}
# Load the pseudopotential family.
pseudo_family = orm.load_group("SSSP/1.3/PBEsol/efficiency")
pseudos = pseudo_family.get_pseudos(structure=structure)
kpoints = orm.KpointsData()
kpoints.set_kpoints_mesh([3, 3, 3])
#
metadata = {
    "options": {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
    }
}

# ===============================================================================
wg = eos_workgraph(
    structure=structure,
    code=code,
    scales=[0.98, 0.99, 1.0, 1.01, 1.02],
    parameters=parameters,
    kpoints=kpoints,
    pseudos=pseudos,
    metadata=metadata,
)
wg.name = "QE-EOS-Si"
# print("correction_energies", correction_energies)
wg.submit()
