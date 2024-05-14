from aiida import load_profile, orm
from ase.build import bulk
from workgraph_collections.cp2k.eos import eos_workgraph


load_profile()

atoms = bulk("Si")
structure = orm.StructureData(ase=atoms)
code = orm.load_code("cp2k-ssmp-2024.1@localhost")
# Parameters.
parameters = {
    "GLOBAL": {
        "RUN_TYPE": "ENERGY_FORCE",
    },
    "FORCE_EVAL": {
        "METHOD": "Quickstep",
        "DFT": {
            "BASIS_SET_FILE_NAME": "BASIS_MOLOPT",
            "POTENTIAL_FILE_NAME": "POTENTIALS",
            "SCF": {
                "ADDED_MOS": 10,
                "SMEAR": {
                    "METHOD": "FERMI_DIRAC",
                    "ELECTRONIC_TEMPERATURE": 500,
                },
            },
            "KPOINTS": {
                "SCHEME": "MONKHORST-PACK 5 5 5",
            },
            "QS": {
                "EPS_DEFAULT": 1.0e-12,
                "METHOD": "GPW",
            },
            "MGRID": {
                "NGRIDS": 4,
                "CUTOFF": 500,
                "REL_CUTOFF": 50,
            },
            "XC": {
                "XC_FUNCTIONAL": {
                    "_": "PBE",
                },
            },
        },
        "SUBSYS": {
            "KIND": [
                {
                    "_": "Si",
                    "BASIS_SET": "DZVP-MOLOPT-GTH",
                    "POTENTIAL": "GTH-PBE",
                },
            ],
        },
    },
}
metadata = {
    "options": {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
    }
}
basis_pseudo_files = {
    "basis": orm.load_node(9041),
    "pseudo": orm.load_node(9042),
}
# ===============================================================================
wg = eos_workgraph(
    structure=structure,
    code=code,
    scales=[0.98, 0.99, 1.0, 1.01, 1.02],
    parameters=parameters,
    basis_pseudo_files=basis_pseudo_files,
    metadata=metadata,
)
wg.name = "CP2K-EOS-Si"
# print("correction_energies", correction_energies)
wg.submit()
