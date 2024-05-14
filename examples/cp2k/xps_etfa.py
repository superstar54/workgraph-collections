from ase.io import  read
from ase.visualize import view
from aiida.orm import Dict, load_code, load_node, StructureData
from aiida import load_profile
from workgraph_collections.cp2k.xps import xps_workflow

load_profile()


#===================================================================
# cp2k_code = load_code("cp2k-ssmp-2024.1@localhost")
cp2k_code = load_code("cp2k-psmp-2024.1@eiger")
basis_file = load_node(9041)
pseudo_file = load_node(9042)
basis_pseudo_files = {
            "basis": basis_file,
            "pseudo": pseudo_file,
        }
# Structure.
etfa = read("datas/ETFA.xyz")
# view(pt_111_333)
structure = StructureData(ase=etfa)
structure.label = "ETFA molecule, vacuum 5.0 Angstroms"
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
                "POISSON": {
                    "PERIODIC": None,
                    "PSOLVER": "MT",
                },
                # "UKS": True,
                # "MULTIPLICITY": 1,
                "SCF": {
                    "EPS_DIIS": 0.1,
                    "EPS_SCF": 1e-06,
                    "OUTER_SCF": {
                        "MAX_SCF": 20,
                        "EPS_SCF": 1e-06,
                    },
                    "OT": {
                        "ORTHO_IRAC": "CHOL",
                        "N_HISTORY_VEC": 7,
                        "SAFE_DIIS": False,
                        "PRECONDITIONER": "FULL_ALL",
                        "ENERGY_GAP": 0.05,
                        "MINIMIZER": "CG",
                        "ALGORITHM": "IRAC",
                        "EPS_IRAC_SWITCH": 0.01,
                    },
                },
                "QS": {
                    "METHOD": "GPW",
                    "EPS_DEFAULT": 1.0e-12,
                    "EXTRAPOLATION_ORDER": 3,
                },
                "MGRID": {
                    "NGRIDS": 4,
                    "CUTOFF": 500,
                    "REL_CUTOFF": 50,
                },
                "XC": {
                    "XC_FUNCTIONAL": {
                        "PBE": {
                        "PARAMETRIZATION": "PBESOL",
                        }
                    },
                },
            },
            "SUBSYS": {
                "KIND": [
                    {
                        "_": "O",
                        "BASIS_SET": "DZVP-MOLOPT-GTH",
                        "POTENTIAL": "GTH-PBESOL-q6",
                    },
                    {
                        "_": "C",
                        "BASIS_SET": "DZVP-MOLOPT-GTH",
                        "POTENTIAL": "GTH-PBESOL-q4",
                    },
                    {
                        "_": "F",
                        "BASIS_SET": "DZVP-MOLOPT-GTH",
                        "POTENTIAL": "GTH-PBE",
                    },
                    {
                        "_": "H",
                        "BASIS_SET": "DZVP-MOLOPT-GTH",
                        "POTENTIAL": "GTH-PBE",
                    },
                    {
                        "_": "Pt",
                        "ELEMENT": "Pt",
                        "BASIS_SET": "DZVP-MOLOPT-SR-GTH-q18",
                        "POTENTIAL": "GTH-PBESOL-q18",
                    },
                ],
            },
        },
    }
metadata = {"options": {'custom_scheduler_commands' : 'export OMP_NUM_THREADS=4',
                    'resources': {
                                'num_machines' : 1,
                                'num_mpiprocs_per_machine' : 1,
                                }
                    }
        }
metadata_eiger = {"options": {
                  'custom_scheduler_commands' : '#SBATCH --account=mr32',
                  'resources': {
                              'num_machines' : 1,
                              'num_mpiprocs_per_machine' : 128,
                              }
                  }
                  }
#---------------------------------------------------------------
core_hole_pseudos = {
    "C": {
            "_": "X",
            "ELEMENT": "C",
            "BASIS_SET": "DZVP-MOLOPT-GTH",
            "POTENTIAL": "GTH-PBESOL-q5_1s1",
            "CORE_CORRECTION": 1,
        },
    "O": {
            "_": "X",
            "ELEMENT": "O",
            "BASIS_SET": "DZVP-MOLOPT-GTH",
            "POTENTIAL": "GTH-PBESOL-q7_1s1",
            "CORE_CORRECTION": 1,
        },
    "Pt": {
            "_": "X",
            "ELEMENT": "Pt",
            "BASIS_SET": "DZVP-MOLOPT-SR-GTH-q18",
            "POTENTIAL": "GTH-PBESOL-q19_4f13",
            "CORE_CORRECTION": 1,
        }
}
wg = xps_workflow()
wg.name = "ETFA"
wg.nodes["get_marked_structures"].set({
        "structure": structure,
        "atoms_list": [(0, "1s"), (1, "1s"), (2, "1s"), (3, "1s")],
        })
wg.nodes["run_scf"].set({
      "parameters": parameters,
      "code": cp2k_code,
      "basis_pseudo_files": basis_pseudo_files,
      "core_hole_pseudos": core_hole_pseudos,
      "metadata": metadata_eiger,
    #   "metadata": metadata,
      "core_hole_treatment": "full",
    })
wg.nodes["binding_energy"].set({
    "corrections": {
        "C": 397.4,
        "O": 746.4,
        "Pt": 460.6,
        "Au": 296.1,
        }})
wg.submit()
# wg.run()
print("Binding energy of Pt (111) surface: ", wg.nodes["binding_energy"].outputs["result"].value)

