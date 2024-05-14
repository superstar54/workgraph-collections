# -*- coding: utf-8 -*-
"""Test."""
from ase.io import read
from aiida import load_profile
from aiida.orm import Dict, KpointsData, StructureData, load_code, load_group, QueryBuilder, Group
from workgraph_collections.qe.xps import xps_workgraph

load_profile()

# ===============================================================================
def load_core_hole_pseudos(pseudo_group="pseudo_demo_pbe"):
    pseudo_group = (
        QueryBuilder().append(Group, filters={"label": pseudo_group}).one()[0]
    )
    core_hole_pseudos = {node.label: node for node in pseudo_group.nodes}
    return core_hole_pseudos, pseudo_group.base.extras.get("correction", {})


# create input structure node
etfa = read("datas/ETFA.xyz")
structure = StructureData(ase=etfa)
# create the PW node
# code = load_code("qe-7.2-pw@localhost")
code = load_code("qe-7.2-pw@eiger")
parameters = Dict(
    {
        "CONTROL": {
            "calculation": "scf",
        },
        "SYSTEM": {
            "ecutwfc": 50,
            "ecutrho": 600,
            "occupations": "fixed",
        },
    }
)
kpoints = KpointsData()
kpoints.set_kpoints_mesh([1, 1, 1])
# Load the pseudopotential family.
core_hole_pseudos, correction_energies = load_core_hole_pseudos("pseudo_demo_pbe")
pseudo_family = load_group("SSSP/1.3/PBE/efficiency")
pseudos = pseudo_family.get_pseudos(structure=structure)
# print("core_hole_pseudos", core_hole_pseudos)
pseudos["C"] = core_hole_pseudos["C_gs"]
#
metadata = {
    "options": {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
    }
}
metadata_eiger = {"options": {
                  'custom_scheduler_commands' : '#SBATCH --account=mr32',
                  'resources': {
                              'num_machines' : 1,
                              'num_mpiprocs_per_machine' : 36,
                              }
                  }
                  }
# ===============================================================================
wg = xps_workgraph()
wg.name = "ETFA"
wg.nodes["get_marked_structures"].set({
        "structure": structure,
        "atoms_list": [(0, "1s"), (1, "1s"), (2, "1s"), (3, "1s")],
        })
wg.nodes["run_scf"].set({
        "code": code,
        "parameters": parameters,
        "kpoints": kpoints,
        "pseudos": pseudos,
        "metadata": metadata_eiger,
        # "metadata": metadata,
        "is_molecule": True,
        "core_hole_pseudos": core_hole_pseudos,
        })
correction_energies={key.split("_")[0]: value['core'] for key, value in correction_energies.items()}
wg.nodes["binding_energy"].set({
    "corrections": correction_energies,
    })

# print("correction_energies", correction_energies)
wg.submit()
