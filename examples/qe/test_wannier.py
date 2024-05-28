from aiida import load_profile, orm
from ase.build import bulk
from copy import deepcopy
from aiida.plugins import DataFactory

load_profile()

StructureData = DataFactory("core.structure")
a = 5.68018817933178
structure = StructureData(
    cell=[[-a / 2.0, 0, a / 2.0], [0, a / 2.0, a / 2.0], [-a / 2.0, a / 2.0, 0]]
)
structure.append_atom(symbols=["Ga"], position=(0.0, 0.0, 0.0))
structure.append_atom(symbols=["As"], position=(-a / 4.0, a / 4.0, a / 4.0))
structure.store()


structure = orm.StructureData(ase=bulk("Si"))
pw_code = orm.load_code("qe-7.2-pw@localhost")
projwfc_code = orm.load_code("qe-7.2-projwfc@localhost")
pw2wannier90_code = orm.load_code("qe-7.2-pw2wannier90@localhost")
wannier90_code = orm.load_code("wannier90@localhost")

scf_paras = {
    "CONTROL": {
        "calculation": "scf",
    },
    "SYSTEM": {
        "ecutwfc": 30,
        "ecutrho": 240,
        "occupations": "smearing",
        "smearing": "cold",
    },
}
nscf_paras = deepcopy(scf_paras)
nscf_paras.get_dict()["CONTROL"]["calculation"] = "nscf"

# Load the pseudopotential family.
pseudo_family = orm.load_group("SSSP/1.3/PBEsol/efficiency")
pseudos = pseudo_family.get_pseudos(structure=structure)
kpoints = orm.KpointsData()
kpoints.set_kpoints_mesh([2, 2, 2])
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
wg = wannier_workgraph()
wg.name = "Wannier-Si"

wannier_inputs = {
    "scf": {
        "pw": {
            "code": pw_code,
            "pseudos": pseudos,
            "parameters": scf_paras,
            "metadata": metadata,
        },
        "kpoints": kpoints,
    },
    "nscf": {
        "pw": {
            "code": pw_code,
            "pseudos": pseudos,
            "parameters": nscf_paras,
            "metadata": metadata,
        },
        "kpoints": kpoints,
    },
    "projwfc": {
        "code": projwfc_code,
        "metadata": metadata,
    },
    "wannier90_pp": {
        "wannier90": {
            "code": wannier90_code,
            "parameters": nscf_paras,
            "metadata": metadata,
            "settings": {"postproc_setup": True},
        },
        "kpoints": kpoints,
    },
    "pw2wannier90": {
        "code": pw2wannier90_code,
        "parameters": {
            "scdm_entanglement": "erfc",
            "scdm_proj": True,
        },
        "metadata": metadata,
    },
    "wannier90": {
        "wannier90": {
            "code": wannier90_code,
            "parameters": {
                "auto_projections": True,
                "band_plot": True,
            },
            "metadata": metadata,
            "settings": {"postproc_setup": False},
        },
        "kpoints": kpoints,
    },
}
wg.submit()
