from workgraph_collections.ase.common.core_level import (
    get_marked_structures,
)
from workgraph_collections.ase.espresso.xps import run_scf
from ase.build import fcc111
from ase.io import read
import pytest
from pathlib import Path

pt111 = fcc111("Pt", (1, 1, 5), vacuum=10.0)
path = Path(__file__).parent / "../datas/Phenylacetylene.xyz"
phenylacetylene = read(path)


@pytest.mark.parametrize(
    "intpus, data",
    (
        (
            {
                "atoms": pt111,
                "atom_list": [0, 2],
                "is_molecule": False,
                "core_hole_pseudos": {
                    "Pt": {
                        "ground": "O.pbe-n-kjpaw_psl.0.1.UPF",
                        "core_hole": "O.star1s.pbe-n-kjpaw_psl.0.1.UPF",
                        "correction": 676.47 - 8.25,
                    }
                },
            },
            {
                "SYSTEM": {
                    "occupations": "smearing",
                    "tot_charge": 0,
                    "nspin": 2,
                    "starting_magnetization(1)": 0,
                },
                "ntask": 3,
                "pseudo_keys": {"Pt", "X"},
            },
        ),
        (
            {
                "atoms": phenylacetylene,
                "atom_list": [0, 2],
                "is_molecule": True,
                "core_hole_pseudos": {
                    "C": {
                        "ground": "C.pbe-n-kjgipaw_psl.1.0.0.UPF",
                        "core_hole": "C.star1s.pbe-n-kjgipaw_psl.1.0.0.UPF",
                        "correction": 345.99 - 6.2,
                    }
                },
            },
            {
                "SYSTEM": {
                    "assume_isolated": "mt",
                    "tot_charge": 1,
                },
                "ntask": 3,
                "pseudo_keys": {"C", "X"},
            },
        ),
    ),
)
def test_run_scf(intpus, data):
    core_hole_pseudos = intpus.pop("core_hole_pseudos")
    results = get_marked_structures(**intpus)
    input_data = {"SYSTEM": {}}
    wg = run_scf(
        results["structures"],
        input_data=input_data,
        pseudopotentials={},
        core_hole_pseudos=core_hole_pseudos,
        is_molecule=intpus["is_molecule"],
    )

    assert wg.tasks[1].inputs["input_data"].value["SYSTEM"] == data["SYSTEM"]
    assert wg.tasks[1].inputs["pseudopotentials"].value.keys() == data["pseudo_keys"]
    assert len(wg.tasks) == data["ntask"]
