from workgraph_collections.ase.common.core_level import (
    get_marked_structures,
)
from ase.build import fcc111
from ase.io import read
import pytest
from pathlib import Path
import numpy as np

pt111 = fcc111("Pt", (1, 1, 5), vacuum=10.0)
path = Path(__file__).parent / "../datas/Phenylacetylene.xyz"
phenylacetylene = read(path)


@pytest.mark.parametrize(
    "inputs, data",
    (
        (
            {
                "atoms": pt111,
                "element_list": ["Pt"],
                "min_cell_length": 10.0,
                "is_molecule": False,
            },
            {
                "cell_length": 11.087434329005065,
                "keys": {"original", "supercell", "Pt_0", "Pt_1", "Pt_2"},
                "natom": 80,
                "equivalent_sites": ["Pt_0", {0, 4}],
            },
        ),
        (
            {
                "atoms": phenylacetylene,
                "element_list": ["C"],
                "min_cell_length": 20,
                "is_molecule": True,
            },
            {
                "cell_length": 20,
                "keys": {
                    "original",
                    "supercell",
                    "C_1",
                    "C_3",
                    "C_5",
                    "C_6",
                    "C_7",
                    "C_0",
                },
                "natom": 14,
                "equivalent_sites": ["C_3", {3, 4}],
            },
        ),
    ),
)
def test_get_marked_structures_element_list(inputs, data):
    """Test the function get_marked_structures for a surface."""
    results = get_marked_structures(**inputs)
    assert set(results["structures"].keys()) == data["keys"]
    assert np.isclose(
        results["structures"]["supercell"].get_cell_lengths_and_angles()[0],
        data["cell_length"],
    )
    assert len(results["structures"]["supercell"]) == data["natom"]
    assert (
        results["structures"][data["equivalent_sites"][0]].info["equivalent_sites"]
        == data["equivalent_sites"][1]
    )


@pytest.mark.parametrize(
    "intpus, data",
    (
        (
            {
                "atoms": pt111,
                "atom_list": [0, 2],
                "min_cell_length": 10.0,
                "is_molecule": False,
            },
            {
                "keys": {"original", "supercell", "Pt_0", "Pt_2"},
                "length": 4,
                "equivalent_sites": ["Pt_0", {0}],
            },
        ),
        (
            {
                "atoms": phenylacetylene,
                "atom_list": [0, 2],
                "min_cell_length": 20,
                "is_molecule": True,
            },
            {
                "keys": {"original", "supercell", "C_0", "C_2"},
                "length": 4,
                "equivalent_sites": ["C_0", {0}],
            },
        ),
    ),
)
def test_get_marked_structures_atom_list(intpus, data):
    """Test the function get_marked_structures for a crystal."""
    results = get_marked_structures(**intpus)
    assert set(results["structures"].keys()) == data["keys"]
    assert len(results["structures"]) == data["length"]
    assert (
        results["structures"][data["equivalent_sites"][0]].info["equivalent_sites"]
        == data["equivalent_sites"][1]
    )


def test_get_binding_energy():
    from workgraph_collections.ase.common.core_level import get_binding_energy

    core_hole_pseudos = {
        "C": {
            "ground": "C.pbe-n-kjgipaw_psl.1.0.0.UPF",
            "core_hole": "C.star1s.pbe-n-kjgipaw_psl.1.0.0.UPF",
            "correction": 100,
        }
    }
    scf_outputs = {
        "ground": {"energy": 1},
        "C_0": {"energy": 300},
        "C_1": {"energy": 301},
    }
    results = get_binding_energy(core_hole_pseudos=core_hole_pseudos, **scf_outputs)
    assert results["C_0"] == 399
    assert results["C_1"] == 400
