from ase.io import read
from workgraph_collections.ase.common.core_level import (
    get_non_equivalent_site,
    get_marked_atoms,
)


def test_get_non_equivalent_site():
    mol = read("datas/Phenylacetylene.xyz")
    structures, parameters = get_non_equivalent_site(mol, is_molecule=True)
    assert len(structures) == 7


def test_get_marked_atoms():
    mol = read("datas/Phenylacetylene.xyz")
    structures, parameters = get_marked_atoms(mol, [1, 2, 3])
    assert len(structures) == 4
