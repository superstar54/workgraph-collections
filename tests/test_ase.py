from workgraph_collections.ase.common.core_level import (
    get_non_equivalent_site,
    get_marked_atoms,
)


def test_get_non_equivalent_site(phenylacetylene):
    structures, parameters = get_non_equivalent_site(phenylacetylene, is_molecule=True)
    assert len(structures) == 7


def test_get_marked_atoms(phenylacetylene):
    structures, parameters = get_marked_atoms(phenylacetylene, [1, 2, 3])
    assert len(structures) == 4
