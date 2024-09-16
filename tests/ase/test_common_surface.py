from workgraph_collections.ase.common.surface import get_slab_from_miller_indices_ase
from ase.build import bulk, surface
import pytest

pt_bulk = bulk("Pt", cubic=True)
pt111 = surface(pt_bulk, (1, 1, 1), 3, 5.0, 1e-5, True)


@pytest.mark.parametrize(
    "intpus, data",
    (
        (
            {
                "atoms": pt_bulk,
                "indices": (1, 1, 1),
                "layers": 3,
                "vacuum": 5.0,
            },
            pt111,
        ),
    ),
)
def test_get_slab_from_miller_indices_ase(intpus, data):
    slab = get_slab_from_miller_indices_ase(**intpus)
    assert slab == data
