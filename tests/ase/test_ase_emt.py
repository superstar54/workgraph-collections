import numpy as np
from workgraph_collections.ase.emt.atomization_energy import AtomizationEnergy


def test_atomization_energy(n_atom, n2_molecule, metadata_aiida):
    wg = AtomizationEnergy.build(
        atom=n_atom, molecule=n2_molecule, metadata=metadata_aiida
    )
    # ------------------------- Submit the calculation -------------------
    wg.run()
    assert np.isclose(wg.outputs.result.value.value, 9.6512352)
