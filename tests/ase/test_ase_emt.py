import numpy as np
from workgraph_collections.ase.emt.atomization_energy import atomization_energy


def test_atomization_energy(n_atom, n2_molecule, metadata_aiida):
    wg = atomization_energy()
    wg.tasks.scf_atom.set(
        {"atoms": n_atom, "computer": "localhost", "metadata": metadata_aiida}
    )
    wg.tasks.scf_mol.set(
        {"atoms": n2_molecule, "computer": "localhost", "metadata": metadata_aiida}
    )
    wg.tasks.calc_atomization_energy.set(
        {"molecule": n2_molecule, "computer": "localhost"}
    )
    # ------------------------- Submit the calculation -------------------
    wg.submit(wait=True, timeout=200)
    assert np.isclose(
        wg.tasks.calc_atomization_energy.outputs.result.value.value, 9.6512352
    )
