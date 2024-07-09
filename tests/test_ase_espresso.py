import numpy as np
from workgraph_collections.ase.espresso.atomization_energy import atomization_energy
from workgraph_collections.ase.espresso.base import vibrations
from aiida_workgraph import WorkGraph
import pytest

input_data = {
    "system": {
        "ecutwfc": 30,
        "ecutrho": 240,
        "occupations": "smearing",
        "degauss": 0.01,
        "smearing": "cold",
    },
}


@pytest.mark.skip(reason="The test is too time-consuming.")
def test_vibrations(n2_molecule, pseudo_dir, metadata_aiida):
    pseudopotentials = {"N": "N.pbe-n-rrkjus_psl.1.0.0.UPF"}
    wg = WorkGraph("test_vibrations")
    vibrations_task = wg.tasks.new("PythonJob", name="vibrations", function=vibrations)
    vibrations_task.set(
        {
            "atoms": n2_molecule,
            "indices": [0],
            "pseudopotentials": pseudopotentials,
            "pseudo_dir": pseudo_dir,
            "input_data": input_data,
            "computer": "localhost",
            "metadata": metadata_aiida,
        }
    )
    wg.submit()


def test_atomization_energy(n_atom, n2_molecule, pseudo_dir, metadata_aiida):
    import os

    pseudopotentials = {"N": "N.pbe-n-rrkjus_psl.1.0.0.UPF"}
    # ------------------------- Set the inputs -------------------------
    wg = atomization_energy()
    wg.tasks["scf_atom"].set(
        {
            "atoms": n_atom,
            "pseudopotentials": pseudopotentials,
            "pseudo_dir": pseudo_dir,
            "input_data": input_data,
            "computer": "localhost",
            "metadata": metadata_aiida,
        }
    )
    wg.tasks["scf_mol"].set(
        {
            "atoms": n2_molecule,
            "pseudopotentials": pseudopotentials,
            "pseudo_dir": pseudo_dir,
            "input_data": input_data,
            "computer": "localhost",
            "metadata": metadata_aiida,
        }
    )
    wg.tasks["calc_atomization_energy"].set(
        {"molecule": n2_molecule, "computer": "localhost"}
    )
    wg.submit(wait=True, timeout=200)
    os.system("verdi process report {}".format(wg.tasks["scf_mol"].pk))
    os.system("verdi calcjob remotecat {} CRASH".format(wg.tasks["scf_mol"].pk))

    assert np.isclose(
        wg.tasks["calc_atomization_energy"].outputs["result"].value.value,
        16.24625509874,
    )
