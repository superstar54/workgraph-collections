import numpy as np
from workgraph_collections.ase.espresso.atomization_energy import atomization_energy
from workgraph_collections.ase.espresso.base import vibrations
from aiida_workgraph import WorkGraph
import pytest

input_data = {
    "CONTROL": {
        "calculation": "scf",
        "verbosity": "high",
    },
    "SYSTEM": {
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
    vibrations_task = wg.add_task("PythonJob", name="vibrations", function=vibrations)
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

    pseudopotentials = {"N": "N.pbe-n-rrkjus_psl.1.0.0.UPF"}
    # ------------------------- Set the inputs -------------------------
    wg = atomization_energy()
    wg.tasks.scf_atom.set(
        {
            "atoms": n_atom,
            "pseudopotentials": pseudopotentials,
            "pseudo_dir": pseudo_dir,
            "input_data": input_data,
            "computer": "localhost",
            "metadata": metadata_aiida,
        }
    )
    wg.tasks.scf_mol.set(
        {
            "atoms": n2_molecule,
            "pseudopotentials": pseudopotentials,
            "pseudo_dir": pseudo_dir,
            "input_data": input_data,
            "computer": "localhost",
            "metadata": metadata_aiida,
        }
    )
    wg.tasks.calc_atomization_energy.set(
        {"molecule": n2_molecule, "computer": "localhost"}
    )
    wg.run()

    assert np.isclose(
        wg.tasks.calc_atomization_energy.outputs.result.value.value,
        16.24625509874,
    )


def test_eos(bulk_si, pseudo_dir, metadata_aiida):
    from workgraph_collections.ase.espresso.eos import eos_workgraph

    pseudopotentials = {"Si": "Si.pbe-nl-rrkjus_psl.1.0.0.UPF"}
    # ------------------------- Set the inputs -------------------------
    wg = eos_workgraph(
        atoms=bulk_si,
        computer="localhost",
        scales=[0.95, 1.0, 1.05],
        command="mpirun -np 2 pw.x",
        pseudopotentials=pseudopotentials,
        pseudo_dir=pseudo_dir,
        input_data=input_data,
        kpts=(4, 4, 4),
        metadata=metadata_aiida,
    )
    # ------------------------- Submit the calculation -------------------
    # wg.run()
    wg.run()

    assert np.isclose(
        wg.tasks["fit_eos"].outputs.result.value.get_dict()["B"],
        88.8909406,
        atol=1e-1,
    )
