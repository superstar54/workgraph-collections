from ase.build import molecule
from ase import Atoms
from workgraph_collections.ase.emt.atomization_energy import atomization_energy
from aiida import load_profile

load_profile()

# create input structure node
n_atom = Atoms("N", pbc=True)
n_atom.center(vacuum=5.0)
n2_molecule = molecule("N2", pbc=True)
n2_molecule.center(vacuum=5.0)

metadata = {
    "options": {
        "prepend_text": """
eval "$(conda shell.posix hook)"
conda activate aiida
export OMP_NUM_THREADS=1
        """,
    }
}
# ------------------------- Set the inputs -------------------------
wg = atomization_energy()
wg.nodes["scf_atom"].set(
    {"atoms": n_atom, "computer": "localhost", "metadata": metadata}
)
wg.nodes["scf_mol"].set(
    {"atoms": n2_molecule, "computer": "localhost", "metadata": metadata}
)
wg.nodes["calc_atomization_energy"].set({"mol": n2_molecule, "computer": "localhost"})
# ------------------------- Submit the calculation -------------------
# wg.run()
wg.submit(wait=True, timeout=200)
# ------------------------- Print the output -------------------------
print(
    "Atomization energy:                  {:0.3f} eV".format(
        wg.nodes["calc_atomization_energy"].outputs["result"].value.value
    )
)
