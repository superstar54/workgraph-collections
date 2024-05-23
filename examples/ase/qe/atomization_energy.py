from ase.build import molecule
from ase import Atoms
from aiida import load_profile, orm
from workgraph_collections.ase.qe.atomization_energy import atomization_energy

load_profile()

# create input structure node
n_atom = Atoms("N", pbc=True)
n_atom.center(vacuum=5.0)
n2_molecule = molecule("N2", pbc=True)
n2_molecule.center(vacuum=5.0)

metadata = {
    "options": {
        'prepend_text' : """eval "$(conda shell.posix hook)"
        conda activate aiida
        export OMP_NUM_THREADS=1
        """,
    }
}
pseudopotentials = {"N": "N.pbe-n-rrkjus_psl.1.0.0.UPF"}
pseudo_dir = "/home/xing/data/ase/espresso_pseudo"
input_data = {
    "system": {"ecutwfc": 30, "ecutrho": 240,
               "occupations": "smearing",
                "degauss": 0.01,
                "smearing": "cold",},
}
#------------------------- Set the inputs -------------------------
wg = atomization_energy()
wg.nodes["scf_atom"].set({"atoms": n_atom,
                          "pseudopotentials": pseudopotentials,
                          "pseudo_dir": pseudo_dir,
                          "input_data": input_data,
                          "computer": "localhost",
                          "metadata": metadata})
wg.nodes["scf_mol"].set({"atoms": n2_molecule,
                          "pseudopotentials": pseudopotentials,
                          "pseudo_dir": pseudo_dir,
                          "input_data": input_data,
                          "computer": "localhost",
                          "metadata": metadata})
wg.nodes["calc_atomization_energy"].set({"mol": n2_molecule, "computer": "localhost"})
#------------------------- Submit the calculation -------------------
# wg.run()
wg.submit(wait=True, timeout=200)
#------------------------- Print the output -------------------------
print('Energy of a N atom:                  {:0.3f}'.format(wg.nodes['scf_atom'].outputs["result"].value.value['energy']))
print('Energy of an un-relaxed N2 molecule: {:0.3f}'.format(wg.nodes['scf_mol'].outputs["result"].value.value['energy']))
print('Atomization energy:                  {:0.3f} eV'.format(wg.nodes['calc_atomization_energy'].outputs["result"].value.value))

