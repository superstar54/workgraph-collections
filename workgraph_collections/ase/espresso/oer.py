import numpy as np
from aiida_workgraph import WorkGraph, task

molecule_energies = {
    "H2O": 0,
    "H2": 0,
}


@task(
    outputs=[
        {
            "name": "structures",
            "identifier": "workgraph.namespace",
            "metadata": {"dynamic": True},
        }
    ]
)
def build_adsorbate(atoms, site, site_symbol, site_position, mols):
    """
    Add O*, OH* and OOH*, and relax the structure
    For 'ontop' site, the 'O' and 'H' vacancy pathway are possible.
    """
    structures = {}
    label = label.replace("/", "_")
    # print(site_position)
    job = "%s_%s" % (label, "Clean")
    structures[job] = atoms.copy()
    if site_symbol not in ["H", "O"]:
        for prefix, mol in mols.items():
            # print(prefix, mol)
            job = "%s_%s" % (label, prefix)
            ads = mol.copy()
            natoms = atoms.copy()
            ads.translate(site_position - ads[0].position)
            natoms = natoms + ads
            structures[job] = natoms
    # O
    ind = site
    if site_symbol == "O":
        job = "%s_%s" % (label, "O")
        structures[job] = atoms.copy()
        atoms = atoms.copy()
        del atoms[[ind]]
        job = "%s_%s" % (label, "Clean")
        structures[job] = atoms.copy()
        for prefix, mol in mols.items():
            # print(prefix, mol)
            if prefix == "O":
                continue
            job = "%s_%s" % (label, prefix)
            ads = mol.copy()
            natoms = atoms.copy()
            ads.translate(atoms[ind].position - ads[0].position)
            natoms = natoms + ads
            structures[job] = natoms
    elif site_symbol == "H":
        atoms = atoms.copy()
        dis = atoms.get_distance(ind, range(len(atoms)))
        indo = dis.index(min(dis))
        # O
        job = "%s_%s" % (label, "O")
        natoms = atoms.copy()
        del natoms[[ind]]
        structures[job] = natoms
        # O2
        job = "%s_%s" % (label, "Clean")
        natoms = atoms.copy()
        del natoms[[ind, indo]]
        structures[job] = natoms
        # OOH
        job = "%s_%s" % (label, "OOH")
        ads = mols[job].copy()
        natoms = atoms.copy()
        ads.translate(atoms[indo].position - ads[0].position)
        natoms = natoms + ads
        structures[job] = natoms
    print("Total number of OER adsorbate: {0}\n".format(len(structures)))
    return structures


@task.graph(outputs=[{"name": "scf_results", "from": "context.results"}])
def relax_structures(structures, pw_inputs):
    """Run the scf calculation for each atoms."""
    from workgraph_collections.ase.espresso.base import pw_calculator

    wg = WorkGraph()
    for key, atoms in structures.items():
        scf = wg.add_task(
            "workgraph.pythonjob", function=pw_calculator, name=f"pw_{key}", atoms=atoms
        )
        scf.set(pw_inputs)
        # save the output parameters to the context
        scf.set_context({f"results.{key}": "parameters"})
    return wg


def get_free_energy(results=None, G_O=None, G_OH=None, G_OOH=None, E_H=None, zpes=None):
    """ """
    G_h2o = molecule_energies["H2O"]
    G_h2 = molecule_energies["H2"]
    if not G_O:
        G_O = G_h2o - G_h2
    if not G_OH:
        G_OH = G_h2o - G_h2 / 2.0
    if not G_OOH:
        G_OOH = 2 * G_h2o - 3 * G_h2 / 2.0
    if not E_H:
        E_H = G_h2 / 2.0
    # fig, ax = plt.subplots()
    label = label.replace("/", "_")
    E0 = results["%s_%s" % (label, "Clean")]["energy"]
    # if atoms[site].symbol not in []:
    dG_OH = (
        results["%s_%s" % (label, "OH")]["energy"]
        - E0
        - G_OH
        + zpes["%s_%s" % (label, "OH")]
    )
    dG_O = (
        results["%s_%s" % (label, "O")]["energy"]
        - E0
        - G_O
        + zpes["%s_%s" % (label, "O")]
    )
    dG_OOH = (
        results["%s_%s" % (label, "OOH")]["energy"]
        - E0
        - G_OOH
        + zpes["%s_%s" % (label, "OOH")]
    )
    dG_O2 = 4.92
    steps = ["OH", "O", "OOH", "O2"]
    free_energies = [dG_OH, dG_O, dG_OOH, dG_O2]
    over_potential = (
        max(
            [
                free_energies[1] - free_energies[0],
                free_energies[2] - free_energies[1],
                free_energies[3] - free_energies[2],
            ]
        )
        - 1.23
    )
    return steps, over_potential


def oer_site_workgraph(
    atoms,
    label="",
    prefix=None,
    site_type="ontop",
    site=None,
    height=2.0,
    calculator=None,
    molecule_energies=None,
    mols=None,
):
    """Workflow:
    0. Read the surface slab
    1. Build the adsorption site
    2. Add O*, OH* and OOH*, and relax the structure
    3. Calculate the dos and pdos
    4. Calculate the ZPE energy using Vibrations module
    5. Calculate the Gibbs free energy
    6. Generate the report for OER overpotential
    """
    if site_type.upper() == "ONTOP":
        site_symbol = atoms[site].symbol
        site_position = atoms[site].position + np.array([0, 0, height])
    elif site_type.upper() == "BRIDGE":
        site_position = (
            atoms[site[0]].position + atoms[site[1]].position
        ) / 2.0 + np.array([0, 0, height])
    elif site_type.upper() == "HOLLOW":
        site_position = (
            atoms[site[0]].position + atoms[site[1]].position + atoms[site[2]].position
        ) / 2.0 + np.array([0, 0, height])
    elif site_type.upper() == "POSITION":
        site_position = site + np.array([0, 0, height])

    wg = WorkGraph("OER")
    build_adsorbate_task = wg.add_task(
        "workgraph.pythonjob",
        function=build_adsorbate,
        name="build_adsorbate",
        atoms=atoms,
        site=site,
        site_symbol=site_symbol,
        site_position=site_position,
        mols=mols,
    )
    wg.task.new(
        relax_structures,
        name=relax_structures,
        structures=build_adsorbate_task.outputs.structures,
    )
