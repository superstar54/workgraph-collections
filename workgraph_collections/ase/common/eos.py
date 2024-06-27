from aiida_workgraph import task
from ase import Atoms


@task(
    outputs=[
        {"identifier": "General", "name": "scaled_atoms"},
        {"identifier": "General", "name": "volumes"},
    ]
)
def generate_scaled_atoms(atoms: Atoms, scales: list) -> dict:
    """Scale the structure by the given scales."""
    volumes = {}
    scaled_atoms = {}
    for i in range(len(scales)):
        atoms1 = atoms.copy()
        atoms1.set_cell(atoms.cell * scales[i], scale_atoms=True)
        scaled_atoms[f"s_{i}"] = atoms1
        volumes[f"s_{i}"] = atoms1.get_volume()
    return {"scaled_atoms": scaled_atoms, "volumes": volumes}


@task()
def fit_eos(volumes: dict, scf_results: dict) -> dict:
    """Fit the EOS of the data."""
    from ase.eos import EquationOfState
    from ase.units import kJ

    volumes_list = []
    energies = []
    for key, data in scf_results.items():
        energy = data["energy"]
        energies.append(energy)
        volumes_list.append(volumes[key])
    #
    eos = EquationOfState(volumes_list, energies)
    v0, e0, B = eos.fit()
    # convert B to GPa
    B = B / kJ * 1.0e24
    eos = {"energy unit": "eV", "v0": v0, "e0": e0, "B": B}
    return eos
