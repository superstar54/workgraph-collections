from aiida_workgraph import task, spec
from ase import Atoms


@task
def calc_atomization_energy(
    molecule: Atoms, molecule_output: dict, atom_output: dict
) -> float:
    energy = atom_output["energy"] * len(molecule) - molecule_output["energy"]
    return energy


@task.graph(
    outputs=spec.namespace(
        atomization_energy=float, molecule_parameters=dict, atom_parameters=dict
    )
)
def AtomizationEnergy(
    atom: Atoms = None,
    molecule: Atoms = None,
    pseudopotentials: dict = None,
    input_data: dict = None,
    pseudo_dir: str = None,
    computer: str = "localhost",
    metadata: dict = None,
):
    """Workgraph for atomization energy calculation using Espresso calculator."""
    from .pw import pw_calculator

    pw_atom_out = pw_calculator(
        atoms=atom,
        pseudopotentials=pseudopotentials,
        pseudo_dir=pseudo_dir,
        input_data=input_data,
        computer=computer,
        metadata=metadata,
    ).parameters
    pw_mol_out = pw_calculator(
        atoms=molecule,
        pseudopotentials=pseudopotentials,
        pseudo_dir=pseudo_dir,
        input_data=input_data,
        calculation="relax",
        computer=computer,
        metadata=metadata,
    ).parameters
    atomization_out = calc_atomization_energy(
        molecule=molecule, atom_output=pw_atom_out, molecule_output=pw_mol_out
    ).result
    return {
        "atomization_energy": atomization_out,
        "molecule_parameters": pw_mol_out,
        "atom_parameters": pw_atom_out,
    }
