from aiida_workgraph import node
from ase import Atoms


@node(outputs=[["General", "atoms"], ["General", "results"]])
def espresso_calculator(
    atoms: Atoms,
    pseudopotentials: dict,
    kpts: list = None,
    binary: str = "pw.x",
    input_data: dict = None,
    pseudo_dir: str = "./pseudopotentials",
):
    """Run a Quantum Espresso calculation on the given atoms object."""
    from ase.calculators.espresso import Espresso, EspressoProfile

    if kpts is None:
        kpts = (1, 1, 1)

    profile = EspressoProfile(
        binary=binary,
        pseudo_dir=pseudo_dir,
    )

    calc = Espresso(
        profile=profile,
        pseudopotentials=pseudopotentials,
        input_data=input_data,
        kpts=kpts,
    )

    atoms.calc = calc

    atoms.get_potential_energy()
    return {"atoms": atoms, "results": atoms.calc.results}
