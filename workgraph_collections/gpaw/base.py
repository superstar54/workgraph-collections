from aiida_workgraph import node
from ase import Atoms


@node(outputs=[["General", "atoms"], ["General", "results"]])
def gpaw_calculator(
    atoms: Atoms, seed: str = "GaAs", kpts: dict = None, **kwargs: dict
):
    """Run a GPAW calculation on the given atoms object."""
    from gpaw import GPAW

    if kpts is None:
        kpts = (1, 1, 1)

    calc = GPAW(kpts=kpts, **kwargs)

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(f"{seed}.gpw", mode="all")
    results = atoms.calc.results
    atoms.calc = None
    return {"atoms": atoms, "results": results}
