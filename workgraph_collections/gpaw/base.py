from aiida_workgraph import task, namespace
from ase import Atoms
from typing import Any


@task.pythonjob(outputs=namespace(atoms=Atoms, results=dict))
def gpaw_calculator(
    atoms: Atoms,
    seed: str = "GaAs",
    kpts: dict = None,
    mode: Any = None,
    xc: str = "LDA",
    occupations: Any = None,
    convergence: dict = None,
    txt: str = None,
):
    """Run a GPAW calculation on the given atoms object."""
    from gpaw import GPAW

    if kpts is None:
        kpts = (1, 1, 1)

    calc = GPAW(
        kpts=kpts,
        mode=mode,
        xc=xc,
        occupations=occupations,
        convergence=convergence,
        txt=txt,
    )

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(f"{seed}.gpw", mode="all")
    results = atoms.calc.results
    atoms.calc = None
    return {"atoms": atoms, "results": results}
