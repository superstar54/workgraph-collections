from aiida_workgraph import task
from ase import Atoms


@task.pythonjob
def emt_calculator(atoms: Atoms) -> dict:
    from ase.calculators.emt import EMT

    atoms.calc = EMT()
    atoms.get_potential_energy()
    return atoms.calc.results
