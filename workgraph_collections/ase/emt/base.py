from aiida_workgraph import task
from ase import Atoms


@task(outputs=[{"identifier": "Any", "name": "results"}])
def emt_calculator(atoms: Atoms) -> float:
    from ase.calculators.emt import EMT

    atoms.calc = EMT()
    atoms.get_potential_energy()
    return atoms.calc.results
