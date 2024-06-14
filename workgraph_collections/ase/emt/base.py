from aiida_workgraph import task
from ase import Atoms


@task(outputs=[["General", "results"]])
def emt_calculator(atoms: Atoms) -> float:
    from ase.calculators.emt import EMT

    atoms.calc = EMT()
    atoms.get_potential_energy()
    return {"results": atoms.calc.results}
