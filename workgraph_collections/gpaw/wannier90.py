from aiida_workgraph import task
from .base import gpaw_calculator
from typing import Any, Annotated


@task.pythonjob()
def wannier90(
    seed: str = "GaAs",
    binary: str = "wannier90.x",
    wannier_plot: bool = False,
    bands: list = None,
    orbitals_ai: Any = None,
):
    """Run Wannier90 calculation."""
    import os
    from gpaw.wannier90 import Wannier90
    from gpaw import GPAW

    calc = GPAW("parent_folder/" + seed + ".gpw", txt=None)
    w90 = Wannier90(calc, seed=seed, bands=bands, orbitals_ai=orbitals_ai)

    w90.write_input(num_iter=1000, plot=wannier_plot)
    w90.write_wavefunctions()
    os.system(f"{binary} -pp " + seed)

    w90.write_projections()
    w90.write_eigenvalues()
    w90.write_overlaps()

    os.system(f"{binary} " + seed)


@task.graph()
def Wannier90Workgraph(
    scf_inputs: Annotated[dict, gpaw_calculator.inputs] = None,
    wannier90_inputs: Annotated[dict, wannier90.inputs] = None,
):
    """Workgraph for Wannier90 calculation."""
    scf_out = gpaw_calculator(
        **scf_inputs,
    )
    wannier_out = wannier90(
        parent_folder=scf_out.remote_folder,
        **wannier90_inputs,
    )
    return wannier_out
