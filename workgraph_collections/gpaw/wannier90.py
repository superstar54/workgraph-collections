from aiida_workgraph import node, WorkGraph
from .base import gpaw_calculator
from ase import Atoms


@node()
def wannier90(
    seed: str = "GaAs",
    binary: str = "wannier90.x",
    wannier_plot: bool = False,
    **kwargs: dict,
):
    """Run Wannier90 calculation."""
    import os
    from gpaw.wannier90 import Wannier90
    from gpaw import GPAW

    calc = GPAW("parent_folder/" + seed + ".gpw", txt=None)
    w90 = Wannier90(calc, seed=seed, **kwargs)

    w90.write_input(num_iter=1000, plot=wannier_plot)
    w90.write_wavefunctions()
    os.system(f"{binary} -pp " + seed)

    w90.write_projections()
    w90.write_eigenvalues()
    w90.write_overlaps()

    os.system(f"{binary} " + seed)


@node.graph_builder()
def wannier90_workgraph(
    atoms: Atoms = None,
):
    """Workgraph for Wannier90 calculation."""
    wg = WorkGraph()
    scf_node = wg.nodes.new(
        gpaw_calculator,
        name="scf",
        atoms=atoms,
        run_remotely=True,
    )
    wg.nodes.new(
        wannier90,
        name="wannier90",
        parent_folder=scf_node.outputs["remote_folder"],
        run_remotely=True,
    )
    return wg
