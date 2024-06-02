import os
import warnings
from ase.calculators.genericfileio import GenericFileIOCalculator
from ase.calculators.espresso import EspressoTemplate as AseEspressoTemplate

compatibility_msg = (
    "Espresso calculator is being restructured.  Please use e.g. "
    "Espresso(profile=EspressoProfile(argv=['mpiexec', 'pw.x'])) "
    "to customize command-line arguments."
)


class EspressoTemplate(AseEspressoTemplate):
    def read_results(self, label):
        """Override to set energy to None if not present."""
        results = super().read_results(label)
        if "energy" not in results:
            results["energy"] = None
        return results


class Espresso(GenericFileIOCalculator):
    def __init__(
        self,
        *,
        profile=None,
        command=GenericFileIOCalculator._deprecated,
        label=GenericFileIOCalculator._deprecated,
        directory=".",
        template=EspressoTemplate(),
        **kwargs,
    ):
        """
        All options for pw.x are copied verbatim to the input file, and put
        into the correct section. Use ``input_data`` for parameters that are
        already in a dict.

        input_data: dict
            A flat or nested dictionary with input parameters for pw.x
        pseudopotentials: dict
            A filename for each atomic species, e.g.
            ``{'O': 'O.pbe-rrkjus.UPF', 'H': 'H.pbe-rrkjus.UPF'}``.
            A dummy name will be used if none are given.
        kspacing: float
            Generate a grid of k-points with this as the minimum distance,
            in A^-1 between them in reciprocal space. If set to None, kpts
            will be used instead.
        kpts: (int, int, int), dict, or BandPath
            If kpts is a tuple (or list) of 3 integers, it is interpreted
            as the dimensions of a Monkhorst-Pack grid.
            If ``kpts`` is set to ``None``, only the Γ-point will be included
            and QE will use routines optimized for Γ-point-only calculations.
            Compared to Γ-point-only calculations without this optimization
            (i.e. with ``kpts=(1, 1, 1)``), the memory and CPU requirements
            are typically reduced by half.
            If kpts is a dict, it will either be interpreted as a path
            in the Brillouin zone (*) if it contains the 'path' keyword,
            otherwise it is converted to a Monkhorst-Pack grid (**).
            (*) see ase.dft.kpoints.bandpath
            (**) see ase.calculators.calculator.kpts2sizeandoffsets
        koffset: (int, int, int)
            Offset of kpoints in each direction. Must be 0 (no offset) or
            1 (half grid offset). Setting to True is equivalent to (1, 1, 1).

        """

        if command is not self._deprecated:
            raise RuntimeError(compatibility_msg)

        if label is not self._deprecated:
            warnings.warn("Ignoring label, please use directory instead", FutureWarning)

        if "ASE_ESPRESSO_COMMAND" in os.environ and profile is None:
            warnings.warn(compatibility_msg, FutureWarning)

        super().__init__(
            profile=profile,
            template=template,
            directory=directory,
            parameters=kwargs,
        )
