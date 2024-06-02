from aiida_workgraph import node
from ase import Atoms


@node(outputs=[["General", "atoms"], ["General", "results"]])
def pw_calculator(
    atoms: Atoms,
    pseudopotentials: dict,
    kpts: list = None,
    command: str = "pw.x",
    input_data: dict = None,
    pseudo_dir: str = "./pseudopotentials",
):
    """Run a Quantum Espresso calculation on the given atoms object."""
    from ase.io.espresso import Namelist
    from workgraph_collections.ase.espresso.calculators.espresso import Espresso
    from ase.calculators.espresso import EspressoProfile

    input_data = {} if input_data is None else input_data
    kpts = (1, 1, 1) if kpts is None else kpts

    profile = EspressoProfile(command=command, pseudo_dir=pseudo_dir)

    input_data = Namelist(input_data)
    input_data.to_nested(binary="pw")

    # Set the output directory
    input_data.setdefault("CONTROL", {})
    input_data["CONTROL"]["outdir"] = "out"

    calc = Espresso(
        profile=profile,
        pseudopotentials=pseudopotentials,
        input_data=input_data,
        kpts=kpts,
    )

    atoms.calc = calc

    atoms.get_potential_energy()
    return {"atoms": atoms, "results": atoms.calc.results}


@node()
def dos_calculator(
    command: str = "dos.x",
    input_data: dict = None,
):
    """Run a dos calculation."""
    from workgraph_collections.ase.espresso.calculators.dos import DosTemplate
    from workgraph_collections.ase.espresso.calculators.espresso import Espresso
    from ase.calculators.espresso import EspressoProfile
    from ase import Atoms

    # Optionally create profile to override paths in ASE configuration:
    profile = EspressoProfile(command=command, pseudo_dir=".")
    input_data["outdir"] = "out"

    calc = Espresso(profile=profile, template=DosTemplate(), input_data=input_data)

    dos = calc.get_property("dos", Atoms())
    return dos


@node()
def projwfc_calculator(
    command: str = "projwfc.x",
    input_data: dict = None,
):
    """Run a projwfc calculation."""

    from workgraph_collections.ase.espresso.calculators.projwfc import ProjwfcTemplate
    from workgraph_collections.ase.espresso.calculators.espresso import Espresso
    from ase.calculators.espresso import EspressoProfile
    from ase import Atoms

    # Optionally create profile to override paths in ASE configuration:
    profile = EspressoProfile(command=command, pseudo_dir=".")
    input_data["outdir"] = "out"

    calc = Espresso(profile=profile, template=ProjwfcTemplate(), input_data=input_data)

    pdos = calc.get_property("pdos", Atoms())
    return pdos


@node()
def pp_calculator(
    command: str = "pp.x",
    input_data: dict = None,
):
    """Run a pp calculation."""

    from workgraph_collections.ase.espresso.calculators.pp import PpTemplate
    from workgraph_collections.ase.espresso.calculators.espresso import Espresso
    from ase.calculators.espresso import EspressoProfile
    from ase import Atoms

    # Optionally create profile to override paths in ASE configuration:
    profile = EspressoProfile(command=command, pseudo_dir=".")
    input_data["outdir"] = "out"

    calc = Espresso(profile=profile, template=PpTemplate(), input_data=input_data)

    pp = calc.get_property("pp", Atoms())
    return pp
