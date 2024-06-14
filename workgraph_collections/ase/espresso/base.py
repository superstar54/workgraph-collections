from aiida_workgraph import task
from ase import Atoms


@task(outputs=[["General", "atoms"], ["General", "results"]])
def pw_calculator(
    atoms: Atoms,
    pseudopotentials: dict,
    kpts: list = None,
    kspacing: float = None,
    command: str = "pw.x",
    input_data: dict = None,
    pseudo_dir: str = "./pseudopotentials",
    calculation: str = None,
) -> dict:
    """Run a Quantum Espresso calculation on the given atoms object."""
    from ase.io.espresso import Namelist
    from ase_quantumespresso.espresso import Espresso, EspressoProfile

    input_data = {} if input_data is None else input_data

    profile = EspressoProfile(command=command, pseudo_dir=pseudo_dir)

    input_data = Namelist(input_data)
    input_data.to_nested(binary="pw")
    # set the calculation type
    if calculation:
        input_data.setdefault("CONTROL", {})
        input_data["CONTROL"]["calculation"] = calculation

    # Set the output directory
    input_data.setdefault("CONTROL", {})
    input_data["CONTROL"]["outdir"] = "out"

    calc = Espresso(
        profile=profile,
        pseudopotentials=pseudopotentials,
        input_data=input_data,
        kpts=kpts,
        kspacing=kspacing,
    )

    atoms.calc = calc

    atoms.get_potential_energy()
    results = atoms.calc.results
    new_atoms = results.pop("atoms")
    # we only update the position and cell of the atoms object
    atoms.positions = new_atoms.positions
    atoms.cell = new_atoms.cell
    # Set atoms.calc to None to avoid pickling error
    atoms.calc = None
    return {"atoms": atoms, "results": results}


@task(outputs=[["General", "results"]])
def dos_calculator(
    command: str = "dos.x",
    input_data: dict = None,
) -> dict:
    """Run a dos calculation."""
    from ase_quantumespresso.dos import DosTemplate
    from ase_quantumespresso.espresso import Espresso, EspressoProfile
    from ase import Atoms

    # Optionally create profile to override paths in ASE configuration:
    profile = EspressoProfile(command=command, pseudo_dir=".")
    input_data["outdir"] = "out"

    calc = Espresso(profile=profile, template=DosTemplate(), input_data=input_data)

    results = calc.get_property("dos", Atoms())
    return {"results": results}


@task(outputs=[["General", "results"]])
def projwfc_calculator(
    command: str = "projwfc.x",
    input_data: dict = None,
) -> dict:
    """Run a projwfc calculation."""

    from ase_quantumespresso.projwfc import ProjwfcTemplate
    from ase_quantumespresso.espresso import Espresso, EspressoProfile
    from ase import Atoms

    # Optionally create profile to override paths in ASE configuration:
    profile = EspressoProfile(command=command, pseudo_dir=".")
    input_data["outdir"] = "out"

    calc = Espresso(profile=profile, template=ProjwfcTemplate(), input_data=input_data)

    results = calc.get_property("pdos", Atoms())
    return {"results": results}


@task(outputs=[["General", "results"]])
def pp_calculator(
    command: str = "pp.x",
    input_data: dict = None,
) -> dict:
    """Run a pp calculation."""

    from ase_quantumespresso.pp import PpTemplate
    from ase_quantumespresso.espresso import Espresso, EspressoProfile
    from ase import Atoms

    # Optionally create profile to override paths in ASE configuration:
    profile = EspressoProfile(command=command, pseudo_dir=".")
    input_data["outdir"] = "out"

    calc = Espresso(profile=profile, template=PpTemplate(), input_data=input_data)

    results = calc.get_property("pp", Atoms())
    return {"results": results}


@task(outputs=[["General", "results"]])
def xspectra_calculator(
    command: str = "xspectra.x",
    input_data: dict = None,
    kpts: list = None,
    koffset: list = None,
) -> dict:
    """Run a xspectra calculation."""

    from ase_quantumespresso.xspectra import XspectraTemplate
    from ase_quantumespresso.espresso import Espresso, EspressoProfile
    from ase import Atoms

    # Optionally create profile to override paths in ASE configuration:
    profile = EspressoProfile(command=command, pseudo_dir=".")
    input_data["outdir"] = "out"

    calc = Espresso(
        profile=profile,
        template=XspectraTemplate(),
        input_data=input_data,
        kpts=kpts,
        koffset=koffset,
    )

    results = calc.get_property("xspectra", Atoms())
    return {"results": results}
