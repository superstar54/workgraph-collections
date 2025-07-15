from aiida_workgraph import task
from ase import Atoms


@task.pythonjob(outputs=[{"name": "parameters"}, {"name": "dos"}])
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
    calc.get_property("dos", Atoms())

    return calc.results


@task.pythonjob(
    outputs=[
        {"name": "parameters"},
        {"name": "dos"},
        {"name": "bands"},
        {"name": "projections"},
    ]
)
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

    calc.get_property("projections", Atoms())
    return calc.results


@task.pythonjob(outputs=[{"name": "results"}])
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


@task.pythonjob(outputs=[{"name": "results"}])
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


@task.pythonjob(outputs=[{"name": "results"}])
def vibrations(
    atoms: Atoms,
    pseudopotentials: dict,
    kpts: list = None,
    kspacing: float = None,
    command: str = "pw.x",
    input_data: dict = None,
    pseudo_dir: str = "./pseudopotentials",
    indices: list = None,
) -> dict:
    """Run a vibrational analysis on the given atoms object."""
    from ase.io.espresso import Namelist
    from ase_quantumespresso.espresso import Espresso, EspressoProfile
    from ase.vibrations import Vibrations

    input_data = {} if input_data is None else input_data

    profile = EspressoProfile(command=command, pseudo_dir=pseudo_dir)

    input_data = Namelist(input_data)
    input_data.to_nested(binary="pw")
    # set the calculation type
    input_data.setdefault("CONTROL", {})
    input_data["CONTROL"]["calculation"] = "scf"
    input_data["CONTROL"]["tprnfor"] = True

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

    vib = Vibrations(atoms, indices=indices)
    vib.run()
    vib.write_jmol()
    vib.write_dos()
    energies = vib.get_energies()
    return {"energies": energies}


@task.pythonjob(
    outputs=[
        {"name": "parameters"},
        {"name": "ld1"},
    ]
)
def ld1_calculator(
    command: str = "ld1.x",
    input_data: dict = None,
    pseudo_potential_test_cards: str = None,
    **kwargs,
) -> dict:
    """Run a ld1 calculation."""

    from ase_quantumespresso.ld1 import Ld1Template
    from ase_quantumespresso.espresso import Espresso, EspressoProfile
    from ase import Atoms

    # Optionally create profile to override paths in ASE configuration:
    profile = EspressoProfile(command=command, pseudo_dir=".")
    input_data["outdir"] = "out"

    calc = Espresso(
        profile=profile,
        template=Ld1Template(),
        input_data=input_data,
        pseudo_potential_test_cards=pseudo_potential_test_cards,
        **kwargs,
    )

    calc.get_property("ld1", Atoms())
    return calc.results
