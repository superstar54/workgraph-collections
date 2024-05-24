from aiida_workgraph import node


@node(outputs=[["General", "atoms"], ["General", "results"]])
def pw_calculator(
    atoms,
    pseudopotentials,
    binary="pw.x",
    input_data=None,
    pseudo_dir="./pseudopotentials",
):
    """Run a Quantum Espresso calculation on the given atoms object."""
    from ase.calculators.espresso import Espresso, EspressoProfile

    profile = EspressoProfile(
        binary=binary,
        pseudo_dir=pseudo_dir,
    )

    calc = Espresso(
        profile=profile,
        pseudopotentials=pseudopotentials,
        input_data=input_data,
    )

    atoms.calc = calc

    atoms.get_potential_energy()
    return {"atoms": atoms, "results": atoms.calc.results}
