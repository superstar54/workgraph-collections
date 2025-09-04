from aiida_workgraph import task


@task()
def calc_correction(ground_output, core_hole_output):
    energy = (core_hole_output["Etot"]["eV"] - core_hole_output["Etotps"]["eV"]) - (
        ground_output["Etot"]["eV"] - ground_output["Etotps"]["eV"]
    )
    return energy


@task.graph()
def core_hole_pseudo_workgraph(
    ground_inputs: dict = None,
    core_hole_inputs: dict = None,
) -> float:
    """Workgraph for atomization energy calculation using Espresso calculator."""

    from .base import ld1_calculator

    ground_inputs = {} if ground_inputs is None else ground_inputs
    core_hole_inputs = {} if core_hole_inputs is None else core_hole_inputs

    ground_out = ld1_calculator(**ground_inputs)
    core_hole_out = ld1_calculator(**core_hole_inputs)
    return calc_correction(
        core_hole_output=core_hole_out.ld1,
        ground_output=ground_out.ld1,
    ).result
