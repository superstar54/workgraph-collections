from aiida_workgraph import task, WorkGraph


@task()
def calc_correction(ground_output, core_hole_output):
    energy = (core_hole_output["Etot"]["eV"] - core_hole_output["Etotps"]["eV"]) - (
        ground_output["Etot"]["eV"] - ground_output["Etotps"]["eV"]
    )
    return energy


@task.graph_builder(outputs=[{"name": "result", "from": "calc_correction.result"}])
def core_hole_pseudo_workgraph(
    ground_inputs: dict = None,
    core_hole_inputs: dict = None,
) -> WorkGraph:
    """Workgraph for atomization energy calculation using Espresso calculator."""

    from .base import ld1_calculator

    ground_inputs = {} if ground_inputs is None else ground_inputs
    core_hole_inputs = {} if core_hole_inputs is None else core_hole_inputs

    wg = WorkGraph("Core-hole pseudo workgraph")
    ground_task = wg.tasks.new(
        "PythonJob",
        function=ld1_calculator,
        name="ground",
    )
    ground_task.set(ground_inputs)
    core_hole_task = wg.tasks.new(
        "PythonJob", function=ld1_calculator, name="core_hole"
    )
    core_hole_task.set(core_hole_inputs)
    # create the task to calculate the atomization energy
    wg.tasks.new(
        "PythonJob",
        function=calc_correction,
        name="calc_correction",
        core_hole_output=core_hole_task.outputs["results"],
        ground_output=ground_task.outputs["results"],
    )
    return wg
