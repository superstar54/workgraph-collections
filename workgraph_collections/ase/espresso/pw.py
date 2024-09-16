from aiida_workgraph import task
from ase import Atoms
from ase_quantumespresso.parsers.exit_code import PwExitCodes

import dataclasses


@dataclasses.dataclass
class RestartType:
    FROM_SCRATCH = "FROM_SCRATCH"
    FULL = "FULL"
    FROM_CHARGE_DENSITY = "FROM_CHARGE_DENSITY"
    FROM_WAVE_FUNCTIONS = "FROM_WAVE_FUNCTIONS"


def set_restart_type(restart_type, task):
    """Set the restart type for the next iteration."""
    parent_folder = task.outputs["remote_folder"].value
    input_data = task.inputs["input_data"].value
    input_data.setdefault("CONTROL", {})
    input_data.setdefault("ELECTRONS", {})
    if parent_folder is None and restart_type != RestartType.FROM_SCRATCH:
        raise ValueError(
            "When not restarting from scratch, a `parent_folder` must be provided."
        )
    if restart_type == RestartType.FROM_SCRATCH:
        input_data["CONTROL"]["restart_mode"] = "from_scratch"
        input_data["ELECTRONS"].pop("startingpot", None)
        input_data["ELECTRONS"].pop("startingwfc", None)
        parent_folder = None
    elif restart_type == RestartType.FULL:
        input_data["CONTROL"]["restart_mode"] = "restart"
        input_data["ELECTRONS"].pop("startingpot", None)
        input_data["ELECTRONS"].pop("startingwfc", None)
    elif restart_type == RestartType.FROM_CHARGE_DENSITY:
        input_data["CONTROL"]["restart_mode"] = "from_scratch"
        input_data["ELECTRONS"]["startingpot"] = "file"
        input_data["ELECTRONS"].pop("startingwfc", None)
    elif restart_type == RestartType.FROM_WAVE_FUNCTIONS:
        input_data["CONTROL"]["restart_mode"] = "from_scratch"
        input_data["ELECTRONS"].pop("startingpot", None)
        input_data["ELECTRONS"]["startingwfc"] = "file"
    # we copy the `pwscf.save` folder to the new work directory
    task.set(
        {
            "input_data": input_data,
            "parent_folder": parent_folder,
            "parent_output_folder": "pwscf.save",
            "parent_folder_name": "pwscf.save",
        }
    )


def handle_relax_recoverable_ionic_convergence_error(task):
    """Handle various exit codes for recoverable `relax` calculations with failed ionic convergence.

    These exit codes signify that the ionic convergence thresholds were not met, but the output structure is usable,
    so the solution is to simply restart from scratch but from the output structure.
    """
    set_restart_type(RestartType.FROM_CHARGE_DENSITY, task)
    task.set({"atoms": task.outputs["atoms"].value.value})
    msg = "no ionic convergence but clean shutdown: restarting from scratch but using output structure."
    return msg


def handle_out_of_walltime(task):
    """Handle `ERROR_OUT_OF_WALLTIME` exit code.
    In this case the calculation shut down neatly and we can simply restart. We consider two cases:
    1. If the structure is unchanged, we do a full restart.
    2. If the structure has changed during the calculation, we restart from scratch.
    """
    if task.outputs["atoms"].value is not None:
        set_restart_type(RestartType.FROM_SCRATCH, task)
        msg = "out of walltime: structure changed so restarting from scratch"
    else:
        set_restart_type(RestartType.FULL, task)
        msg = "simply restart from the last calculation"
    return msg


def handle_electronic_convergence_not_reached(task):
    """Handle `ERROR_ELECTRONIC_CONVERGENCE_NOT_REACHED` error.
    Decrease the mixing beta and fully restart from the previous calculation.
    """
    factor = 0.8
    input_data = task.inputs["input_data"].value
    mixing_beta = input_data.get("ELECTRONS", {}).get("mixing_beta", 0.7)
    mixing_beta_new = mixing_beta * factor
    input_data["ELECTRONS"]["mixing_beta"] = mixing_beta_new
    set_restart_type(RestartType.FULL, task)
    msg = f"reduced beta mixing from {mixing_beta} to {mixing_beta_new} and restarting from the last calculation"
    return msg


def handle_relax_recoverable_electronic_convergence_error(task):
    """Handle various exit codes for recoverable `relax` calculations with failed electronic convergence.

    These exit codes signify that the electronic convergence thresholds were not met, but the output structure is
    usable, so the solution is to simply restart from scratch but from the output structure and with a reduced
    ``mixing_beta``.
    """
    factor = 0.8
    input_data = task.inputs["input_data"].value
    mixing_beta = input_data.get("ELECTRONS", {}).get("mixing_beta", 0.7)
    mixing_beta_new = mixing_beta * factor
    input_data.setdefault("ELECTRONS", {})
    input_data["ELECTRONS"]["mixing_beta"] = mixing_beta_new
    task.set({"atoms": task.outputs["atoms"].value.value})
    set_restart_type(RestartType.FROM_SCRATCH, task)
    msg = (
        f"no electronic convergence but clean shutdown: reduced beta mixing from {mixing_beta} to {mixing_beta_new}"
        "restarting from scratch but using output structure."
    )
    return msg


@task.pythonjob(
    outputs=[
        {"name": "atoms"},
        {"name": "parameters"},
        {"name": "kpoints"},
        {"name": "band"},
        {"name": "trajectory"},
        {"name": "atomic_occupations"},
        {"name": "energy"},
    ],
    error_handlers=[
        {
            "handler": handle_relax_recoverable_ionic_convergence_error,
            "exit_codes": [
                PwExitCodes.ERROR_IONIC_CONVERGENCE_NOT_REACHED,
                PwExitCodes.ERROR_IONIC_CYCLE_BFGS_HISTORY_FAILURE,
                PwExitCodes.ERROR_IONIC_CYCLE_EXCEEDED_NSTEP,
                PwExitCodes.ERROR_IONIC_CYCLE_BFGS_HISTORY_AND_FINAL_SCF_FAILURE,
            ],
            "max_retries": 5,
        },
        {
            "handler": handle_out_of_walltime,
            "exit_codes": [PwExitCodes.ERROR_OUT_OF_WALLTIME],
            "max_retries": 5,
        },
        {
            "handler": handle_electronic_convergence_not_reached,
            "exit_codes": [PwExitCodes.ERROR_ELECTRONIC_CONVERGENCE_NOT_REACHED],
            "max_retries": 5,
        },
        {
            "handler": handle_relax_recoverable_electronic_convergence_error,
            "exit_codes": [
                PwExitCodes.ERROR_IONIC_CYCLE_ELECTRONIC_CONVERGENCE_NOT_REACHED,
                PwExitCodes.ERROR_IONIC_CONVERGENCE_REACHED_FINAL_SCF_FAILED,
            ],
            "max_retries": 5,
        },
    ],
)
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
    # input_data["CONTROL"]["outdir"] = "out"

    calc = Espresso(
        profile=profile,
        pseudopotentials=pseudopotentials,
        input_data=input_data,
        kpts=kpts,
        kspacing=kspacing,
    )

    atoms.calc = calc

    atoms.get_potential_energy()
    return atoms.calc.results
