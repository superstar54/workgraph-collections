from aiida_workgraph import task


@task()
def bader_calculator(
    command: str = "pw.x",
    charge_density_folder: str = "./",
    charge_density_filename: str = "charge_density.cube",
    reference_charge_density_folder: str = "./",
    reference_charge_density_filename: str = "charge_density.cube",
):
    """Run Bader charge analysis."""
    import os

    command_str = f"{command} {charge_density_folder}/{charge_density_filename}"
    if reference_charge_density_filename:
        command_str += f" -ref {reference_charge_density_folder}/{reference_charge_density_filename}"
    os.system(command_str)

    with open("ACF.dat", "r") as f:
        lines = f.readlines()
        charges = [float(line.split()[4]) for line in lines[2:-4]]

    return charges
