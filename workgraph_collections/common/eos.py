from aiida import orm
from aiida_workgraph import task, WorkGraph


# explicitly define the output socket name to match the return value of the function
@task.calcfunction(outputs=[{"name": "structures"}, {"name": "volumes"}])
def scale_structure(structure: orm.StructureData, scales: list):
    """Scale the structure by the given scales."""
    atoms = structure.get_ase()
    volumes = {}
    structures = {}
    for i in range(len(scales)):
        atoms1 = atoms.copy()
        atoms1.set_cell(atoms.cell * scales[i], scale_atoms=True)
        structure = orm.StructureData(ase=atoms1)
        structures[f"s_{i}"] = structure
        volumes[f"s_{i}"] = structure.get_cell_volume()
    return {"structures": structures, "volumes": orm.Dict(volumes)}


@task.calcfunction()
# because this is a calcfunction, and the input scf_outputs are dynamic, we need use **scf_outputs.
def fit_eos(volumes: dict = None, **scf_outputs):
    """Fit the EOS of the data."""
    from ase.eos import EquationOfState
    from ase.units import kJ

    volumes_list = []
    energies = []
    for key, data in scf_outputs.items():
        unit = data.dict.energy_units
        energy = data.dict.energy
        if unit == "a.u.":  # convert to eV
            energy = energy * 27.21138602
        energies.append(energy)
        volumes_list.append(volumes.get_dict()[key])
    #
    eos = EquationOfState(volumes_list, energies)
    v0, e0, B = eos.fit()
    # convert B to GPa
    B = B / kJ * 1.0e24
    eos = orm.Dict({"energy unit": "eV", "v0": v0, "e0": e0, "B": B})
    return eos


# Output result from context to the output socket
@task.graph(outputs=[{"name": "result", "from": "context.result"}])
def all_scf(calculator, structures, scf_inputs):
    """Run the scf calculation for each structure."""
    from aiida_workgraph import WorkGraph

    wg = WorkGraph()
    for key, structure in structures.items():
        scf = wg.add_task(calculator, name=f"scf_{key}", structure=structure)
        scf.set(scf_inputs)
        # save the output parameters to the context
        scf.set_context({f"result.{key}": "output_parameters"})
    return wg


@task.graph(outputs=[{"name": "result", "from": "fit_eos.result"}])
def eos_workgraph(
    structure: orm.StructureData = None,
    scales: list = None,
    scf_inputs: dict = None,
    calculator: callable = None,
):
    """Workgraph for EOS calculation.
    1. Get the scaled structures.
    2. Run the SCF calculation for each scaled structure.
    3. Fit the EOS.
    """
    wg = WorkGraph("EOS")
    scale_structure1 = wg.add_task(
        scale_structure, name="scale_structure", structure=structure, scales=scales
    )
    all_scf1 = wg.add_task(
        all_scf,
        name="all_scf",
        calculator=calculator,
        structures=scale_structure1.outputs.structures,
        scf_inputs=scf_inputs,
    )
    wg.add_task(
        fit_eos,
        name="fit_eos",
        volumes=scale_structure1.outputs["volumes"],
        scf_outputs=all_scf1.outputs.result,
    )
    return wg
