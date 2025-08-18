from aiida import orm
from aiida_workgraph import task, spec
from typing import Annotated, Any


# explicitly define the output socket name to match the return value of the function
@task.calcfunction(
    outputs=spec.namespace(structures=spec.dynamic(orm.StructureData), volumes=dict)
)
def scale_structure(structure: orm.StructureData, scales: list) -> dict:
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
def fit_eos(
    volumes: dict = None, **scf_outputs: Annotated[dict, spec.dynamic(dict)]
) -> orm.Dict:
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
@task.graph(outputs=spec.namespace(result=spec.dynamic(Any)))
def all_scf(
    calculator,
    structures: Annotated[dict, spec.dynamic(Any)],
    scf_inputs: Annotated[dict, spec.dynamic(Any)],
) -> dict:
    """Run the scf calculation for each structure."""

    result = {}
    for key, structure in structures.items():
        scf_out = calculator(structure=structure, **scf_inputs)
        # save the output parameters to the context
        result[key] = scf_out.output_parameters
    return result


@task.graph
def eos_workgraph(
    structure: orm.StructureData = None,
    scales: list = None,
    scf_inputs: Annotated[dict, spec.dynamic(Any)] = None,
    calculator: callable = None,
):
    """Workgraph for EOS calculation.
    1. Get the scaled structures.
    2. Run the SCF calculation for each scaled structure.
    3. Fit the EOS.
    """
    scale_structure_out = scale_structure(structure=structure, scales=scales)
    all_scf_out = all_scf(
        calculator=calculator,
        structures=scale_structure_out.structures,
        scf_inputs=scf_inputs,
    )
    return fit_eos(
        volumes=scale_structure_out.volumes,
        scf_outputs=all_scf_out.result,
    ).result
