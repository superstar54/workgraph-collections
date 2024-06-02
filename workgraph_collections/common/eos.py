from aiida_workgraph import node
from aiida import orm


# explicitly define the output socket name to match the return value of the function
@node.calcfunction(outputs=[["General", "structures"], ["General", "volumes"]])
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


@node.calcfunction()
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
