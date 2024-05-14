from aiida_workgraph import node
from aiida import orm


# explicitly define the output socket name to match the return value of the function
@node.calcfunction(outputs=[["General", "structures"]])
def scale_structure(structure, scales):
    """Scale the structure by the given scales."""
    atoms = structure.get_ase()
    structures = {}
    for i in range(len(scales)):
        atoms1 = atoms.copy()
        atoms1.set_cell(atoms.cell * scales[i], scale_atoms=True)
        structure = orm.StructureData(ase=atoms1)
        structures[f"s_{i}"] = structure
    return {"structures": structures}


@node.calcfunction()
# because this is a calcfunction, and the input datas are dynamic, we need use **datas.
def fit_eos(**datas):
    """Fit the EOS of the data."""
    from ase.eos import EquationOfState

    volumes = []
    energies = []
    for _, data in datas.items():
        volumes.append(data.dict.volume)
        energies.append(data.dict.energy)
        unit = data.dict.energy_units
    #
    eos = EquationOfState(volumes, energies)
    v0, e0, B = eos.fit()
    eos = orm.Dict({"unit": unit, "v0": v0, "e0": e0, "B": B})
    return eos
