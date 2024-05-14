from aiida.orm import Kind, Site, StructureData, Dict
from aiida_workgraph import node


@node.calcfunction(outputs=[["General", "structures"]])
def get_marked_structures(
    structure: StructureData = None, atoms_list: list = None, marker: str = "X"
):
    """Get the marked structures for each atom."""
    structures = {"ground": structure.clone()}
    for data in atoms_list.get_list():
        index, orbital = data
        marked_structure = StructureData()
        kinds = {kind.name: kind for kind in structure.kinds}
        marked_structure.set_cell(structure.cell)

        for i, site in enumerate(structure.sites):
            if i == index:
                marked_kind = Kind(name=marker.value, symbols=site.kind_name)
                marked_site = Site(kind_name=marked_kind.name, position=site.position)
                marked_structure.append_kind(marked_kind)
                marked_structure.append_site(marked_site)
                symbol = site.kind_name
            else:
                if site.kind_name not in [kind.name for kind in marked_structure.kinds]:
                    marked_structure.append_kind(kinds[site.kind_name])
                new_site = Site(kind_name=site.kind_name, position=site.position)
                marked_structure.append_site(new_site)
        label = f"{symbol}_{index}"
        structures[label] = marked_structure
        marked_structure.base.extras.set(
            "info",
            {
                "index": index,
                "symbol": symbol,
                "orbital": orbital,
                "peak": f"{symbol}_{orbital}",
            },
        )

    return {"structures": structures}


@node.calcfunction()
def binding_energy(
    corrections: dict = None, energy_units: str = "eV", **scf_outputs: dict
):
    output_ground = scf_outputs.pop("ground")
    results = {}
    for key, output in scf_outputs.items():
        symbol, index = key.split("_")
        de = output["energy"] - output_ground["energy"]
        if energy_units == "a.u":
            de = (output["energy"] - output_ground["energy"]) * 27.211324570
        e = de + corrections[symbol]
        results[key] = e
    return {"binding_energy": Dict(results)}
