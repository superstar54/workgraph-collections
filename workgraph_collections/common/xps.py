from aiida.orm import Dict
from aiida_workgraph import node


@node.calcfunction()
def binding_energy(
    corrections: dict = None,
    energy_units: str = "eV",
    sites_info: dict = None,
    **scf_outputs
):
    output_ground = scf_outputs.pop("ground")
    results = {}
    equivalent_sites_data = sites_info.get_dict()["equivalent_sites_data"]
    for key, output in scf_outputs.items():
        symbol = equivalent_sites_data[key]["symbol"]
        de = output["energy"] - output_ground["energy"]
        if energy_units.value == "a.u":
            de *= 27.211324570
        e = de + corrections[symbol]
        results[key] = e
    return Dict(results)
