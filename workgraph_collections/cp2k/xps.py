from aiida_workgraph import WorkGraph, node
from aiida.engine import calcfunction
from aiida.orm import Kind, Site, Dict, StructureData


@node.calcfunction(outputs=[["General", "structures"], ["General", "site_info"]])
def get_structures(structure, atoms_list, marker="X"):
    """"""
    structures = {
        "ground": StructureData(ase=structure.get_ase()),
    }
    

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
        marked_structure.base.extras.set("info", {
            "index": index,
            "symbol": symbol,
            "orbital": orbital,
            "peak": f"{symbol}_{orbital}",
        })

    return {"structures": structures}


# the structures is used to generate the workgraph dynamically.
@node.graph_builder(outputs=[["context.scf", "result"]])
def run_scf(
    structures,
    code,
    parameters,
    basis_pseudo_files,
    core_hole_pseudos,
    metadata,
):
    from aiida_cp2k.calculations import Cp2kCalculation
    wg = WorkGraph("run_scf")
    # ground state
    scf_ground = wg.nodes.new(Cp2kCalculation, name="scf_ground")
    scf_ground.set(
        {
            "code": code,
            "parameters": Dict(parameters),
            "metadata": metadata,
            "file": basis_pseudo_files,
            "structure": structures.pop("ground"),
        }
    )
    scf_ground.to_context = [["output_parameters", "scf.ground"]]
    # excited state node
    for key, structure in structures.items():
        ch_parameters = parameters.copy()
        symbol = key.split("_")[0]
        ch_parameters["FORCE_EVAL"]["SUBSYS"]["KIND"].append(core_hole_pseudos[symbol])
        ch_parameters["FORCE_EVAL"]["DFT"].update({"UKS": True,
                                            "MULTIPLICITY": 2,
                                            "CHARGE": -1
                                            })
        scf_ch = wg.nodes.new(Cp2kCalculation, name=f"scf_{key}")
        scf_ch.set(
            {
                "code": code,
                "parameters": Dict(ch_parameters),
                "metadata": metadata,
                "file": basis_pseudo_files,
                "structure": structure,
            }
        )
        scf_ch.to_context = [["output_parameters", f"scf.{key}"]]
    return wg



@calcfunction
def binding_energy(corrections, **scf_outputs):
    from aiida.orm import Float
    output_ground = scf_outputs.pop("ground")
    results = {}
    for key, output in scf_outputs.items():
        symbol, index = key.split("_")
        e = (output["energy"] - output_ground["energy"])*27.211324570 + corrections[symbol]
        results[key] = e
    return {"binding_energy": Dict(results)}

@node.graph_builder()
def xps_workflow(name="binding_energy"):
    wg = WorkGraph(name)
    structures_node = wg.nodes.new(get_structures, name="get_structures")
    scf_node = wg.nodes.new(run_scf, name="run_scf",
                            structures=structures_node.outputs["structures"])
    wg.nodes.new(binding_energy, name="binding_energy",
                    scf_outputs=scf_node.outputs["result"])
    return wg
