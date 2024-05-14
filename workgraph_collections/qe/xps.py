from aiida_workgraph import WorkGraph, node
from aiida.orm import Kind, Site, StructureData, Dict, KpointsData

@node.calcfunction(outputs=[["General", "structures"], ["General", "site_info"]])
def get_marked_structures(structure, atoms_list, marker="X"):
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
    kpoints,
    pseudos,
    core_hole_pseudos,
    core_hole_treatment = "xch",
    is_molecule=False,
    metadata=None,
):
    from aiida_workgraph import WorkGraph
    from aiida_quantumespresso.calculations.pw import PwCalculation
    from copy import deepcopy
    #
    wg = WorkGraph("run_scf")
    # ground state
    pw_ground = wg.nodes.new(PwCalculation, name="ground")
    pw_ground.set(
        {
            "code": code,
            "parameters": parameters,
            "kpoints": kpoints,
            "pseudos": pseudos,
            "metadata": metadata,
            "structure": structures.pop("ground"),
        }
    )
    pw_ground.to_context = [["output_parameters", "scf.ground"]]
    # excited state node
    for key, structure in structures.items():
        pseudos1 = pseudos.copy()
        peak = structure.base.extras.get("info")["peak"]
        pseudos1["X"] = core_hole_pseudos[peak]
        # remove pseudo of non-exist element
        pseudos1 = {kind.name: pseudos1[kind.name] for kind in structure.kinds}
        # update parameters
        ch_parameters = deepcopy(parameters)
        if is_molecule:
            ch_parameters['SYSTEM']['assume_isolated']='mt'
            settings = Dict(dict={'gamma_only':True})
            kpoints = KpointsData()
            kpoints.set_kpoints_mesh([1, 1, 1])
            core_hole_treatment = "FULL"
        else:
            settings = None
        if core_hole_treatment.upper() == "XCH_SMEAR":
            ch_parameters["SYSTEM"].update({
                "occupations": "smearing",
                "tot_charge": 0,
                "nspin": 2,
                "starting_magnetization(1)": 0,
            })
        elif core_hole_treatment.upper() == "XCH_FIXED":
            ch_parameters["SYSTEM"].update({
                "occupations": "fixed",
                "tot_charge": 0,
                "nspin": 2,
                "tot_magnetization": 1,
            })
        elif core_hole_treatment.upper() == "FULL":
            ch_parameters["SYSTEM"].update({
                "tot_charge": 1,
            })
        pw_excited = wg.nodes.new(PwCalculation, name=f"pw_excited_{key}")
        pw_excited.set(
            {
                "code": code,
                "parameters": ch_parameters,
                "kpoints": kpoints,
                "pseudos": pseudos1,
                "metadata": metadata,
                "structure": structure,
                "settings": settings,
            }
        )
        pw_excited.to_context = [["output_parameters", f"scf.{key}"]]
    return wg

@node.calcfunction()
def binding_energy(corrections, **scf_outputs):
    output_ground = scf_outputs.pop("ground")
    results = {}
    for key, output in scf_outputs.items():
        symbol, index = key.split("_")
        e = output["energy"] - output_ground["energy"] + corrections[symbol]
        results[key] = e
    return {"binding_energy": Dict(results)}


def xps_workgraph():
    wg = WorkGraph("xps")
    get_marked_structures1 = wg.nodes.new(
        get_marked_structures, name="get_marked_structures", marker="X"
    )
    run_scf1 = wg.nodes.new(run_scf, name="run_scf", structures=get_marked_structures1.outputs["structures"])
    wg.nodes.new(binding_energy, name="binding_energy", scf_outputs=run_scf1.outputs["result"])
    return wg
