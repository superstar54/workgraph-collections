from aiida_workgraph import WorkGraph, node
from aiida.engine import calcfunction
from aiida.orm import Kind, Site, Dict, StructureData


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
    basis_pseudo_files,
    core_hole_pseudos,
    core_hole_treatment = "xch",
    metadata=None,
):
    from aiida_cp2k.calculations import Cp2kCalculation
    from copy import deepcopy
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
        ch_parameters = deepcopy(parameters)
        symbol = key.split("_")[0]
        ch_parameters["FORCE_EVAL"]["SUBSYS"]["KIND"].append(core_hole_pseudos[symbol])
        if core_hole_treatment.upper == "XCH":
            ch_parameters["FORCE_EVAL"]["DFT"].update({"UKS": True,
                                            "MULTIPLICITY": 2,
                                            "CHARGE": -1
                                            })
        else:
            ch_parameters["FORCE_EVAL"]["DFT"].update({"UKS": False,
                                            "MULTIPLICITY": 1,
                                            "CHARGE": 0
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
        e = (output["energy"] - output_ground["energy"])*27.211324570 + corrections[key]
        results[key] = e
    return {"binding_energy": Dict(results)}


def xps_spectra_broadening(
    points, equivalent_sites_data, gamma=0.3, sigma=0.3, label="", intensity=1.0
):
    """Broadening the XPS spectra with Voigt function and return the spectra data"""

    import numpy as np
    from scipy.special import voigt_profile  # pylint: disable=no-name-in-module

    result_spectra = {}
    fwhm_voight = gamma / 2 + np.sqrt(gamma**2 / 4 + sigma**2)
    for element, point in points.items():
        result_spectra[element] = {}
        final_spectra_y_arrays = []
        total_multiplicity = sum(
            [equivalent_sites_data[site]["multiplicity"] for site in point]
        )
        max_core_level_shift = max(point.values())
        min_core_level_shift = min(point.values())
        # Energy range for the Broadening function
        x_energy_range = np.linspace(
            min_core_level_shift - fwhm_voight - 1.5,
            max_core_level_shift + fwhm_voight + 1.5,
            500,
        )
        for site in point:
            # Weight for the spectra of every atom
            intensity = equivalent_sites_data[site]["multiplicity"] * intensity
            relative_core_level_position = point[site]
            y = (
                intensity
                * voigt_profile(
                    x_energy_range - relative_core_level_position, sigma, gamma
                )
                / total_multiplicity
            )
            result_spectra[element][site] = [x_energy_range, y]
            final_spectra_y_arrays.append(y)
        total = sum(final_spectra_y_arrays)
        result_spectra[element]["total"] = [x_energy_range, total]
    return result_spectra

@node.graph_builder()
def xps_workflow(name="binding_energy"):
    wg = WorkGraph(name)
    structures_node = wg.nodes.new(get_marked_structures, name="get_marked_structures")
    scf_node = wg.nodes.new(run_scf, name="run_scf",
                            structures=structures_node.outputs["structures"])
    wg.nodes.new(binding_energy, name="binding_energy",
                    scf_outputs=scf_node.outputs["result"])
    return wg
