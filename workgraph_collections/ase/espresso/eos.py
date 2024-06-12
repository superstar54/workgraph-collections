from aiida_workgraph import node, WorkGraph
from workgraph_collections.ase.common.eos import generate_scaled_atoms, fit_eos
from ase import Atoms


@node.graph_builder(outputs=[["context.results", "scf_results"]])
def all_scf(scaled_atoms, scf_inputs):
    """Run the scf calculation for each atoms."""
    from aiida_workgraph import WorkGraph
    from .base import pw_calculator

    wg = WorkGraph()
    wg.context = {"scaled_atoms": scaled_atoms}
    # becareful, we generate new data here, thus break the data provenance!
    # that's why I put the scaled atoms in the context, so that we can link them
    for key, atoms in scaled_atoms.value.items():
        scf = wg.nodes.new(
            pw_calculator, name=f"scf_{key}", atoms=atoms, run_remotely=True
        )
        scf.set(scf_inputs)
        # save the output parameters to the context
        scf.to_context = [["results", f"results.{key}"]]
    return wg


@node.graph_builder(outputs=[["fit_eos.result", "result"]])
def eos_workgraph(
    atoms: Atoms = None,
    command: str = "pw.x",
    computer: str = "localhost",
    scales: list = None,
    pseudopotentials: dict = None,
    pseudo_dir: str = None,
    kpts: list = None,
    input_data: dict = None,
    metadata: dict = None,
    run_relax: bool = True,
):
    """Workgraph for EOS calculation.
    1. Get the scaled atoms.
    2. Run the SCF calculation for each scaled atoms.
    3. Fit the EOS.
    """
    from .base import pw_calculator
    from copy import deepcopy

    input_data = input_data or {}

    wg = WorkGraph("EOS")
    # -------- relax -----------
    if run_relax:
        relax_node = wg.nodes.new(
            pw_calculator,
            name="relax",
            atoms=atoms,
            run_remotely=True,
            metadata=metadata,
            computer=computer,
        )
        relax_input_data = deepcopy(input_data)
        relax_input_data.setdefault("CONTROL", {})
        relax_input_data["CONTROL"]["calculation"] = "vc-relax"
        relax_node.set(
            {
                "command": command,
                "input_data": relax_input_data,
                "kpts": kpts,
                "pseudopotentials": pseudopotentials,
                "pseudo_dir": pseudo_dir,
            }
        )
        atoms = relax_node.outputs["atoms"]
    # -------- scale_atoms -----------
    scale_atoms_node = wg.nodes.new(
        generate_scaled_atoms,
        name="scale_atoms",
        atoms=atoms,
        scales=scales,
        computer=computer,
        metadata=metadata,
        run_remotely=True,
    )
    # -------- all_scf -----------
    all_scf1 = wg.nodes.new(
        all_scf,
        name="all_scf",
        scaled_atoms=scale_atoms_node.outputs["scaled_atoms"],
        scf_inputs={
            "command": command,
            "input_data": input_data,
            "kpts": kpts,
            "pseudopotentials": pseudopotentials,
            "pseudo_dir": pseudo_dir,
            "metadata": metadata,
            "computer": computer,
        },
    )
    # -------- fit_eos -----------
    wg.nodes.new(
        fit_eos,
        name="fit_eos",
        volumes=scale_atoms_node.outputs["volumes"],
        scf_results=all_scf1.outputs["scf_results"],
        computer=computer,
        metadata=metadata,
        run_remotely=True,
    )
    return wg
