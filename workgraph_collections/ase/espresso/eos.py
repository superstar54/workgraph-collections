from aiida_workgraph import task, spec
from workgraph_collections.ase.common.eos import generate_scaled_atoms, fit_eos
from ase import Atoms
from typing import Annotated
from .pw import pw_calculator


@task.graph()
def calc_all_structures(
    structures: Annotated[dict, spec.dynamic(Atoms)],
    pw_inputs: Annotated[
        dict, pw_calculator.inputs, spec.select(exclude=["atoms"])
    ] = None,
) -> Annotated[
    dict, spec.namespace(results=spec.dynamic(pw_calculator.outputs.parameters))
]:
    """Run SCF calculations for all structures."""

    pw_inputs = pw_inputs or {}
    results = {}
    for name, atoms in structures.items():
        scf_output = pw_calculator(
            atoms=atoms,
            **pw_inputs,
        )
        results[name] = scf_output.parameters

    return {"results": results}


@task.graph()
def EosWorkGraph(
    atoms: Atoms = None,
    scales: list = None,
    pw_inputs: Annotated[
        dict, pw_calculator.inputs, spec.select(exclude=["atoms"])
    ] = None,
    run_relax: bool = True,
):
    """Workgraph for EOS calculation.
    1. Get the scaled atoms.
    2. Run the SCF calculation for each scaled atoms.
    3. Fit the EOS.
    """

    pw_inputs = pw_inputs or {}
    # -------- relax -----------
    if run_relax:
        pw_inputs["calculation"] = "vc-relax"
        relax_output = pw_calculator(atoms=atoms, **pw_inputs)
        atoms = relax_output.atoms
    # -------- scale_atoms -----------
    scale_output = generate_scaled_atoms(atoms=atoms, scales=scales)
    scf_outputs = calc_all_structures(
        structures=scale_output.scaled_atoms, pw_inputs=pw_inputs
    )
    # -------- fit_eos -----------
    return fit_eos(volumes=scale_output.volumes, scf_results=scf_outputs.results).result
