from aiida_workgraph import WorkGraph, task
from pymatgen.analysis.elasticity.strain import DeformedStructureSet
from ase import Atoms


@task()
def get_deformed_structure_set(
    atoms: Atoms, norm_strains: list, shear_strains: list, symmetry: bool = True
) -> DeformedStructureSet:
    """Get the deformed structure set."""
    from pymatgen.io.ase import AseAtomsAdaptor

    structure = AseAtomsAdaptor.get_structure(atoms)

    deformed_structure_set = DeformedStructureSet(
        structure,
        norm_strains=norm_strains,
        shear_strains=shear_strains,
        symmetry=symmetry,
    )

    return deformed_structure_set


@task.graph_builder(outputs=[{"name": "relax_results", "from": "context.results"}])
def run_relaxation(
    deformed_structure_set: DeformedStructureSet, relax_inputs: dict
) -> WorkGraph:
    """Run relaxation for each deformed structure."""
    from aiida_workgraph import WorkGraph
    from .pw import pw_calculator

    wg = WorkGraph()
    wg.context = {"deformed_structure": deformed_structure_set}
    deformed_atoms = [
        structure.to_ase_atoms() for structure in deformed_structure_set.value
    ]
    # becareful, we generate new data here, thus break the data provenance!
    # that's why I put the deformed_structure in the context, so that we can link them
    for i in range(len(deformed_atoms)):
        relax = wg.add_task(
            "workgraph.pythonjob",
            function=pw_calculator,
            name=f"relax_{i}",
            atoms=deformed_atoms[i],
        )
        relax.set(relax_inputs)
        # save the output parameters to the context
        relax.set_context({f"results.{i}": "parameters"})
    return wg


@task()
def fit_elastic_constants(
    atoms: Atoms,
    deformed_structure_set: DeformedStructureSet,
    relax_results: dict,
    symmetry: bool = True,
):
    """Fit the elastic constants."""
    from pymatgen.analysis.elasticity.elastic import ElasticTensor
    from pymatgen.analysis.elasticity import Stress
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.core.tensors import TensorMapping

    structure = AseAtomsAdaptor.get_structure(atoms)

    def restore_full_strains_stresses(
        structure,
        strains,
        stresses,
        symprec: float = 1e-5,
    ):
        """
        Use symmetry to restore full strains and stresses.
        """
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        symm_ops = sga.get_symmetry_operations(cartesian=True)
        mapping = TensorMapping(strains, stresses)
        for i, strain in enumerate(strains):
            for symm_op in symm_ops:
                new_strain = strain.transform(symm_op)
                mapping[new_strain] = stresses[i].transform(symm_op)
        return mapping._tensor_list, mapping.values()

    stresses = []
    strains = []
    for i in range(len(deformed_structure_set)):
        stress = Stress.from_voigt(relax_results[f"{i}"]["stress"])
        stresses.append(stress)
        strains.append(deformed_structure_set.deformations[i].green_lagrange_strain)
    if symmetry:
        strains, stresses = restore_full_strains_stresses(structure, strains, stresses)
    elastic_tensor = ElasticTensor.from_pseudoinverse(strains, stresses)
    # elastic_tensor = ElasticTensor.from_independent_strains(strains, stresses)
    elastic_constants = elastic_tensor.voigt
    return elastic_constants


@task.graph_builder(outputs=[{"name": "result", "from": "fit_elastic.result"}])
def elastic_workgraph(
    atoms: Atoms = None,
    command: str = "pw.x",
    computer: str = "localhost",
    norm_strains: list = [-0.01, -0.005, 0.005, 0.01],
    shear_strains: list = [-0.06, -0.03, 0.03, 0.06],
    symmetry: bool = True,
    pseudopotentials: dict = None,
    pseudo_dir: str = None,
    kpts: list = None,
    input_data: dict = None,
    metadata: dict = None,
    run_relax: bool = True,
):
    """Workgraph for elastic calculation.
    1. Get the deformed atoms.
    2. Run the relax calculation for each deformed atoms.
    3. Fit the elastic.
    """
    from .pw import pw_calculator
    from copy import deepcopy

    input_data = input_data or {}

    wg = WorkGraph("Elastic")
    # -------- relax -----------
    if run_relax:
        relax_task = wg.add_task(
            "workgraph.pythonjob",
            function=pw_calculator,
            name="relax",
            atoms=atoms,
            calculation="vc-relax",
            metadata=metadata,
            computer=computer,
        )
        relax_input_data = deepcopy(input_data)
        relax_task.set(
            {
                "command": command,
                "input_data": relax_input_data,
                "kpts": kpts,
                "pseudopotentials": pseudopotentials,
                "pseudo_dir": pseudo_dir,
            }
        )
        atoms = relax_task.outputs["atoms"]
    # -------- deformed_structure -----------
    deformed_structure_task = wg.add_task(
        "workgraph.pythonjob",
        function=get_deformed_structure_set,
        name="deformed_structure",
        atoms=atoms,
        norm_strains=norm_strains,
        shear_strains=shear_strains,
        symmetry=symmetry,
        computer=computer,
        metadata=metadata,
    )
    # -------- run_relaxation -----------
    run_relaxation_task = wg.add_task(
        run_relaxation,
        name="run_relaxation",
        deformed_structure_set=deformed_structure_task.outputs.result,
        relax_inputs={
            "command": command,
            "input_data": input_data,
            "kpts": kpts,
            "pseudopotentials": pseudopotentials,
            "pseudo_dir": pseudo_dir,
            "metadata": metadata,
            "computer": computer,
        },
    )
    # -------- fit_elastic -----------
    wg.add_task(
        "workgraph.pythonjob",
        function=fit_elastic_constants,
        name="fit_elastic",
        atoms=atoms,
        deformed_structure_set=deformed_structure_task.outputs.result,
        relax_results=run_relaxation_task.outputs["relax_results"],
        symmetry=symmetry,
        computer=computer,
        metadata=metadata,
    )
    return wg
