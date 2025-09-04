from aiida_workgraph import WorkGraph, task, dynamic
from pymatgen.analysis.elasticity.strain import DeformedStructureSet
from ase import Atoms
from typing import Annotated, Any


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


@task.graph(outputs=dynamic(Any))
def run_relaxation(
    deformed_structure_set: DeformedStructureSet, relax_inputs: dict
) -> WorkGraph:
    """Run relaxation for each deformed structure."""
    from .pw import pw_calculator

    # Be careful, we generate new data here, thus break the data provenance!
    deformed_atoms = [
        structure.to_ase_atoms() for structure in deformed_structure_set.value
    ]
    results = {}
    for i in range(len(deformed_atoms)):
        relax_out = pw_calculator(
            atoms=deformed_atoms[i],
            **relax_inputs,
        )
        results[f"atoms_{i}"] = relax_out.trajectory
    return results


@task()
def fit_elastic_constants(
    atoms: Atoms,
    deformed_structure_set: DeformedStructureSet,
    relax_results: Annotated[dict, dynamic(dict)],
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
        stress = relax_results[f"atoms_{i}"].get_array("stress")[0]
        stress = [
            stress[0][0],
            stress[1][1],
            stress[2][2],
            stress[1][2],
            stress[0][2],
            stress[0][1],
        ]
        stress = Stress.from_voigt(stress)
        stresses.append(stress)
        strains.append(deformed_structure_set.deformations[i].green_lagrange_strain)
    if symmetry:
        strains, stresses = restore_full_strains_stresses(structure, strains, stresses)
    elastic_tensor = ElasticTensor.from_pseudoinverse(strains, stresses)
    # elastic_tensor = ElasticTensor.from_independent_strains(strains, stresses)
    elastic_constants = elastic_tensor.voigt
    return elastic_constants


@task.graph()
def ElasticWorkgraph(
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

    # -------- relax -----------
    if run_relax:
        relax_input_data = deepcopy(input_data)
        inputs = {
            "command": command,
            "input_data": relax_input_data,
            "kpts": kpts,
            "pseudopotentials": pseudopotentials,
            "pseudo_dir": pseudo_dir,
        }
        relax_out = pw_calculator(
            atoms=atoms,
            calculation="vc-relax",
            metadata=metadata,
            computer=computer,
            **inputs,
        )
        atoms = relax_out.atoms
    # -------- deformed_structure -----------
    deformed_structure_out = get_deformed_structure_set(
        atoms=atoms,
        norm_strains=norm_strains,
        shear_strains=shear_strains,
        symmetry=symmetry,
    )
    # -------- run_relaxation -----------
    run_relaxation_out = run_relaxation(
        deformed_structure_set=deformed_structure_out.result,
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
    return fit_elastic_constants(
        atoms=atoms,
        deformed_structure_set=deformed_structure_out.result,
        relax_results=run_relaxation_out,
        symmetry=symmetry,
    )
