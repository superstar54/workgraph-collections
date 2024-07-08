import pytest
from ase.build import molecule
from ase import Atoms

from aiida import load_profile

load_profile()


@pytest.fixture
def n_atom():
    n_atom = Atoms("N", pbc=True)
    n_atom.center(vacuum=5.0)
    return n_atom


@pytest.fixture
def n2_molecule():
    n2_molecule = molecule("N2", pbc=True)
    n2_molecule.center(vacuum=5.0)
    return n2_molecule


@pytest.fixture
def metadata_aiida():
    metadata = {
        "options": {
            "prepend_text": """
    eval "$(conda shell.posix hook)"
    conda activate base
    export OMP_NUM_THREADS=1
            """,
        }
    }
    return metadata


@pytest.fixture
def pseudo_dir():
    # current file path + espresso_pseudo
    return "/".join(__file__.split("/")[:-1]) + "/espresso_pseudo"
