from workgraph_collections.ase.espresso.base import ld1_calculator
import numpy as np

input_data = {
    "input": {
        "title": "Pt",
        "zed": 78.0,
        "rel": 1,
        "config": "[Xe] 4f13 6s1 6p0 5d9",
        "iswitch": 3,
        "dft": "PBE",
    },
    "inputp": {
        "lpaw": True,
        "use_xsd": False,
        "pseudotype": 3,
        "file_pseudopw": "Pt.star4f.pbe-n-kjpaw_psl.1.0.0.UPF",
        "author": "Test",
        "lloc": -1,
        "rcloc": 2.4,
        "which_augfun": "PSQ",
        "rmatch_augfun_nc": True,
        "nlcc": True,
        "new_core_ps": True,
        "rcore": 1.8,
        "tm": True,
    },
}
pseudo_potential_test_cards = """
6
6S  1  0  1.00  0.00  2.00  2.20  0.0
6S  1  0  0.00  4.40  2.00  2.20  0.0
6P  2  1  0.00  0.00  2.30  2.50  0.0
6P  2  1  0.00  6.40  2.30  2.50  0.0
5D  3  2  9.00  0.00  1.00  2.20  0.0
5D  3  2  0.00  0.80  1.00  2.20  0.0
"""


def test_base_ld1():
    results = ld1_calculator(
        input_data=input_data, pseudo_potential_test_cards=pseudo_potential_test_cards
    )
    np.isclose(results["Etot"]["eV"], -500976.284196)
