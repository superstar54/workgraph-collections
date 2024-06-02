from .namelist import NamelistTemplate


class ProjwfcTemplate(NamelistTemplate):
    _label = "projwfc"

    def __init__(self):
        super().__init__(
            [
                "pdos",
            ]
        )

    def read_results(self, directory):
        results = read_pdos_out(directory)
        return results


def read_pdos_out(directory):
    """Read the dos.
    Adapted from https://github.com/aiidateam/aiida-quantumespresso/blob/main/src/
    aiida_quantumespresso/parsers/projwfc.py
    """
    import numpy as np
    import os

    pdos_tot_filen = directory / "pwscf.pdos_tot"
    pdos_tot = {}
    with open(pdos_tot_filen, "r") as f:
        pdos_tot_array = np.atleast_2d(np.genfromtxt(f))
        pdos_tot["energy"] = pdos_tot_array[:, 0]
        pdos_tot["dos"] = pdos_tot_array[:, 1]
    pdos_atm_array_dict = {}
    # find all files with name in the form of *pdos_atm_*
    pdos_atm_files = {
        name: directory / name for name in os.listdir(directory) if "pdos_atm" in name
    }
    for name, file in pdos_atm_files.items():
        with open(file, "r") as pdosatm_file:
            pdos_atm_array_dict[name] = np.atleast_2d(np.genfromtxt(pdosatm_file))

    return {"pdos": {"atom": pdos_atm_array_dict, "totol": pdos_tot}}
