from .namelist import NamelistTemplate


class DosTemplate(NamelistTemplate):
    _label = "dos"

    def __init__(self):
        super().__init__(
            [
                "dos",
            ]
        )

    def read_results(self, directory):
        path = directory / "pwscf.dos"
        results = read_dos_out(path)
        return results


def read_dos_out(filepath):
    """Read the dos.
    Adapted from https://github.com/aiidateam/aiida-quantumespresso/blob/main/src/aiida_quantumespresso/parsers/dos.py
    """
    import numpy as np

    array_names = [[], []]
    array_units = [[], []]
    array_names[0] = [
        "dos_energy",
        "dos",
        "integrated_dos",
    ]  # When spin is not displayed
    array_names[1] = [
        "dos_energy",
        "dos_spin_up",
        "dos_spin_down",
        "integrated_dos",
    ]  # When spin is displayed
    array_units[0] = ["eV", "states/eV", "states"]  # When spin is not displayed
    array_units[1] = [
        "eV",
        "states/eV",
        "states/eV",
        "states",
    ]  # When spin is displayed

    with open(filepath, "r") as dos_file:
        lines = dos_file.readlines()
        dos_header = lines[0]
        try:
            dos_data = np.genfromtxt(lines[1:])
        except ValueError:
            raise "dosfile could not be loaded using genfromtxt"
        if len(dos_data) == 0:
            raise "Dos file is empty."
        if np.isnan(dos_data).any():
            raise "Dos file contains non-numeric elements."

        # Checks the number of columns, essentially to see whether spin was used
        if len(dos_data[0]) == 3:
            # spin is not used
            array_names = array_names[0]
            array_units = array_units[0]
            spin = False
        elif len(dos_data[0]) == 4:
            # spin is used
            array_names = array_names[1]
            array_units = array_units[1]
            spin = True
        else:
            raise "Dos file in format that the parser is not designed to handle."

        i = 0
        array_data = {}
        array_data["header"] = np.array(dos_header)
        while i < len(array_names):
            array_data[array_names[i]] = dos_data[:, i]
            array_data[array_names[i] + "_units"] = np.array(array_units[i])
            i += 1
        return {"dos": array_data, "spin": spin}
