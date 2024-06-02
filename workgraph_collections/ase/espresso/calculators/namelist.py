from pathlib import Path
from ase.io.espresso import Namelist, write_fortran_namelist
from ase.calculators.espresso import EspressoProfile
from ase.calculators.genericfileio import CalculatorTemplate


class NamelistTemplate(CalculatorTemplate):
    _label = ""  # This is a placeholder for the label of the calculator

    def __init__(self, properties=None):
        super().__init__(
            self._label,
            properties,
        )
        self.inputname = f"{self._label}.{self._label}i"
        self.outputname = f"{self._label}.{self._label}o"
        self.errorname = f"{self._label}.err"

    def execute(self, directory, profile):
        profile.run(
            directory, self.inputname, self.outputname, errorfile=self.errorname
        )

    def load_profile(self, cfg, **kwargs):
        return EspressoProfile.from_config(cfg, self.name, **kwargs)

    def socketio_parameters(self, unixsocket, port):
        return {}

    def socketio_argv(self, profile, unixsocket, port):
        if unixsocket:
            ipi_arg = f"{unixsocket}:UNIX"
        else:
            ipi_arg = f"localhost:{port:d}"  # XXX should take host, too
        return profile.get_calculator_command(self.inputname) + [
            "--ipi",
            ipi_arg,
        ]

    def write_input(self, profile, directory, atoms, parameters, properties):
        """Write the input file to the directory."""

        input_data = Namelist(parameters.pop("input_data", None))
        input_data.to_nested(self._label)

        parameters["input_data"] = input_data

        with Path(directory, self.inputname).open(mode="w") as fd:
            write_fortran_namelist(
                fd,
                binary=self._label,
                properties=properties,
                **parameters,
            )
