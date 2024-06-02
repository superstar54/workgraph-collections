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
        path = directory / "pwscf.dos"
        results = read_pdos_out(path)
        return results


def read_pdos_out(filepath):
    """"""
    return {"pdos": None, "spin": None}
