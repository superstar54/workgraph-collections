from .namelist import NamelistTemplate


class PpTemplate(NamelistTemplate):
    _label = "pp"

    def __init__(self):
        super().__init__(
            [
                "pp",
            ]
        )

    def read_results(self, directory):
        return {"pp": None}
