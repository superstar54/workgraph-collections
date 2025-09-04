from aiida_cp2k.calculations import Cp2kCalculation
from aiida_workgraph import task

Cp2kTask = task(Cp2kCalculation)
