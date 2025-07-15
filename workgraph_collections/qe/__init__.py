from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.calculations.dos import DosCalculation
from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation
from aiida_quantumespresso.calculations.pp import PpCalculation
from aiida_workgraph import task

PwBaseTask = task()(PwBaseWorkChain)
PwRelaxTask = task()(PwRelaxWorkChain)
DosTask = task()(DosCalculation)
ProjwfcTask = task()(ProjwfcCalculation)
PpTask = task()(PpCalculation)
