from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.calculations.dos import DosCalculation
from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation
from aiida_quantumespresso.calculations.pp import PpCalculation
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_workgraph import task, spec
from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import (
    seekpath_structure_analysis,
)
from aiida import orm

PwBaseTask = task()(PwBaseWorkChain)
PwRelaxTask = task()(PwRelaxWorkChain)
PwTask = task()(PwCalculation)
DosTask = task()(DosCalculation)
ProjwfcTask = task()(ProjwfcCalculation)
PpTask = task()(PpCalculation)

# Add only two outputs port here, because we only use these outputs in the following.
SeekpathTask = task(
    outputs=spec.namespace(
        primitive_structure=orm.StructureData, explicit_kpoints=orm.KpointsData
    )
)(seekpath_structure_analysis)
