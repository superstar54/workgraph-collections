# workgraph-collections
[![PyPI version](https://badge.fury.io/py/workgraph-collections.svg)](https://badge.fury.io/py/workgraph-collections)
[![Unit test](https://github.com/superstar54/workgraph-collections/actions/workflows/ci.yaml/badge.svg)](https://github.com/superstar54/workgraph-collections/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/superstar54/workgraph-collections/branch/main/graph/badge.svg)](https://codecov.io/gh/superstar54/workgraph-collections)
[![Docs status](https://readthedocs.org/projects/workgraph-collections/badge)](http://workgraph-collections.readthedocs.io/)


This repository offers a variety of workgraphs tailored to different computational codes, providing practical demonstrations of how to utilize aiida-workgraph. Please note, however, that these examples are intended for illustrative purposes only and may not always represent the most correct or efficient usage.


## Installation
Install the latest version of the package from github:

```bash
pip install git+https://github.com/superstar54/workgraph-collections.git
```


## Qauntum ESPRESSO

- [Equation of state (EOS)](https://workgraph-collections.readthedocs.io/en/latest/qe/eos.html)
- [Bands structure](https://workgraph-collections.readthedocs.io/en/latest/qe/bands.html)
- [Projected density of states (PDOS)](https://workgraph-collections.readthedocs.io/en/latest/qe/pdos.html)
- [X-ray photoelectron spectroscopy (XPS)](https://workgraph-collections.readthedocs.io/en/latest/qe/xps.html)
- [Bader Charge](https://workgraph-collections.readthedocs.io/en/latest/qe/bader.html)
- [Wannier90](https://workgraph-collections.readthedocs.io/en/latest/qe/wannier90.html)

## CP2K

- [Equation of state (EOS)](https://workgraph-collections.readthedocs.io/en/latest/cp2k/eos.html)
- [X-ray photoelectron spectroscopy (XPS)](https://workgraph-collections.readthedocs.io/en/latest/cp2k/xps.html)


## ASE

- EMT calculator

  - [Calculator](https://workgraph-collections.readthedocs.io/en/latest/ase/emt/base.html)
  - [Atomization energy](https://workgraph-collections.readthedocs.io/en/latest/ase/emt.html)

- Espresso calculator

  - [Calculator](https://workgraph-collections.readthedocs.io/en/latest/ase/espresso/base.html)
  - [Atomization energy](https://workgraph-collections.readthedocs.io/en/latest/ase/espresso/atomization.html)
  - [Equation of state (EOS)](https://workgraph-collections.readthedocs.io/en/latest/ase/espresso/eos.html)
  - [Elastic constants](https://workgraph-collections.readthedocs.io/en/latest/ase/espresso/elastic.html)
  - [Bands structure](https://workgraph-collections.readthedocs.io/en/latest/ase/espresso/bands.html)
  - [Projected density of states (PDOS)](https://workgraph-collections.readthedocs.io/en/latest/ase/espresso/pdos.html)
  - [Surface slabs](https://workgraph-collections.readthedocs.io/en/latest/ase/espresso/surface_slabs.html)
  - [Bader Charge](https://workgraph-collections.readthedocs.io/en/latest/ase/espresso/bader.html)
  - [X-ray photoelectron spectroscopy (XPS)](https://workgraph-collections.readthedocs.io/en/latest/ase/espresso/xps.html)
  - [X-ray Absorption Near Edge Structure (XANES)](https://workgraph-collections.readthedocs.io/en/latest/ase/espresso/xas.html) (Ongoing)
  - [Generate core-hole pseudopotential](https://workgraph-collections.readthedocs.io/en/latest/ase/espresso/core_hole_pseudo.html)


## GPAW

- [Calculator](https://workgraph-collections.readthedocs.io/en/latest/gpaw/base.html)
- [Wannier90](https://workgraph-collections.readthedocs.io/en/latest/gpaw/wannier90.html)
