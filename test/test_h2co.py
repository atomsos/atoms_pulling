import numpy as np

import ase.io
from ase.build.molecule import molecule

from atoms_pulling import PullingAtoms, PullingBFGS
from ase.calculators.emt import EMT
from ase.calculators.gromacs import Gromacs
from ase.build import minimize_rotation_and_translation
from atoms_pulling.pulling import pulling


def main3():
    start = ase.io.read('h2+co.xyz')
    end = molecule('H2CO')
    pulling(start, end, pulling_threshold=0.5, use_sopt=True)


if __name__ == "__main__":
    main3()
