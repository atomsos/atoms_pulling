import numpy as np

import ase.io
from ase.build.molecule import molecule

from atoms_pulling import PullingAtoms, PullingBFGS
from ase.calculators.emt import EMT
from ase.calculators.gromacs import Gromacs
from ase.build import minimize_rotation_and_translation
from atoms_pulling.pulling import pulling, pulling_rigid
from atoms_pulling.rigid import RobustCalculator


def main(axis='x', name='H2CO'):
    start = ase.io.read('TiO2_H2.cif')
    end = ase.io.read('TiO2_H-H.cif')
    # end.rotate(180, axis, center=(0, 0, 0))
    ase.io.write('start.traj', [start, end])
    calc = RobustCalculator(start)
    start, end = pulling_rigid(start, end)
    pulling(start, end, pulling_threshold=1.0,
            use_sopt=False, remove_translate_rotate=False,
            calculator=calc)


if __name__ == '__main__':
    main()
