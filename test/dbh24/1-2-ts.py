import numpy as np

import ase.io
from ase.build.molecule import molecule

from atoms_pulling import PullingAtoms, PullingBFGS
from ase.calculators.emt import EMT
from ase.calculators.gromacs import Gromacs
from ase.build import minimize_rotation_and_translation
from atoms_pulling.pulling import pulling
from atoms_pulling.robust import RigidCalculator
from ase.calculators.dftd3 import DFTD3

import argparse


def main(args):
    start = ase.io.read('1.xyz')
    end = ase.io.read('2.xyz')
    # end.rotate(180, axis, center=(0, 0, 0))
    ase.io.write('start.traj', [start, end])
    calc = RigidCalculator(start)
    if args.calc.lower() == 'dftd3':
        calc = DFTD3()
    start.calc = calc
    use_sopt = args.use_sopt
    # pulling_threshold=1.0
    pulling(start, end, pulling_threshold=args.pulling_threshold,
            max_pulling_step=1000,
            use_sopt=use_sopt,
            remove_translate_rotate=False,
            calculator=calc,
            converge_threshold=args.converge_threshold,
            )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_sopt', action='store_true')
    parser.add_argument('--calc', type=str,
                        choices=['rigid', 'dftd3'], default='Rigid')
    parser.add_argument('--pulling_threshold', type=float, default=1.0)
    parser.add_argument('--converge_threshold', type=float, default=0.5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
