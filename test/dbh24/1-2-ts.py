import numpy as np

import ase.io
from ase import Atoms
from ase.build.molecule import molecule

from atoms_pulling import PullingAtoms, PullingBFGS
from ase.calculators.emt import EMT
from ase.calculators.gromacs import Gromacs
from ase.build import minimize_rotation_and_translation
from atoms_pulling.pulling import pulling, pulling_rigid
from atoms_pulling.rigid import RobustCalculator
from ase.calculators.dftd3 import DFTD3

import argparse


def run(start: Atoms, end: Atoms, args):
    ase.io.write('start.traj', [start, end])
    if args.calc.lower() == 'dftd3':
        calc = DFTD3()
        start.calc = calc
    elif args.calc.lower() == 'robust':
        calc = RobustCalculator(start)
        start.calc = calc
    use_sopt = args.use_sopt
    # pulling_threshold=1.0
    start0 = start.copy()
    end0 = end.copy()
    if args.use_rigid:
        # import pdb ; pdb.set_trace()
        start, end = pulling_rigid(start, end)
        ase.io.write('init.traj', [start0, start, end, end0])


    pulling(start, end, pulling_threshold=args.pulling_threshold,
            max_pulling_step=1000,
            use_sopt=use_sopt,
            remove_translate_rotate=False,
            calculator=calc,
            converge_threshold=args.converge_threshold,
            )


def main(args):
    import os

    start = ase.io.read('1.xyz')
    end = ase.io.read('2.xyz')
    # end.rotate(180, axis, center=(0, 0, 0))

    basedir = os.getcwd()

    os.chdir(basedir)
    os.makedirs('forward', exist_ok=True)
    os.chdir('forward')
    run(start.copy(), end.copy(), args)

    os.chdir(basedir)
    os.makedirs('backward', exist_ok=True)
    os.chdir('backward')

    run(end.copy(), start.copy(), args)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_sopt', action='store_true')
    parser.add_argument('--use_rigid', action='store_true')
    parser.add_argument('--calc', type=str,
                        choices=['robust', 'dftd3'], default='Rigid')
    parser.add_argument('--pulling_threshold', type=float, default=1.0)
    parser.add_argument('--converge_threshold', type=float, default=0.5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
