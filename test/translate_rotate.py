import numpy as np

import ase.io
from ase.build.molecule import molecule

from atoms_pulling import PullingAtoms, PullingBFGS
from ase.calculators.emt import EMT
from ase.calculators.gromacs import Gromacs
from ase.build import minimize_rotation_and_translation
from atoms_pulling.pulling import pulling


def translate():
    start = molecule('H2CO')
    end = molecule('H2CO')
    end.positions += [10, 10, 10]
    ase.io.write('start.traj', [start, end])
    pulling(start, end, pulling_threshold=1.0,
            use_sopt=True, remove_translate_rotate=False)


def rotate(axis='x'):
    start = molecule('H2CO')
    end = molecule('H2CO')
    end.rotate(180, axis, center=(0, 0, 0))
    ase.io.write('start.traj', [start, end])
    pulling(start, end, pulling_threshold=1.0,
            use_sopt=True, remove_translate_rotate=False)


def rotatex():
    rotate('x')


def rotatey():
    rotate('y')


def rotatez():
    rotate('z')


def mkdir_and_run(name, function):
    import os
    dirname = os.getcwd()
    os.chdir(dirname)
    os.makedirs(name)
    os.chdir(name)
    function()
    os.chdir(dirname)


if __name__ == '__main__':
    mkdir_and_run('translate', translate)
    mkdir_and_run('rotatex', rotatex)
    mkdir_and_run('rotatey', rotatey)
    mkdir_and_run('rotatez', rotatez)
