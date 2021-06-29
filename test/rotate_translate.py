import numpy as np

import ase.io
from ase.build.molecule import molecule

from atoms_pulling import PullingAtoms, PullingBFGS
from ase.calculators.emt import EMT
from ase.calculators.gromacs import Gromacs
from ase.build import minimize_rotation_and_translation
from atoms_pulling.pulling import pulling
from atoms_pulling.robust import RigidCalculator


def translate(name='H2CO'):
    start = molecule(name)
    end = molecule(name)
    end.positions += [10, 10, 10]
    calc = RigidCalculator(start)
    ase.io.write('start.traj', [start, end])
    pulling(start, end, pulling_threshold=1.0,
            use_sopt=False, remove_translate_rotate=False,
            calculator=calc)


def rotate(axis='x', name='H2CO'):
    start = molecule(name)
    end = molecule(name)
    end.rotate(180, axis, center=(0, 0, 0))
    ase.io.write('start.traj', [start, end])
    calc = RigidCalculator(start)
    # start.calc = calc
    pulling(start, end, pulling_threshold=1.0,
            use_sopt=False, remove_translate_rotate=False,
            calculator=calc)


def rotatex(name='H2CO'):
    rotate('x', name)


def rotatey(name='H2CO'):
    rotate('y', name)


def rotatez(name='H2CO'):
    rotate('z', name)


def trans_and_rotate(name='H2CO'):
    start = molecule(name)
    end = molecule(name)
    end.rotate(180, 'x', center=(0, 0, 0))
    end.positions += 10
    ase.io.write('start.traj', [start, end])
    calc = RigidCalculator(start)
    # start.calc = calc
    pulling(start, end, pulling_threshold=1.0,
            use_sopt=False, remove_translate_rotate=False,
            calculator=calc)


def mkdir_and_run(dname, function, **kwargs):
    import os
    dirname = os.getcwd()
    os.chdir(dirname)
    os.makedirs(dname)
    os.chdir(dname)
    function(**kwargs)
    os.chdir(dirname)


if __name__ == '__main__':
    mkdir_and_run('translate', translate, name='H2CO')
    mkdir_and_run('rotatex', rotatex, name='H2CO')
    mkdir_and_run('rotatey', rotatey, name='H2CO')
    mkdir_and_run('rotatez', rotatez, name='H2CO')

    mkdir_and_run('rotatex_C60', rotatex, name='C60')
    mkdir_and_run('rotatey_C60', rotatey, name='C60')
    mkdir_and_run('rotatez_C60', rotatez, name='C60')

    mkdir_and_run('trans_and_rotate', trans_and_rotate, name='H2CO')
