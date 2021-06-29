import numpy as np

import ase.io
from ase.build.molecule import molecule

from atoms_pulling import PullingAtoms, PullingBFGS
from ase.calculators.emt import EMT
from ase.calculators.gromacs import Gromacs
from ase.build import minimize_rotation_and_translation
from atoms_pulling.pulling import pulling

from atoms_pulling.robust import RigidCalculator


def main():
    start = ase.io.read('first.gro')
    end = ase.io.read('first.gro')
    minimize_rotation_and_translation(start, end)
    end.positions += 15.0
    start = PullingAtoms(start)
    start.set_pair_atoms(end)
    start.set_spring(k=0.1, order=1)
    # start.calculator = EMT()
    start.set_calculator(Gromacs(clean=False))
    # end.calculator = EMT()

    opt = PullingBFGS(start, pair_atoms=end,
                      pulling_threshold=3, logfile='-', trajectory='traj.traj')
    opt.run()


def main2():
    start = ase.io.read('hcoh.xyz')
    end = molecule('H2CO')
    pulling(start, end)


def main3():
    start = ase.io.read('h2+co.xyz')
    end = molecule('H2CO')
    pulling(start, end)


def main_gro():
    from atoms_pulling.gromacs_index import get_index
    start = ase.io.read('first.gro')
    end = ase.io.read('first.gro')

    calc = Gromacs(clean=False)
    calc.params_runs['extra_mdrun_parameters'] = '-nt 1'
    calc.params_runs['extra_grompp_parameters'] = '-n index.ndx'
    # calc.atoms = start

    index = get_index('index.ndx', 'protein_lig')

    pulling(start, end, index=index, calculator=calc)


def get_forces():
    start = ase.io.read('first.gro')
    calc = Gromacs(clean=False)
    calc.params_runs['extra_mdrun_parameters'] = '-nt 1'
    calc.params_runs['extra_grompp_parameters'] = '-n index.ndx'
    start.calc = calc
    # start.set_calculator(calc)
    calc.atoms = start
    forces = start.get_forces()
    import pdb
    pdb.set_trace()


def main_6mrc():
    from atoms_pulling.gromacs_index import get_index
    start = ase.io.read('first.gro')
    end = ase.io.read('first.gro')
    # import pdb; pdb.set_trace()

    # move UFF out
    uff_index = np.where(end.arrays['residuenames'] == 'UFF')
    end.positions[uff_index] += [50.0, 0, 0]  # increment 50 Ang in x axis
    end.write('end.gro')

    calc = Gromacs(clean=False)
    calc.params_runs['extra_mdrun_parameters'] = '-nt 1'
    calc.params_runs['extra_grompp_parameters'] = '-n index.ndx'

    # calc = RigidCalculator(start)
    # calc.atoms = start

    index = get_index('index.ndx', 'protein_lig')

    pulling(start, end, index=index, calculator=calc)


if __name__ == '__main__':
    # main3()
    main_6mrc()
    # get_forces()
