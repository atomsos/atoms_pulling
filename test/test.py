from ase.build.molecule import molecule
import ase.io
from atoms_pulling import PullingAtoms, PullingBFGS
from ase.calculators.emt import EMT
from ase.calculators.gromacs import Gromacs
from ase.build import minimize_rotation_and_translation
from atoms_pulling.pulling import pulling


def main():
    start = ase.io.read('first.gro')
    end = ase.io.read('first.gro')
    minimize_rotation_and_translation(start, end)
    end.positions += 15.0
    start = PullingAtoms(start)
    start.set_pair_atoms(end)
    start.set_spring_k(0.1)
    # start.calculator = EMT()
    start.set_calculator(Gromacs(clean=False))
    # end.calculator = EMT()

    opt = PullingBFGS(start, pair_atoms=end,
                      pulling_threshold=3, logfile='-', trajectory='traj.traj')
    opt.run()


def main2():
    start = ase.io.read('h2cox.xyz')
    end = molecule('H2CO')
    pulling(start, end)


if __name__ == '__main__':
    main2()
