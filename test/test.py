import ase.build.molecule as molecule
import ase.io
from atoms_pulling import PullingAtoms, PullingBFGS
from ase.calculators.emt import EMT
from ase.calculators.gromacs import Gromacs
from ase.build import minimize_rotation_and_translation


def main():
    start = ase.io.read('first.gro')
    end = ase.io.read('first.gro')
    minimize_rotation_and_translation(start, end)
    end.positions += 15.0
    start = PullingAtoms(start)
    start.set_pair_atoms(end, 0.1)
    # start.calculator = EMT()
    start.set_calculator(Gromacs(clean=False))
    # end.calculator = EMT()

    opt = PullingBFGS(start, pair_atoms=end, k=0.1,
                      pulling_threshold=3, logfile='-', trajectory='traj.traj')
    opt.run()


if __name__ == '__main__':
    main()
