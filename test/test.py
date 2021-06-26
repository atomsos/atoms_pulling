from math import log
import ase.build.molecule as molecule
from atoms_pulling import PullingAtoms, PullingBFGS
from ase.calculators.emt import EMT


def main():
    start = molecule('CH3OH')
    end = molecule('CH3OH')
    start = PullingAtoms(start, pair_atoms=end, k=0.1)
    start.calculator = EMT()
    # end.calculator = EMT()

    opt = PullingBFGS(start, end, k=0.1, pulling_threshold=1, logfile='-')
    opt.run()


if __name__ == '__main__':
    main()