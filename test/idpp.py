from ase import Atoms
import ase.io
import ase.build.molecule
from ase.neb import NEB, idpp_interpolate
from ase.calculators.emt import EMT


def main():
    x: Atoms = ase.build.molecule("H2CO")
    y = x.copy()
    y.rotate(170, 'x')
    # y.positions += 5
    # x.positions -= 5
    images = []
    nimages = 10
    for _ in range(nimages):
        new = x.copy()
        new.positions = x.positions + (y.positions - x.positions) / nimages * _
        new.calc = EMT()
        images.append(new)

    ase.io.write('start.traj', images)
    idpp_interpolate(images, traj='idpp.traj' ,steps=100)
    ase.io.write('end.traj', images)

if __name__ == "__main__":
    main()
