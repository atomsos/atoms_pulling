import ase.build.molecule
from atoms_pulling.pulling_atoms import SubAtoms
from ase.calculators.emt import EMT


def main():
    mol = ase.build.molecule('H2CO')
    mol.calc = EMT()

    submol = SubAtoms(mol)
    submol.set_subatoms_index([0, 1, 2])
    print(submol.positions.shape, submol.get_positions().shape)
    submol.write('H2CO.xyz')
    print(submol.get_forces().shape)


if __name__ == '__main__':
    main()
