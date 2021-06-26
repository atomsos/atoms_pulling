import numpy as np
from ase.build import minimize_rotation_and_translation
from .pulling_optimizer import PullingAtoms, PullingBFGS
from ase.calculators.gromacs import Gromacs
from ase.calculators.emt import EMT


def get_gromcas_index(index_file, name):
    content = open(index_file).read()
    idx = content.find('[ ' + name + ' ]\n')
    if idx == -1:
        return None
    content = content[idx:]
    content = content[content.index(']')+1:].strip()
    if '\n[' in content:
        content = content[:content.find('\n[')]
    return list(map(lambda x: int(x) - 1, content.strip().split()))


def pulling(atoms, pair_atoms, index=None, calculator=EMT()):
    minimize_rotation_and_translation(atoms, pair_atoms)
    start = PullingAtoms(atoms)
    start.set_pair_atoms(pair_atoms, 0.1, index=index)
    # start.set_calculator(Gromacs(clean=False))
    start.set_calculator(EMT())
    converged = False
    while not converged:
        pull = PullingBFGS(start, pair_atoms=pair_atoms, k=0.1,
                           pulling_threshold=3, logfile='-', trajectory='traj.traj')
        pull.run()
        # optimize start with spherical opt
        opt = SphericalOptimization(start, pair_atoms, index, logfile='-')
        opt.run()
        dx = start.positions - pair_atoms.positions
        if np.linalg.norm(dx[index, :]) < 0:
            converged = True


def pulling_gromacs(atoms, pair_atoms,
                    index_filename='index.ndx', index_name='protein_lig'):
    index = get_gromcas_index('index.ndx', 'protein_lig')
    pulling(atoms, pair_atoms, index)
