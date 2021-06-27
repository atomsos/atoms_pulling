import numpy as np

from ase import Atoms
import ase.io
from ase.build import minimize_rotation_and_translation
from .pulling_optimizer import PullingAtoms, PullingMDMin
from .spherical_optimizer import BFGS_SOPT

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


def pulling(atoms, pair_atoms: Atoms, index=None, init_spring_k=0.2, calculator=None):
    minimize_rotation_and_translation(atoms, pair_atoms)
    start = PullingAtoms(atoms)
    start.set_pair_atoms(pair_atoms, index=index)
    # ase.io.write('trajx.traj', [start, pair_atoms])
    # start.set_calculator(Gromacs(clean=False))
    calculator = calculator or EMT()
    start.set_calculator(calculator)
    converged = False
    pulling_step = 0
    max_pulling_step = 50
    spring_k = init_spring_k
    while not converged and pulling_step < max_pulling_step:
        start.set_spring_k(spring_k)
        pull = PullingMDMin(start, pair_atoms=pair_atoms,
                            pulling_threshold=3, logfile='-',
                            trajectory='traj.traj', append_trajectory=True)
        pull.run()
        # optimize start with spherical opt
        # opt = BFGS_SOPT(start, pair_atoms, index, logfile='-')
        # opt.run()
        dx = start.get_positions() - pair_atoms.get_positions()
        if np.linalg.norm(dx[index, :]) < 0.5:
            converged = True
            print('Converged!', pulling_step)
        pulling_step += 1
        spring_k *= 2


def pulling_gromacs(atoms, pair_atoms,
                    index_filename='index.ndx', index_name='protein_lig'):
    index = get_gromcas_index('index.ndx', 'protein_lig')
    pulling(atoms, pair_atoms, index)
