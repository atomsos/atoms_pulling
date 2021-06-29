import os
import numpy as np

from ase import Atoms
import ase.io
from ase.io.trajectory import Trajectory

# from ase.build import minimize_rotation_and_translation
from .pulling_atoms import PullingAtoms, SubAtoms
from .pulling_optimizer import PullingMDMin, PullingLBFGS
from .spherical_optimizer import SOPT_BFGS, SOPT_LBFGS


from ase.calculators.gromacs import Gromacs
from ase.calculators.emt import EMT
from atoms_pulling.robust import RigidCalculator


def minimize_rotation_and_translation(target, atoms, index=None):
    """Minimize RMSD between atoms and target.

    Rotate and translate atoms to best match target.  For more details, see::

        Melander et al. J. Chem. Theory Comput., 2015, 11,1055
    """
    from ase.build.rotate import rotation_matrix_from_points
    p = atoms.get_positions()
    p0 = target.get_positions()
    if index:
        p = p[index]
        p0 = p0[index]
    # centeroids to origin
    c = np.mean(p, axis=0)
    p -= c
    c0 = np.mean(p0, axis=0)
    p0 -= c0

    # Compute rotation matrix
    R = rotation_matrix_from_points(p.T, p0.T)

    atoms.set_positions(np.dot(atoms.get_positions() - c, R.T) + c0)


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


def pulling(atoms: Atoms, pair_atoms: Atoms, index=None,
            init_spring_k=0.2, pulling_threshold=5,
            use_sopt=False, remove_translate_rotate=True,
            calculator=None):

    import copy
    if remove_translate_rotate:
        minimize_rotation_and_translation(atoms, pair_atoms, index)
    start = PullingAtoms(atoms)
    start.arrays = copy.deepcopy(atoms.arrays)
    index = None
    if index:
        start.set_subatoms_index(index)
        pair_atoms = SubAtoms(pair_atoms)
        pair_atoms.set_subatoms_index(index)
    start.set_pair_atoms(pair_atoms)
    # ase.io.write('trajx.traj', [start, pair_atoms])
    # start.set_calculator(Gromacs(clean=False))
    # pair_atoms.get_forces()

    calculator = calculator or EMT()
    start.calc = calculator
    if isinstance(calculator, (Gromacs, RigidCalculator)):
        calculator.atoms = start

    converged = False
    pulling_step = 0
    max_pulling_step = 100
    spring_k = init_spring_k
    spring_order = 1
    enable_gro = False
    if 'residuenames' in start.arrays:
        enable_gro = True
        os.system('rm -f pull.gro')

    traj = Trajectory('res.traj', mode='w')
    start.set_spring(spring_k, spring_order)
    traj.write(start)
    while not converged and pulling_step < max_pulling_step:
        start.enable_pulling()
        print('start Pulling')

        # pull = PullingMDMin(atoms=start, pair_atoms=pair_atoms,
        #                     pulling_threshold=pulling_threshold, logfile='-',
        #                     trajectory='pull.traj', append_trajectory=True)
        pull = PullingLBFGS(atoms=start, pair_atoms=pair_atoms,
                            pulling_threshold=pulling_threshold, restart=None,
                            logfile='-', trajectory='pull.traj',
                            maxstep=0.05, memory=100, damping=1, alpha=70,
                            use_line_search=False, master=None,
                            force_consistent=None, append_trajectory=True)
        # import pdb; pdb.set_trace()
        pull.run(fmax=0.2, steps=200)

        if enable_gro:
            start.write('.pull.gro')
            os.system('echo >> .pull.gro; cat .pull.gro >> pull.gro; ')

        # optimize start with spherical opt
        if use_sopt:
            start.disable_pulling()
            print('start BFGS_SOPT')
            opt = SOPT_BFGS(atoms=start, anchor=pair_atoms, index=index,
                            logfile='-',
                            trajectory='sopt.traj', append_trajectory=True)
            opt.run(fmax=0.1)

        dx = start.get_positions() - pair_atoms.get_positions()
        if abs(dx).max() < 0.8:
            converged = True
            print('Converged!', pulling_step)

        if not pull.pulling_stop():
            spring_k *= 1.5
            start.set_spring(spring_k, spring_order)

        pulling_step += 1
        print("Write res.traj")
        # import pdb; pdb.set_trace()
        start.disable_pulling()
        traj.write(start)


def pulling_gromacs(atoms, pair_atoms,
                    index_filename='index.ndx', index_name='protein_lig'):
    index = get_gromcas_index('index.ndx', 'protein_lig')
    pulling(atoms, pair_atoms, index)
