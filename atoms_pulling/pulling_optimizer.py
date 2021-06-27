import numpy as np

from ase import Atoms
from ase.optimize import MDMin, BFGS, LBFGS


class PullingAtoms(Atoms):

    def set_pair_atoms(self, pair_atoms, index=None):
        self.pair_atoms = pair_atoms
        self.pair_index = index
        assert isinstance(self.pair_atoms, Atoms)

    def set_spring_k(self, k):
        self.spring_k = k

    def get_forces(self, apply_constraint=True, md=False):
        delta_positions = self.pair_atoms.positions - self.positions
        if self.pair_index is not None:
            for i in self.pair_index:
                delta_positions[i, :] = 0
        scale = np.square(np.linalg.norm(delta_positions, axis=1))  # xis 1
        delta_forces = delta_positions * scale[:, np.newaxis]
        delta_forces *= 0.5 * self.spring_k
        return super().get_forces(apply_constraint=apply_constraint, md=md) + delta_forces


class PullingOptimizer():
    def __init__(self, atoms, pair_atoms, pulling_threshold=1):
        assert isinstance(atoms, PullingAtoms)
        self.atoms = atoms
        self.pair_atoms = pair_atoms
        self.pulling_threshold = pulling_threshold
        self.initial_position = self.atoms.positions.copy()


class PullingBFGS(BFGS):

    def __init__(self, atoms, pair_atoms, pulling_threshold=1,
                 restart=None, logfile='-', trajectory=None,
                 maxstep=None, master=None, alpha=None):
        # self.atoms = PullingAtoms(atoms, pair_atoms=pair_atoms, k=k)
        assert isinstance(atoms, PullingAtoms)
        self.atoms = atoms
        self.pair_atoms = pair_atoms
        self.pulling_threshold = pulling_threshold
        self.initial_position = self.atoms.positions.copy()
        super().__init__(atoms, restart, logfile, trajectory, maxstep, master, alpha)

    def pulling_stop(self):
        if np.linalg.norm(self.atoms.positions - self.initial_position) > self.pulling_threshold:
            print("Pulling Stop")
            return True
        return False

    def converged(self, forces=None):
        return super().converged(forces=forces) or self.pulling_stop()


class PullingLBFGS(LBFGS):

    def __init__(self, atoms, pair_atoms, pulling_threshold=1,
                 restart=None, logfile='-', trajectory=None,
                 maxstep=None, memory=100, damping=1.0, alpha=70.0,
                 use_line_search=False, master=None,
                 force_consistent=None):

        # self.atoms = PullingAtoms(atoms, pair_atoms=pair_atoms, k=k)
        assert isinstance(atoms, PullingAtoms)
        self.atoms = atoms
        self.pair_atoms = pair_atoms
        self.pulling_threshold = pulling_threshold
        self.initial_position = self.atoms.positions.copy()
        super().__init__(atoms, restart, logfile, trajectory,
                         maxstep, memory, damping, alpha,
                         use_line_search, master,
                         force_consistent)

    def pulling_stop(self):
        if np.linalg.norm(self.atoms.positions - self.initial_position) > self.pulling_threshold:
            print("Pulling Stop")
            return True
        return False

    def converged(self, forces=None):
        return super().converged(forces=forces) or self.pulling_stop()


class PullingMDMin(MDMin):

    def __init__(self, atoms, pair_atoms, pulling_threshold=1,
                 restart=None, logfile='-', trajectory=None,
                 dt=None, master=None,
                 append_trajectory=False, force_consistent=False,
                 ):
        # self.atoms = PullingAtoms(atoms, pair_atoms=pair_atoms, k=k)
        assert isinstance(atoms, PullingAtoms)
        self.atoms = atoms
        self.pair_atoms = pair_atoms
        self.pulling_threshold = pulling_threshold
        self.initial_position = self.atoms.positions.copy()
        super().__init__(atoms, restart, logfile, trajectory,
                         dt, master, append_trajectory, force_consistent)

    def pulling_stop(self):
        if np.linalg.norm(self.atoms.positions - self.initial_position) > self.pulling_threshold:
            print("Pulling Stop")
            return True
        return False

    def converged(self, forces=None):
        return super().converged(forces=forces) or self.pulling_stop()

# d = 0.9575
# t = np.pi / 180 * 104.51
# water = Atoms('H2O',
#               positions=[(d, 0, 0),
#                          (d * np.cos(t), d * np.sin(t), 0),
#                          (0, 0, 0)],
#               calculator=EMT())
# dyn = BFGS(water)
# dyn.run(fmax=0.05)
