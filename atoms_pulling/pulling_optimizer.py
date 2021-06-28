import numpy as np

from ase import Atoms
from ase.optimize import MDMin, BFGS, LBFGS
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG

from .pulling_atoms import PullingAtoms


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
                 force_consistent=None, append_trajectory=False):

        # self.atoms = PullingAtoms(atoms, pair_atoms=pair_atoms, k=k)
        assert isinstance(atoms, PullingAtoms)
        self.atoms = atoms
        self.pair_atoms = pair_atoms
        self.pulling_threshold = pulling_threshold
        self.initial_position = self.atoms.positions.copy()

        super().__init__(atoms=atoms, restart=restart, logfile=logfile, trajectory=trajectory,
                         maxstep=maxstep, memory=memory, damping=damping, alpha=alpha,
                         use_line_search=use_line_search, master=master,
                         force_consistent=force_consistent, append_trajectory=append_trajectory)

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
        import pdb
        pdb.set_trace()
        if np.linalg.norm(self.atoms.positions - self.initial_position) > self.pulling_threshold:
            print("Pulling Stop")
            return True
        return False

    def converged(self, forces=None):
        # import pdb; pdb.set_trace()
        return super().converged(forces=forces) or self.pulling_stop()


class PullingCG(SciPyFminCG):

    def __init__(self, atoms, pair_atoms, pulling_threshold=1,
                 logfile='-', trajectory=None,
                 callback_always=False, alpha=70.0, master=None,
                 append_trajectory=False, force_consistent=None):
        # self.atoms = PullingAtoms(atoms, pair_atoms=pair_atoms, k=k)
        assert isinstance(atoms, PullingAtoms)
        self.atoms = atoms
        self.pair_atoms = pair_atoms
        self.pulling_threshold = pulling_threshold
        self.initial_position = self.atoms.positions.copy()

        super().__init__(atoms, logfile=logfile, trajectory=trajectory,
                         callback_always=callback_always, alpha=alpha, master=master,
                         append_trajectory=append_trajectory,
                         force_consistent=force_consistent)

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
