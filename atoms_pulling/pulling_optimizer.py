from ase import Atoms
from ase.optimize import MDMin
import numpy as np
from numpy.lib.arraysetops import isin


class PullingAtoms(Atoms):

    def set_pair_atoms(self, pair_atoms, k, pair_index=None):
        self.pair_atoms = pair_atoms
        self.k = k
        self.pair_index = pair_index
        assert isinstance(self.pair_atoms, Atoms)

    def get_forces(self, apply_constraint=True, md=False):
        delta_positions = self.pair_atoms.positions - self.positions
        if self.pair_index is not None:
            for i in self.pair_index:
                delta_positions[i,:] = 0
        scale = np.square(np.linalg.norm(delta_positions, axis=1))  # xis 1
        delta_forces = delta_positions * scale[:, np.newaxis]
        delta_forces *= 0.5 * self.k
        return super().get_forces(apply_constraint=apply_constraint, md=md) + delta_forces


class PullingBFGS(MDMin):

    def __init__(self, atoms, pair_atoms, k=0.1, pulling_threshold=1,
                 restart=None, logfile='-', trajectory=None,
                 maxstep=None, master=None, alpha=None):
        # self.atoms = PullingAtoms(atoms, pair_atoms=pair_atoms, k=k)
        assert isinstance(atoms, PullingAtoms)
        self.atoms = atoms
        self.pair_atoms = pair_atoms
        self.k = k
        self.pulling_threshold = pulling_threshold
        self.initial_position = self.atoms.positions.copy()
        super().__init__(atoms, restart, logfile, trajectory)

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
