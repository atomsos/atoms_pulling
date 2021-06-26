from ase import Atoms
from ase.optimize import BFGS
import numpy as np
from numpy.lib.arraysetops import isin


class PullingAtoms(Atoms):
    def __init__(self, symbols=None,
                 positions=None, numbers=None,
                 tags=None, momenta=None, masses=None,
                 magmoms=None, charges=None,
                 scaled_positions=None,
                 cell=None, pbc=None, celldisp=None,
                 constraint=None,
                 calculator=None,
                 info=None,
                 velocities=None,
                 pair_atoms=None,
                 k=0.1
                 ):
        super().__init__(symbols=symbols, positions=positions,
                         numbers=numbers, tags=tags, momenta=momenta, masses=masses,
                         magmoms=magmoms, charges=charges,
                         scaled_positions=scaled_positions,
                         cell=cell, pbc=pbc, celldisp=celldisp, constraint=constraint,
                         calculator=calculator, info=info, velocities=velocities)
        self.pair_atoms = pair_atoms
        assert isinstance(self.pair_atoms, Atoms)

    def get_forces(self, apply_constraint, md):
        delta_positions = self.pair_atoms.position - self.positions
        scale = np.square(np.linalg.norm(delta_positions, axis=1))  # xis 1
        delta_forces = delta_positions * scale[:, np.newaxis]
        delta_forces *= 0.5 * self.k
        return super().get_forces(apply_constraint=apply_constraint, md=md) + delta_forces


class PullingBFGS(BFGS):

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
        super().__init__(self, atoms, restart, logfile, trajectory, maxstep, master, alpha)

    def pulling_stop(self):
        if np.linalg.norm(self.positions - self.initial_position) < self.pulling_threshold:
            return True
        return False

    def converged(self, forces=None):
        if not super().converged():
            if self.pulling_stop():
                return True
        return False

# d = 0.9575
# t = np.pi / 180 * 104.51
# water = Atoms('H2O',
#               positions=[(d, 0, 0),
#                          (d * np.cos(t), d * np.sin(t), 0),
#                          (0, 0, 0)],
#               calculator=EMT())
# dyn = BFGS(water)
# dyn.run(fmax=0.05)
