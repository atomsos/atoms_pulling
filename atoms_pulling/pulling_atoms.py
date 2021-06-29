import numpy as np

from ase.atoms import Atoms


class SubAtoms(Atoms):

    def set_subatoms_index(self, index):
        self.subatoms_index = index

    def get_subatoms_index(self):
        return getattr(self, 'subatoms_index', None)

    def get_symbol(self):
        index = getattr(self, 'subatoms_index', None)
        if index is None:
            return super().get_symbol()
        return self.symbols[index].copy()

    def get_positions(self):
        index = getattr(self, 'subatoms_index', None)
        if index is None:
            return super().get_positions()
        return self.positions[index].copy()

    def get_forces(self, apply_constraint=True, md=False):
        index = getattr(self, 'subatoms_index', None)
        if index is None:
            return super().get_forces(apply_constraint, md)
        return super().get_forces(apply_constraint, md)[index].copy()

    def set_positions(self, positions):
        if self.calc and hasattr(self.calc, 'reset'):
            self.calc.reset()
            self.calc.atoms = self
        index = getattr(self, 'subatoms_index', None)
        if index is None:
            super().set_positions(positions)
        else:
            if np.array(positions).shape[0] == len(index):
                self.positions[index] = positions
            else:
                super().set_positions(positions)

    def write(self, filename, format=None, **kwargs):
        index = self.get_subatoms_index()
        self.set_subatoms_index(None)
        super().write(filename, format, **kwargs)
        self.set_subatoms_index(index)

    def __len__(self):
        if self.get_subatoms_index():
            return len(self.get_subatoms_index())
        return super().__len__()


class PullingAtoms(SubAtoms, Atoms):

    def set_pair_atoms(self, pair_atoms: Atoms):
        self.pair_atoms = pair_atoms
        self.pulling = True
        assert isinstance(self.pair_atoms, Atoms)

    def set_spring(self, k, order=1.0, adaptive_spring=False):
        self.spring_k = k
        self.spring_order = order
        self.adaptive_spring = adaptive_spring

    def enable_pulling(self):
        self.pulling = True

    def disable_pulling(self):
        self.pulling = False

    def _get_vertical_forces(self, forces):
        rand = np.random.rand(len(self), 3)
        # vertical
        rand[:, 2] = -np.divide(rand[:, 0] * forces[:, 0] +
                                rand[:, 1] * forces[:, 1], forces[:, 2],
                                out=np.zeros_like(forces[:, 2]),
                                where=forces[:, 2] != 0.)

        # rand force scale should be 0.1 of forces
        scale_f = np.hstack(
            [np.linalg.norm(forces, axis=1).flatten()[:, np.newaxis]] * 3)
        scale_r = np.hstack(
            [np.linalg.norm(rand, axis=1).flatten()[:, np.newaxis]] * 3)
        # rand = rand / scale_r * scale_f
        rand = scale_f * \
            np.divide(rand, scale_r, out=np.zeros_like(
                scale_r), where=scale_r != 0)
        rand *= 0.5
        return rand

    def _get_spring_forces(self, origin_force=None):
        d_positions = self.pair_atoms.get_positions() - self.get_positions()
        # if self.pair_index is not None:
        #     for i in self.pair_index:
        #         d_positions[i, :] = 0
        scale = np.linalg.norm(d_positions, axis=1)  # axis 1
        scale = scale ** self.spring_order
        spring_forces = d_positions * scale[:, np.newaxis]
        if origin_force is not None and self.adaptive_spring:
            scale_origin_f = np.linalg.norm(origin_force, axis=1)
            scale_spring_f = np.linalg.norm(spring_forces, axis=1)
            spring_forces *= (scale_origin_f.max() / scale_spring_f.max())
            spring_forces /= 2
        else:
            spring_forces *= 0.5 * self.spring_k

        use_verticle = True
        if not use_verticle:
            return spring_forces
        else:
            vert_rand_force = self._get_vertical_forces(spring_forces)
            return spring_forces + vert_rand_force

    def _get_forces(self, apply_constraint=True, md=False):
        raw_forces = super().get_forces(apply_constraint=apply_constraint, md=md)
        if not self.pulling:
            return raw_forces
        else:
            spring_forces = self._get_spring_forces(raw_forces)
            return raw_forces + spring_forces

    def get_forces(self, apply_constraint=True, md=False):
        forces = self._get_forces(apply_constraint, md)
        index = getattr(self, 'subatoms_index', None)
        if index is None:
            return forces
        return forces[index].copy()
