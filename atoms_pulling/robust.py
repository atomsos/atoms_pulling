import numpy as np
from numpy.linalg import norm
from itertools import combinations, product

from ase import Atoms
from ase.calculators.calculator import Calculator


def compute_dist_X_Y(X, Y):
    """
    compute cross distance matrix between X and Y
    X and Y are 3D arrays, the number may not the same
    result[i][j] means distance between Xi and Yj
    """
    X, Y = np.array(X), np.array(Y)
    assert X.shape[-1] == Y.shape[-1] == 3
    X2 = np.sum(np.square(X), axis=1).reshape((len(X), 1))
    Y2 = np.sum(np.square(Y), axis=1).reshape((len(Y), 1)).T
    C = np.dot(X, Y.T)
    dists = X2 + Y2 - 2*C
    dists = np.sqrt(abs(dists))
    return dists


def compute_distance_matrix(X, cell=None):
    if isinstance(cell, list):
        cell = np.array(cell)
    assert cell is None or \
        hasattr(cell, 'shape') and cell.shape == (3, 3)
    res_dist = compute_dist_X_Y(X, X)
    if cell is not None:
        for i, j, k in product(range(-1, 2), range(-1, 2), range(-1, 2)):
            if i == j == k == 0:
                continue
            Y = X + cell[0] * i + cell[1] * j + cell[2] * k
            Z = compute_dist_X_Y(X, Y)
            dists = np.min([Z, Z.T], axis=0)
            res_dist = np.min([res_dist, dists], axis=0)
    np.fill_diagonal(res_dist, 0)
    return res_dist


def get_distance_matrix(atoms):
    """
    get atoms distance matrix
    Input:
        atoms: Atoms like object
    Output:
        np.ndarray, distance matrix
    """
    cell = atoms.cell.copy()
    pbc = atoms.pbc
    if pbc.any():
        for i in range(3):
            if not pbc[i]:
                cell[i] = (0, 0, 0)
    else:
        cell = None
    dist_matrix = compute_distance_matrix(atoms.get_positions(), cell)
    return dist_matrix


def get_groups(connectivity):
    mol_index = [None] * (len(connectivity))
    mol_group = []
    mol_i = 0
    length = 0
    start = 0
    queue = []
    # import pdb; pdb.set_trace()
    while None in mol_index:
        if len(queue) == 0:
            start = mol_index.index(None)
            mol_index[start] = mol_i
            mol_group.append([start])
            mol_i += 1
        else:
            start = queue.pop()
        for i in range(len(connectivity)):
            if mol_index[i] == None and connectivity[i][start]:
                if not i in queue:
                    queue.append(i)
                mol_group[mol_index[start]].append(i)
                mol_index[i] = mol_index[start]
        # print(mol_index, mol_group, queue)
    return {'mol_index': mol_index, 'mol_group': mol_group}


class RigidCalculator():
    # regatd atoms as rigid parts

    def __init__(self, atoms: Atoms):
        self.atoms = atoms
        self.update_connections()

    def update_connections(self):
        # self._dist_matrix = get_distance_matrix(self.atoms)
        # self._connection = self._dist_matrix < 2.0
        import ase.neighborlist
        self.nl = ase.neighborlist.build_neighbor_list(
            self.atoms, bothways=True)
        self._cutoffs = np.array(ase.neighborlist.natural_cutoffs(self.atoms))
        self._cutoff_matrix = self._cutoffs.reshape(
            (1, -1)) + self._cutoffs.reshape((-1, 1))
        self._dist_matrix = get_distance_matrix(self.atoms)
        self._group = get_groups(self._dist_matrix < self._cutoff_matrix * 1.5)

    def get_potential_energy(self, atoms=None, force_consistent=None):
        return 0.0

    # def get_not_same_group_index(self, index):
    #     return mol_index != mol_index[index]

    def get_mask(self, distances, i):
        """
        1. not same group: only nearby
        2. same group: nearby * 5
        """
        mol_index = np.array(self._group['mol_index'])
        not_same_group = mol_index != mol_index[i]
        thresholds = self._cutoffs + self._cutoffs[i]
        too_far = distances > thresholds
        # too_close2 = distances < thresholds * 3
        return np.logical_and(not_same_group, too_far)

    def _get_rigid_force(self):
        dist_matrix = get_distance_matrix(self.atoms)
        # print(dist_matrix, '\n', self._dist_matrix)
        natoms = len(self.atoms)
        forces = np.zeros((natoms, 3))
        # threshold = 2.5
        positions = self.atoms.get_positions()
        for i in range(natoms):
            _dist: np.ndarray = dist_matrix[i] - self._dist_matrix[i]
            # if _dist.max() > 2:
            # not_same_group_index = self.get_not_same_group_index(i)
            # too_close = dist_matrix[i]
            mask = self.get_mask(dist_matrix[i], i)
            # import pdb; pdb.set_trace()
            _dist[mask] = 0
            f0 = (np.exp(abs(_dist) * 10) - 1.0) * np.sign(_dist)
            f = f0.dot(positions - positions[i])
            # print('\n', _dist, thresholds, f0, f)
            forces[i] = f
        return forces

    def get_forces(self, apply_constraint=True, md=False):
        rigid_force = self._get_rigid_force()
        return rigid_force

    def reduce_force(self):
        pass


if __name__ == "__main__":
    import os
    import ase.io
    from ase.build import molecule
    from ase.optimize.lbfgs import LBFGS
    from ase.neb import NEB
    atoms = molecule('H2O')

    os.remove('traj.traj')
    ase.io.write('traj.traj', atoms)

    atoms.calc = RigidCalculator(atoms)
    print(atoms.get_forces())
    atoms.positions += 100
    print(atoms.get_forces())
    atoms.positions -= 100
    atoms.positions[0][0] += 1
    print(atoms.get_forces())

    opt = LBFGS(atoms, trajectory='traj.traj', append_trajectory=True)
    opt.run()

    # neb = NEB(images)
