import numpy as np
from numpy.lib.arraysetops import isin
from numpy.linalg import norm
from itertools import combinations, product

from ase import Atoms
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.optimize.optimize import Optimizer
from ase.calculators.calculator import Calculator

from numba import njit


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


def get_connection_matrix(atoms, sparse=False, multi=2.0):
    import ase.neighborlist
    _cutoffs = np.array(
        ase.neighborlist.natural_cutoffs(atoms, mult=multi))
    nl = ase.neighborlist.build_neighbor_list(
        atoms, cutoffs=_cutoffs,  bothways=True, self_interaction=True)
    # connection_matrix = _dist_matrix < _cutoff_matrix
    matrix = nl.get_connectivity_matrix(sparse=sparse)
    return matrix


if __name__ == '__main__':
    import ase.build
    x = ase.build.molecule("H2O")
    print(get_connection_matrix(x))


class RobustCalculator():
    # regatd atoms as rigid parts

    def __init__(self, atoms: Atoms):
        self.atoms = atoms
        self.cutoff_multiply = 1.5
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
        self._group = get_groups(
            self._dist_matrix < self._cutoff_matrix * 1.45)

    def get_potential_energy(self, atoms=None, force_consistent=None):
        return 0.0

    # def get_not_same_group_index(self, index):
    #     return mol_index != mol_index[index]

    # @njit()
    def get_mask(self, distances, i, multi=3):
        """
        1. not same group: only nearby
        2. same group: nearby * 5
        """
        mol_index = np.array(self._group['mol_index'])
        not_same_group = mol_index != mol_index[i]
        thresholds = self._cutoffs + self._cutoffs[i]
        too_far = distances > thresholds * self.cutoff_multiply
        # too_close2 = distances < thresholds * 3
        return np.logical_and(not_same_group, too_far)

    def _get_rigid_force(self):
        dist_matrix = get_distance_matrix(self.atoms)
        # print(dist_matrix, '\n', self._dist_matrix)
        natoms = len(self.atoms)
        forces = np.zeros((natoms, 3))
        # threshold = 2.5
        positions = self.atoms.get_positions()
        # import pdb; pdb.set_trace()
        for i in range(natoms):
            _dist: np.ndarray = dist_matrix[i] - self._dist_matrix[i]
            # if _dist.max() > 2:
            # not_same_group_index = self.get_not_same_group_index(i)
            # too_close = dist_matrix[i]
            mask = self.get_mask(dist_matrix[i], i)
            _dist[mask] = 0
            f0 = (np.exp(abs(_dist)) - 1.0) * np.sign(_dist)
            dp = positions - positions[i]
            dp[i] = [0, 0, 1]
            dp /= np.linalg.norm(dp, axis=1)[:, np.newaxis]
            f = f0.dot(dp)
            # print('\n', _dist, thresholds, f0, f)
            forces[i] = f
        # if (abs(forces).max() > 1e10):
        #     self.cutoff_multiply = min(self.cutoff_multiply-0.1, 1.5)
        return forces

    def get_forces(self, apply_constraint=True, md=False):
        rigid_force = self._get_rigid_force()
        return rigid_force

    def reduce_force(self):
        mol_index = np.array(self._group['mol_index'].copy())
        mol_group = self._group['mol_group'].copy()
        for group in mol_group:
            pass


if __name__ == "__main__":
    import os
    import ase.io
    from ase.build import molecule
    from ase.optimize.lbfgs import LBFGS
    from ase.neb import NEB
    atoms = molecule('H2O')

    if os.path.exists('traj.traj'):
        os.remove('traj.traj')
    ase.io.write('traj.traj', atoms)

    atoms.calc = RobustCalculator(atoms)
    print(atoms.get_forces())
    atoms.positions += 100
    print(atoms.get_forces())
    atoms.positions -= 100
    atoms.positions[0][0] += 1
    print(atoms.get_forces())

    opt = LBFGS(atoms, trajectory='traj.traj', append_trajectory=True)
    opt.run()

    # neb = NEB(images)


class RigidOptimizer():

    def __init__(self, atoms: Atoms, pair_atoms: Atoms,
                 trajectory=None, logfile='-',
                 refinement=False):
        self.atoms = atoms
        self.pair_atoms = pair_atoms
        self.refinement = refinement
        if refinement:
            self.cutoff_multiply = 1.5
        else:
            self.cutoff_multiply = 2.0
        if trajectory is not None:
            if isinstance(trajectory, str):
                trajectory = Trajectory(trajectory, 'w')
            assert isinstance(
                trajectory, TrajectoryWriter), 'trajectory must be a Trajectory object' + type(trajectory)
        self.trajectory = trajectory
        if isinstance(logfile, str):
            if logfile == '-':
                import sys
                logfile = sys.stdout
            elif logfile == '/dev/null':
                logfile = None
            else:
                logfile = open(logfile, 'w')
        assert logfile is None or hasattr(logfile, 'write')
        self.logfile = logfile
        self.update_connections()

    def update_connections(self):
        connection_matrix = get_connection_matrix(
            self.atoms, multi=self.cutoff_multiply)
        self._group = get_groups(connection_matrix)

    def rigid_rotate_scan(self, group_index):
        atoms = origin_atoms = self.atoms.copy()[group_index]
        pair_positions = self.pair_atoms.get_positions()[group_index]
        # atoms.calc = self.__class__(atoms)
        mass_center = atoms.get_center_of_mass()
        initial_position = atoms.get_positions()
        min_position = None
        min_dpos = np.linalg.norm(atoms.positions - pair_positions)

        if self.refinement:
            phi_max, theta_max, psi_max = [20, 20, 20]
            phi_min, theta_min, psi_min = [-20, -20, -20]
            seg = 5
        else:
            phi_max, theta_max, psi_max = [180, 180, 180]
            phi_min, theta_min, psi_min = [-180, -180, -180]
            seg = 20

        for phi in np.linspace(phi_min, phi_max, int((phi_max - phi_min) / seg) + 1):
            for theta in np.linspace(theta_min, theta_max, int((theta_max - theta_min) / seg) + 1):
                for psi in np.linspace(psi_min, psi_max, int((psi_max - psi_min) / seg) + 1):
                    atoms.set_positions(initial_position.copy())
                    # atoms.rotate
                    atoms.euler_rotate(phi, theta, psi, mass_center)
                    new_pos = atoms.get_positions()
                    if self.collapse(new_pos, group_index, only_dist=not self.refinement):
                        # self.logfile.write('collapsed')
                        continue
                    dpos = np.linalg.norm(atoms.positions - pair_positions)
                    if dpos < min_dpos:
                        min_dpos = dpos
                        min_position = atoms.get_positions()
                        self.logfile.write(
                            f'rotate, min_dpos: {min_dpos}, {phi}, {theta}, {psi}\n')
        if min_position is not None:
            self.atoms.positions[group_index] = min_position
        else:
            if self.logfile:
                self.logfile.write("not opted\n")

    def _rigid_translate_opt_scan(self, group_index):
        atoms = self.atoms.copy()
        pair_atoms = self.pair_atoms.copy()
        atoms = atoms[group_index]
        pair_atoms = pair_atoms[group_index]
        # atoms.calc = self.__class__(atoms)
        positions = atoms.get_positions()
        pair_positions = pair_atoms.get_positions()
        mass_center = atoms.get_center_of_mass()
        pair_center = pair_atoms.get_center_of_mass()
        v_v = mass_center - pair_center
        vdist = np.linalg.norm(v_v)
        if abs(vdist) < 1:
            return True
        v_v /= vdist

        # calculate horizontal vector and min/max
        dpos = positions - pair_center
        v_hx = dpos.dot(v_v)
        v_hy = [0, 0, 0]
        segment = 0.5
        hx_min, hx_max = 0, 0
        hy_min, hy_max = 0, 0

        # scan over all space
        for v in range(0, 2 * vdist, segment):
            for hx in range(hx_min, hx_max, segment):
                for hy in range(hy_min, hy_max, segment):
                    new_pos = mass_center + v * v_v + hx * v_hx + hy * v_hy
                    atoms.set_positions(new_pos)
                    dpos = np.linalg.norm(atoms.positions - pair_positions)
                    if dpos < min_dpos:
                        min_dpos = dpos
                        min_position = atoms.get_positions()
        self.atoms.positions[group_index] = min_position

    def get_atoms_by_group_index(self, group_index):
        atoms, pair_atoms = self.atoms.copy(), self.pair_atoms.copy()
        return atoms[group_index], pair_atoms[group_index]

    def collapse(self, new_pos, group_index, only_dist=False):
        # import pdb; pdb.set_trace()
        natoms = len(self.atoms)
        non_group_index = np.arange(
            natoms)[~np.isin(np.arange(natoms), group_index)]

        other_pos = self.atoms.positions[non_group_index]
        # for p in new_pos:
        #     if (np.linalg.norm(other_pos - p, axis=1) < 1.5).any():
        #         return True
        # https://sparrow.dev/pairwise-distance-in-numpy/
        dist: np.ndarray = np.linalg.norm(
            other_pos[:, None, :] - new_pos[None, :, :], axis=-1)
        if (dist < 2.0).any():
            return True
        elif only_dist:
            return False
        # return False

        new_atoms = self.atoms.copy()
        new_atoms.positions[group_index] = new_pos
        new_connection = get_connection_matrix(
            new_atoms, multi=self.cutoff_multiply)
        if new_connection[group_index][:, non_group_index].any():
            return True
        return False

    def rigid_translate_opt_scan(self, group_index):
        init_pos = self.atoms.get_positions()[group_index]
        pair_pos = self.pair_atoms.get_positions()[group_index]
        _pos = [self.atoms.positions - init_pos.mean(axis=0),
                self.pair_atoms.positions - init_pos.mean(axis=0)]
        mins = np.min(_pos, axis=(0, 1)) - 2.0
        maxs = np.max(_pos, axis=(0, 1)) + 2.0
        seg = 1

        if self.refinement:
            mins = [-1, -1, -1]
            maxs = [1, 1, 1]
            seg = 0.4

        minx, miny, minz = mins
        maxx, maxy, maxz = maxs

        best_positions = None
        best_norm = np.linalg.norm(init_pos - pair_pos)
        for z in np.linspace(minz, maxz, 1 + int((maxz - minz) / seg)):
            for y in np.linspace(miny, maxy, 1 + int((maxy - miny) / seg)):
                for x in np.linspace(minx, maxx, 1 + int((maxx - minx) / seg)):
                    # self.logfile.write(x, y, z)
                    dpos = [x, y, z]
                    # if z > 12:
                    #     import pdb; pdb.set_trace()
                    new_pos = dpos + init_pos
                    if self.collapse(new_pos, group_index, only_dist=not self.refinement):
                        # self.logfile.write('collapsed')
                        continue
                    dpos_norm = np.linalg.norm(new_pos - pair_pos)
                    if dpos_norm < best_norm:
                        best_positions = new_pos
                        best_norm = dpos_norm
                        self.logfile.write(
                            f'translate min_norm: {best_norm}\n')
                    # else:
                    #     self.logfile.write('best_norm', best_norm, dpos_norm)
        if best_positions is not None:
            self.atoms.positions[group_index] = best_positions
        else:
            self.logfile.write("not opted\n")

    def rigid_optimization(self):
        mol_group: list = self._group['mol_group'].copy()
        mol_group.sort(key=lambda x: len(x))
        traj = self.trajectory
        log = self.logfile
        for _ in range(2):
            if log is not None:
                log.write(f" * Rigid opt time {_}\n")
            for i, group in enumerate(mol_group):
                if log:
                    log.write(f" * Opt group {i}\n")
                self.rigid_translate_opt_scan(group)
                # if not self.refinement:
                self.rigid_rotate_scan(group)
                # self.rigid_traasenslate_opt(group)
                if log:
                    log.flush()
            if traj is not None:
                traj.write(self.atoms)

        if traj is not None:
            traj.close()
        return self.atoms


if __name__ == '__main__':
    import ase.build
    h2o = ase.build.molecule("H2O")
    newh2o = h2o.copy()
    newh2o.positions += 5
    h2o += newh2o
    start = h2o.copy()
    end = h2o.copy()
    end.positions[[3, 4, 5]] += 5

    opt = RigidOptimizer(start, end)
    assert opt.collapse(newh2o.positions - 5, [3, 4, 5]), 'not collapsed'
    assert not opt.collapse(newh2o.positions + 5, [3, 4, 5]), 'collapsed'
    opt.rigid_translate_opt_scan([3, 4, 5])
    # opt.atoms.write('atoms.traj')

    start = ase.io.read('../test/TiO2-trans_rotate/TiO2_H2.cif')
    end = ase.io.read('../test/TiO2-trans_rotate/TiO2_H-H.cif')
    group = [len(start) - 2, len(start) - 1]

    opt = RigidOptimizer(start.copy(), end.copy())
    # import pdb; pdb.set_trace()
    opt.rigid_translate_opt_scan(group)
    ase.io.write('translate.traj', [start, end, opt.atoms])

    start = ase.io.read('../test/TiO2-trans_rotate/TiO2_H2.cif')
    end = ase.io.read('../test/TiO2-trans_rotate/TiO2_H-H.cif')
    group = [len(start) - 2, len(start) - 1]

    opt = RigidOptimizer(start.copy(), end.copy())
    opt.rigid_rotate_scan(group)
    ase.io.write('rotate.traj', [start, end, opt.atoms])

    # Opt

    start = ase.io.read('../test/TiO2-trans_rotate/TiO2_H2.cif')
    end = ase.io.read('../test/TiO2-trans_rotate/TiO2_H-H.cif')
    group = [len(start) - 2, len(start) - 1]

    opt = RigidOptimizer(start.copy(), end.copy())
    opt.rigid_optimization()
    ase.io.write('optim.traj', [start, end, opt.atoms])
