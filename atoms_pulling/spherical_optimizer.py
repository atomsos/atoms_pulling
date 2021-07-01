# -*- coding: utf-8 -*-
import warnings

import numpy as np
from numpy.linalg import eigh

from ase.atoms import Atoms
from ase.optimize.optimize import Optimizer
from ase.utils import basestring
from ase.optimize import MDMin, BFGS, LBFGS


class CriticalPointError(Exception):
    pass


def normalized(v):
    n = np.linalg.norm(v)
    if n < 1e-5:
        return v
    return v/n


class SphericalBFGS(BFGS):
    def __init__(self, atoms: Atoms, anchor: Atoms, restart=None, logfile='-', trajectory=None,
                 maxstep=0.04, master=None, alpha=None):
        self.anchor = anchor
        super().__init__(atoms, restart, logfile,
                         trajectory, maxstep, master, alpha)


class SOPT_BFGS(Optimizer):
    def __init__(self, atoms: Atoms, anchor: Atoms, index=None, restart=None, logfile='-', trajectory=None,
                 maxstep=0.04, master=None, append_trajectory=False,
                 force_consistent=False,):
        """Modified BFGS optimizer for Spherical Optimization.

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        anchor: Atoms object
            The Atoms pairing with atoms as an anchor

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.04 Å).

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        debug: boolean
            debug option
        """
        if maxstep > 1.0:
            warnings.warn('You are using a much too large value for '
                          'the maximum step size: %.1f Å' % maxstep)
        assert anchor is not None, \
            'Partner is needed when Using Spherical Optimization'
        self.maxstep = maxstep
        self.anchor = anchor
        self.index = index
        self.debug = False

        Optimizer.__init__(self, atoms, restart, logfile,
                           trajectory, master,
                           append_trajectory, force_consistent)

    def todict(self):
        d = Optimizer.todict(self)
        if hasattr(self, 'maxstep'):
            d.update(maxstep=self.maxstep)
        return d

    def initialize(self):
        self.H = None
        self.r0 = None
        self.f0 = None
        if hasattr(self.anchor, 'get_positions'):
            self.r_anchor = self.anchor.get_positions()
        else:
            assert np.array(self.anchor).shape == (len(self.atoms), 3)
            self.r_anchor = np.array(self.anchor)
        self.R2 = np.square(self.atoms.get_positions() - self.r_anchor).sum()
        self.index = np.argmax(
            np.fabs(self.atoms.get_positions() - self.r_anchor).ravel())

    def read(self):
        self.H, self.r0, self.f0, self.maxstep = self.load()

    def step(self, f):
        atoms = self.atoms
        r_real = atoms.get_positions()
        r = r_real.ravel()
        force_real = f.ravel()
        # get new force by projecting radial part
        r_anchor = self.r_anchor.ravel()
        r_v = normalized(r - r_anchor)
        force_mod = (force_real - np.dot(force_real, r_v)
                     * r_v).reshape((-1, 3))
        # determine index of the removed one
        index = self.index
        # if True:
        #     import pdb
        #     pdb.set_trace()

        def rm_index(r, index=0):
            return np.delete(r, index)

        def add_index(r, index=0, val=0):
            return np.insert(r, index, val)

        def get_real_positions(r_mod, r_anchor, R2, index, sign=1):
            """
            r_mod: 3n-1 vector, modified positions
            r_anchor: 3n vector, positions of anchor
            """
            dr_mod = r_mod - rm_index(r_anchor, index)
            if R2 - np.square(dr_mod).sum() < 0:
                raise ValueError('dr too large')
            val = np.sqrt(R2 - np.square(dr_mod).sum()) * \
                sign + r_anchor[index]
            return add_index(r_mod, index, val)

        def gradF(r, r_anchor, R2, index, sign=1):
            dri = r[index] - r_anchor[index]
            dr_all = rm_index(r-r_anchor, index)
            if abs(dri) < 1e-5:
                raise CriticalPointError('dri too small')
            gradf = -1 * dr_all / dri
            return gradf

        r_mod = rm_index(r, index)
        sign = 1
        if abs(get_real_positions(r_mod, r_anchor, self.R2, index)[index] - r[index]) > 1e-5:
            sign = -1

        gradf = gradF(r, r_anchor, self.R2, index, sign)
        f = rm_index(force_real, index) + force_real[index] * gradf
        r0 = self.r0
        if r0 is not None:
            r0 = rm_index(self.r0, index)
        self.update(r_mod, f, r0, self.f0)
        omega, V = eigh(self.H)
        dr = np.dot(V, np.dot(f, V) / np.fabs(omega))

        # dr_value = np.dot(gradf, dr)
        # dr = add_index(dr, index, dr_value)
        # index_pos = r[index] + dr_value
        # while abs(index_pos - get_real_positions(r+dr, r_anchor, self.R2, index, sign)) > 1e-5:
        #     dr_value = get_real_positions(r+dr, r_anchor, self.R2, index, sign) - r[index]
        #     dr[index] = dr_value
        #     # steplengths = (dr.reshape((-1,3))**2).sum(1)**0.5
        #     steplengths = np.linalg.norm(dr.reshape((-1,3), axis=0)
        #     dr = self.determine_step(dr.reshape((-1,3)), steplengths).ravel()
        #     newr = r + dr
        #     index_pos = newr[index]
        # xdr =
        dr = self.determine_step(dr, abs(dr))
        #print("r, r_mod, r0, dr", r, r_mod, r0, dr)
        unavailable = True
        while unavailable:
            try:
                new_r = get_real_positions(
                    r_mod+dr, r_anchor, self.R2, index, sign)
                unavailable = False
            except ValueError:
                dr /= 2.
        # atoms.set_positions(r_real + dr.reshape((-1,3)))
        atoms.set_positions(new_r.reshape((-1, 3)))
        newR2 = np.square(self.atoms.get_positions() - self.r_anchor).sum()
        if abs(newR2 - self.R2) > 0.1:
            raise ValueError()
        self.r0 = r.copy()
        self.f0 = f.copy()
        self.dump((self.H, self.r0, self.f0, self.maxstep))
        # force_mod = add_index(f, index, 0).reshape((-1,3))
        # print('modf:\n', force_mod)
        return force_mod

    def determine_step(self, dr, steplengths):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        maxsteplength = np.max(steplengths)
        if maxsteplength >= self.maxstep:
            dr *= self.maxstep / maxsteplength

        return dr

    def update(self, r, f, r0, f0):
        if self.H is None:
            self.H = np.eye(len(r)) * 70.0
            return
        dr = r - r0

        if np.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        df = f - f0
        a = np.dot(dr, df)
        dg = np.dot(self.H, dr)
        b = np.dot(dr, dg)
        self.H -= np.outer(df, df) / a + np.outer(dg, dg) / b

    def replay_trajectory(self, traj):
        """Initialize hessian from old trajectory."""
        if isinstance(traj, basestring):
            from ase.io.trajectory import Trajectory
            traj = Trajectory(traj, 'r')
        self.H = None
        atoms = traj[0]
        r0 = atoms.get_positions().ravel()
        f0 = atoms.get_forces().ravel()
        for atoms in traj:
            r = atoms.get_positions().ravel()
            f = atoms.get_forces().ravel()
            self.update(r, f, r0, f0)
            r0 = r
            f0 = f

        self.r0 = r0
        self.f0 = f0

    def run(self, fmax=0.05, steps=100000000):
        """ call Dynamics.run and keep track of fmax"""
        self.fmax = fmax
        if steps:
            self.max_steps = steps
        for converged in self.irun(fmax, steps):
            pass
        return converged

    def irun(self, fmax=0.05, steps=100000000, debug=False):
        """Run structure optimization algorithm as generator. This allows, e.g.,
        to easily run two optimizers at the same time.

        Examples:
        >>> opt1 = BFGS(atoms)
        >>> opt2 = BFGS(StrainFilter(atoms)).irun()
        >>> for _ in opt2:
        >>>     opt1.run()
        """

        if self.force_consistent is None:
            self.set_force_consistent()
        self.fmax = fmax
        step = 0
        # import pdb; pdb.set_trace()
        if self.converged():
            yield True
            return True
        while step < steps:
            origin_f = self.atoms.get_forces()
            # f = self.step(origin_f)
            try:
                f = self.step(origin_f)
            except CriticalPointError:
                if debug:
                    print('ValueError, initialize, step {0}, index: {1}'.format(
                        step, self.index))
                self.initialize()
                if debug:
                    print('index: {0}'.format(self.index))
                continue
            self.log(f)
            self.call_observers()
            if self.converged(f):
                yield True
                return
            yield False
            self.nsteps += 1
            step += 1

        yield False


class SOPT_LBFGS(LBFGS):
    def __init__(self, atoms, anchor, index=None, restart=None, logfile='-', trajectory=None,
                 maxstep=None, memory=100, damping=1, alpha=70,
                 use_line_search=False, master=None,
                 force_consistent=None, append_trajectory=False):
        self.anchor = anchor
        self.index = index
        super().__init__(atoms, restart=restart, logfile=logfile,
                         trajectory=trajectory,
                         maxstep=maxstep, memory=memory, damping=damping,
                         alpha=alpha, use_line_search=use_line_search, master=master,
                         force_consistent=force_consistent, append_trajectory=append_trajectory)

