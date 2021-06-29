# Atoms Pulling



Solve the problem of protein connecting with a molecule


2 molecule disconnecting




## Basic idea


A--B --> A + B
S --> E

question: chain is unknown, and we want to construct it.


## Solution

* Spherical optimization

* Steps
  1. init Start, End, remove translation/rotation, copy Start as S
  2. add force to S with 1/2 * k (p_E - p_S)^2
  3. optimize S until |p_S - p_S0| > threshold, maybe 1 Ang?
  4. optimize S with spherical optimization
  5. goto 2 until |p_E - p_S| < threshold, maybe 1A?
  6. store the path as Chain of States
  7. optimize with NEB


```python

class SuperAtoms(Atoms):

    def __init__(self, pair_atoms):
        # some initial here

    def get_forces(self):
        # add force to S with 1/2 * k (p_E - p_S)^2
        delta_positions = self.positions - self.pair_atoms.position
        scale = np.linalg.norm(delta_positions, axis=1) # ?? axis 1 or 0
        delta_forces = 0.5 * self.k * np.dot(delta_positions, scale**2)
        return self.atoms.get_positions() + delta_forces


class Pulling_Optimization(Optimization):


    def stop(self):
        if np.linalg.norm(self.positions - self.initial_position) > self.threshold:
            return True
        return False

```




## TS searching

* One point problem:
    * Given Rr, find nearby minumums and the path p

* Two Point TS problem:
    * Given Rr, Rp, find a path p so that min_p{max{E_p}}

* Minimum guessing
    * guess the minimums nearby

2P problem is much easier than 1P problem. 
Or 1P problem can be regarded as 2P problem & minimum guessing


* Path problem
* Minimum problem

### Path Problem


Given structure Rr, Rp, find a path p so that min_p{max{E_p}}

Chain of States methods are best suitable.


1. generate a chain
    * pair the atoms(ARRM)
    * interpolate the coordinates
2. optimize the chain
    * NEB/CI-NEB



