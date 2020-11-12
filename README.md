# DGCG solver

This python code correspond to the one developped for the paper 
**A Generalized Conditional Gradient Method for Dynamic Inverse Problems with 
Optimal Transport Regularization** by Kristian Bredies, Marcello Carioni, 
Silvio Fanzon and Francisco Romero-Hinrichsen.

A *dynamical inverse problem* is one in which the both the data, and
the forward operators are allowed to change in time. 

The code is meant to solve *dynamic inverse problems* using a Tikhonov
regularization posed in the space of Radon measures, which penalizes
both the total variation norm, together with the Benamou-Brenier energy. 

``` Link here to the main equation to minimize```

Given the considered penalizations, the obtained solution will be a 
*sparse* dynamic Radon measure, this is, a Radon measure with the
following structure

``` Link here to the general structure```

that is a positively-weighted sum of Dirac deltas transported by curves in `H^1`.


### Theoretical requirements

#### Strong requirements (unavoidables)

- A finite family of Hilbert spaces `H_i` that can be numerically represented.
- A corresponding finite set of time samples `t_i`.
- Forward operators `K_i^*: M(Ω) -> H_i` that represent your measurements at each time sample,
with predual `K_i: H_i -> C(Ω)` mapping in particular into differentiable functions.
- Data `g_i \in H_i` corresponding to the measurements of the ground truth at each time sample.

#### Soft requirements (avoidable, but will require additional work)

- The time samples between `[0,1]`: very easy to adapt.
- Dimension `d = 2` of domain `Ω`: intermediate work.
- 2-dimensional non-periodic domain of interest `Ω = [0,1]x[0,1]`: 
intermediate work, should not be an issue as long as the desired set is convex or the
curves are far apart from the boundary. Otherwise, quite challenging.
- Forward operators `K_i^*` smoothly vanishing on the boundary `∂Ω`: very hard to 
adapt, the whole implemented code relies on the solutions lying on the interior 
of the domain. To lift this requirement, the insertion step and sliding
step of the algorithm must consider projected gradient descent strategies
to optimize for curves touching the boundary. 
But, given any forward operator `K_i^*`, it is possible to smoothly *cut-off* 
the values near `∂Ω`. The implemented Fourier measurements consider such
cut-off to enforce this condition.



### Manual

#### Download

To get this repository, clone it locally with the command

``` 
git clone ASDF@ASDF
```

#### Foreword

The code itself is **not** plug and play. It will require rewriting your 
operators and spaces to fit the implemented structures. 
In particular, given that the output of this method is a Radon measure
composed of a weighted sum of deltras transported by curves, 
the output of this algorithm is such object encoded in the
`measure class` implemented in the `curves.py` module. 

These measure class objects have as method `.animate()` which allows
to save them as `.mp4` files. 

#### Basic instruction

If both the *strong and soft requirements* are satisfied, then to use

### Working example

The file `main.py` Contains a script that runs the code for the default 
implemented case, this is, low frequency Fourier measurements. 

The script will generate a folder where the iteration results will be stored. 

### Fast and easy way to consider a forward operator `K^*`

One can define/construct forward operators using integration kernels.
Let `φ` be a differentiable function `blabla`. Then, we can define

```
blabla
```

Furthermore, given differentiability, we can:

```
blablabla
```




#### Main structures

```curves.py```
Module that implements th


#### WARNING:
The code is heavily sub-optimized. Therefore expect long execution times.
See table 1 in paper.




------------

### TODO's:
- Test if the method tolerates changing frequency numbers.
- Write a tutorial using the main.py function as example
- Put on an examples folder the known operators and data.
- General settings access via the DGCG module (all those in `config.py`)
- Rework logger class
- - Fix messages that do not go through it
- - Add a silent execution option with custom outputs
- - Eliminate the stepsize signaling
- - Possibly add a loading bar instead of printing every iteration.
- - Compress the tabu-curve plot to some histogram of sorts with explanation.
- - Well-document the pickled varibles whenever saving
- - Test an `output to logfile` option.
- Silence the `cvxopt` module.
- Output of the insertion step to not be a `None` Measure
- Refactor code:
- - Separate curves.py into curves.py and measures.py
- - There are two circular imports: the `misc.py` imports optimization to get 
the `dual_gap` method and the `checker.py` module with the `operators.py`.

--------------

### Import relationships

Everyone requires config. Config requires no one.

checker < operators. To use K_t^*
operators < checker. To test that everything works.
##### Solution. Take out the assertion steps inside the operators?

curves < misc to and draw (supersample method)
measures < misc to animate (Animate class)
measures < operators, for op.main_energy

operators < curv only for the assert statements
operators < misc for: sample line, cut_off definition, etc.

optimization < curves, gradient construction, curve_product
             < measure, candidate_measure 
             < operators, instantiate w_t, compute K_t, K_t^* for quadratic opt
             < insertion_mod

misc > optimization (dual_gap) -> I have to place this function somewhere else

optimization (dual_gap) < measures, operators (time integration, w_t)
