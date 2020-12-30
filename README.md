# DGCG solver

This solver is correspods to the one developed in
[this paper](https://arxiv.org/abs/2012.11706) named
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

### Documentation

The documentation of the code is available 
[here](https://dgcg-algorithm.readthedocs.io/en/latest/).

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

Requirements to run this code with virtual environment:
- Python3 (3.6 or above).
- The packages specified in the `requirements.txt` file.
- (optionally) `ffmpeg` installed and linked to `matplotlib` so the reconstructed
measure can be saved as `.mp4` videofiles. If `ffmpeg` is not available, 
this option can be disabled.

Alternatively, it can be run using [docker](https://www.docker.com/) without
any pre-requisite in any operative system. To run with this method:
- Install Docker in your system.
- Clone locally the repository.
- Execute the command `run -v $(pwd):$(pwd) -w $(pwd) --rm panchoop/dgcg_alg:v0.1`
It will execute your script saved as `main.py` in the same folder.

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

The file `src/Example_1.py` runs the numerical experiment #1 that is presented
in the paper. Run it directly inside the folder. To further understand 
how to use the module, it is recommended to take a look in the file. 
It is well commented.

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

#### Troubleshooting:
- When running the algorithm, nearing convergence the energy is not monotonously decreasing! 
- - **answer:** Try setting the tolerance value to something higher. Likely there are rounding errors, see [this issue](https://github.com/panchoop/DGCG_algorithm/issues/13#issue-774344239)


