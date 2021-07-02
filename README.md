# DGCG solver

This solver is correspods to the one developed in
[this paper](https://arxiv.org/abs/2012.11706) named
**A Generalized Conditional Gradient Method for Dynamic Inverse Problems with 
Optimal Transport Regularization** by Kristian Bredies, Marcello Carioni, 
Silvio Fanzon and Francisco Romero-Hinrichsen.

A *dynamical inverse problem* is one in which the both the data, and the
forward operators are allowed to change in time.  For instance, one could
consider a medical imaging example, in which the patient is not completely
immobile (data changing in time), while simultaneously the measuring instrument
is not realising the same measurements at each time, as it is the case of a
rotating scanner in CT/SPECT/PET, or change of measurement frequencies in MRI.

The presented method tackles such *dynamic inverse problems* by 
proposing a convex energy to penalize jointly with a data discrepancy term; 
together these give us a target minimization problem from which we find a 
solution to the target dynamic inverse problem.

For clear, deep, and mathematically correct explanations, please refer to 
[the paper](https://arxiv.org/abs/2012.11706). The following is a mathematically
incomplete description of the considered Energy and minimization problem, but 
it is enough to intuitively describe it.

We look for **solutions** in the space of **dynamic Radon measures**, these are
[Radon measure](https://en.wikipedia.org/wiki/Radon_measure) defined on 
time and space `[0,1] x Ω`. 

Given a differentiable curve γ:[0,1] -> Ω, the 
[Lebesgue measure](https://en.wikipedia.org/wiki/Lebesgue_measure) `dt`, and the 
[Dirac delta](https://en.wikipedia.org/wiki/Dirac_delta_function#As_a_measure)
δ, one can consider the following
 [product measure](https://en.wikipedia.org/wiki/Product_measure) 

<p align="center">
ρ<sub>γ</sub> := dt x δ<sub>γ(t)</sub>
</p>
representing a Dirac delta transported in time by the curve γ in space, which
is a dynamic Radon measure.

The energy we consider to solve the target *dynamic inverse problem* is
parametrized by α, β > 0, and acts
in the following way over such element:
<p align="center">
<img src="https://github.com/panchoop/DGCG_algorithm/blob/assets/tex/eq_4.gif" width="300">
</p>
Intuitively, the energy is penalizing the squared speed of the dynamic Radon
measures.

Since measure spaces are in particular vector spaces, given a family of weights
ω<sub>i</sub> >0,  and a family of curves γ<sub>i</sub>, we can now consider μ, 
a weighted sum of these transported Dirac deltas
<p align="center">
<img src="https://github.com/panchoop/DGCG_algorithm/blob/assets/tex/eq_5.gif" width="600">
</p>
which is also a dynamic Radon measure.

The measures are "moving time continuously", but the measurements are gathered
by sampling discretely in time. Fix those time samples as 0 = t<sub>0</sub> < 
t<sub>1</sub> < ... < t<sub>T</sub> = 1, then, at each time sample, the
considered dynamic Radon measures are simply Radon measures. We therefore 
consider at each of these time samples t<sub>i</sub>, a **forward operator**
mapping from the space of Radon measures, into some **data space** H<sub>i</sub>

<p align="center">
<img src="https://github.com/panchoop/DGCG_algorithm/blob/assets/tex/eq_6.gif" width="200">
</p>

Where at each time sample t<sub>i</sub>, the respective data spaces
H<sub>i</sub> are allowed to be different. Theoretically, these data spaces
are real [Hilbert spaces](https://en.wikipedia.org/wiki/Hilbert_space), numerically,
these need to be finite dimensional.

Given **data** gathered at each time sample f<sub>0</sub> ∈ H<sub>0</sub>,
f<sub>1</sub> ∈ H<sub>1</sub>, ...  f<sub>T</sub> ∈ H<sub>T</sub>, and given
any dynamical Radon measure ν, the data discrepancy term of our minimization
problem is

<p align="center">
<img src="https://github.com/panchoop/DGCG_algorithm/blob/assets/tex/eq_7.gif" width="350">
</p>

And putting together the data discrepancy term with the proposed 
energy J<sub>α, β</sub> to minimize, we build up the target 
functional that is minimized by our algorithm.

<p align="center">
<img src="https://github.com/panchoop/DGCG_algorithm/blob/assets/tex/eq_1.gif" width="800">
</p>
                                                                                                            
The energy J<sub>α, β</sub> will promote sparse dynamic measures  μ, and the
proposed algorithm will return one such measure.

To see an animated example of Dynamic sources, measured data, and obtained reconstructions,
please see [this video](https://www.youtube.com/watch?v=daKkJZH3WD4).

### Documentation

The documentation of the code is available 
[here](https://dgcg-algorithm.readthedocs.io/en/latest/).

### Theoretical requirements

#### Strong requirements (unavoidables)

- A finite family of Hilbert spaces H<sub>i</sub> that can be numerically represented.
- A corresponding finite set of time samples t<sub>i</sub>.
- Forward operators K<sub>i</sub><sup>\*</sup>: M(Ω) -> H<sub>i</sub>, 
 that represent your measurements at each time sample,
with predual K<sub>i</sub>: H<sub>i</sub> -> C(Ω), 
mapping in particular into differentiable functions.
- Data g<sub>i</sub> ∈ H<sub>i</sub> corresponding to the
  measurements of the ground truth at each time sample.

#### Soft requirements (avoidable, but will require additional work)

- The time samples in the interval `[0,1]`: very easy to adapt.
- Dimension `d = 2` of domain `Ω`: intermediate work.
- 2-dimensional non-periodic domain of interest `Ω = [0,1]x[0,1]`: 
intermediate work, should not be an issue as long as the desired domain is convex
or the curves are far apart from the boundary. Otherwise, quite challenging.
- Forward operators K<sub>i</sub><sup>\*</sup> smoothly vanishing on the
  boundary `∂Ω`: very hard to adapt, the whole implemented code relies on the
solutions lying on the interior of the domain. To lift this requirement, the
insertion step and sliding step of the algorithm must consider projected
gradient descent strategies to optimize for curves touching the boundary. 
But, given any forward operator K<sub>i</sub><sup>\*</sup>, it is possible to
smoothly *cut-off* the values near `∂Ω`. The implemented Fourier measurements
consider such cut-off to enforce this condition.

### Manual

#### Download

To get this repository, clone it locally with the command

``` 
git clone https://github.com/panchoop/DGCG_algorithm.git
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

#### Warning 1

The code itself is **not** plug and play. It will require rewriting your 
operators and spaces to fit the implemented structures. 

Specifically:
- Properly incorporate your forward operator.
- Properly specify your Hilbert space and its inner product

Additionally, keep in mind that given that the output of this method is a Radon
measure composed of a weighted sum of deltas transported by curves, the output
of this algorithm is such object encoded in the `measure class` implemented in
the `classes.py` module. 

These objects are [pickled](https://docs.python.org/3/library/pickle.html)
 for later use, simultaneously they can be
exported as a `.mp4` video file (via the `.animate()` class method).

#### Warning 2
The code is heavily sub-optimized. Therefore expect long execution times.
See table 1 in paper.

### Working example/Tutorial

The file `examples/Example_1.py` runs the numerical experiment #1 that is presented
in the paper. Run it directly inside the folder. To further understand 
how to use the module, it is recommended to take a look in the file. 
It is well commented.

The script will generate a folder where the iteration results will be stored. 

Further files `Example_2_*.py` and `Example_3` are the ones presented in the
paper.

### Fast and easy way to consider a forward operator K<sup>\*</sup>

One can define/construct forward operators using integration kernels.
Let φ:Ω -> H be a differentiable function mapping from our domain of interest
Ω to some Hilbert space H. Then, we can define the forward operator 
K<sup>\*</sup>: M(Ω) -> H, and its predual K: H -> C(Ω) as

![eq_3](https://github.com/panchoop/DGCG_algorithm/blob/assets/tex/eq_3.gif)

Differentiability of φ is required for the differentiability of K(h),
allowing us to minimize the linearized minimization problem arising from
the insertion step of this algorithm.

#### Troubleshooting:
- When running the algorithm, nearing convergence the energy is not
  monotonously decreasing! 
- - **answer:** Try setting the tolerance value to something higher. Likely
    there are rounding errors, see [this
issue](https://github.com/panchoop/DGCG_algorithm/issues/13#issue-774344239)


