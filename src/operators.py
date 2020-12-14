#Standard imports
import numpy as np
import itertools

# Local imports
from . import config, misc, checker
from . import curves as curv

# File where the considered spaces are defined, these are H_t.
# Also the forward operators are defined, these are K_t, K_t^*
# Time integration and dot products.

"""
 0. Time sampling
       The considered sampling times are to be defined in the config file,
       all the following spaces and methods are based on these time samples
       to be fixed.

1. The H_t spaces
       These spaces, for each time sample, could be different Hilbert spaces.
       This is why they have to be precisely defined, so each operator works
       properly. In the case of this code example, we will consider them all
       to be the complex space.
       Then, these H_t spaces are bundled together in a single space H.
       In the complex case, if H_t = C^{k_t}, then H = product_t C^{k_t}
       And such a Hilbert space can be embedded into H = C^{T, max_t(k_t)}
"""

# K is the dimension of each H_t. If needed, this dimension can be variable.
# defaul values

test_func = None
grad_test_func = None
K = None


def H_t_product(t, f_t, g_t):
    assert checker.is_in_H_t(t, f_t) and checker.is_in_H_t(t, g_t)
    # Computes the H_t product at between two elements in H_t
    # Input: t ∈ {0,1,...,T-1}
    #        f_t, g_t ∈ H_t = 1d numpy array with size K[t] 
    # Output: real value <f_t,g_t>_{H_t}.
    return np.real(np.dot(f_t, np.conj(g_t)))/K[t]

def H_t_product_set_vector(t, f_t, g_t):
    assert checker.set_in_H_t(t, f_t) and checker.is_in_H_t(t, g_t)
    # Computes the H_t product at between a numpy array of H_t elements with 
    # a sigle element in H_t
    # Input: t ∈ {0,1,...,T-1}
    #        f_t ∈ np.array of H_t elements,  g_t ∈ H_t = 1d numpy array
    # Output: Nx1 numpy array with values <f_t[i],g_t>_{H_t} for each H_t 
    #         element in H_t
    return np.real(np.dot(f_t, np.conj(g_t))).reshape(-1, 1)/K[t]

def int_time_H_t_product(f, g):
    assert checker.is_in_H(f) and checker.is_in_H(g)
    # Computes ∫<f_t, g_t>_{H_t} dt
    # Input : f,g ∈ H.
    # Output: real number.
    output = 0
    time_weights = config.time_weights
    for t in range(config.T):
        output += time_weights[t]*H_t_product(t, f[t], g[t])
    return output

"""
2. Forward operators K_t, K_t^*
        A simple way to define these bounded linear operators is to use
        convolutional kernels. In our case, we consider a family of test
        functions, whose sampling in certain locations in Fourier space,
        correspond to our H_t spaces.
        φ_t(x) is a continuous function from Ω to H_t, therefore
        K_t: H_t -> C(Ω); K_t(f_t) = <f_t,φ_t(x)>_{H_t}
        K_t_star: M(Ω) -> H_t; K_t(ρ) = <ρ, φ_t>_{M, C}

2.1 Sampling patterns
        To map into a Hilbert space that is manageable with a computer, we
        sample these φ_t(x) functions, leading to data in a complex C^K space.
"""

"""
2.2 Test functions and cut-off
        For the first order optimality conditions to be always valid, it is
        required to cut-off the boundary values.
"""

def K_t(t,f):
    assert checker.is_valid_time(t) and checker.is_in_H(f)
    # K_t: H_t -> C(\overline Ω)
    # K_t(f) = <f, φ(.,x)>_{H_t}
    # # It allows evaluation on a set of point x in Nx2 numpy array
    # Input: t ∈ {0,1,...,T-1}, f ∈ H.
    # Output: function x ∈ NxD -> Nx1, N is the number of eval. points.
    return lambda x: np.array([[H_t_product(t, f[t], test_func_j)
                                for test_func_j in test_func(t,x)]]).T

def grad_K_t(t,f):
    assert checker.is_valid_time(t) and checker.is_in_H(f)
    # ∇ K_t: H_t -> C(\overline Ω)
    # K_t(f) = <f, ∇φ(.,x)>_{H_t}
    # # It allows evaluation on a set of point x in Nx2 numpy array
    # Input: t ∈ {0,1,...,T-1}, f ∈ H.
    # Output: function x ∈ NxD -> 2xNx1, N number of eval. points, 2 for dx,dy.
    return lambda x: np.array([H_t_product_set_vector(t, dxdy, f[t]) for
                               dxdy in grad_test_func(t,x)])

def K_t_star(t,rho):
    assert checker.is_valid_time(t) and isinstance(rho, curv.measure)
    # K_t^*: M(Ω) -> H_t
    # K_t^*(ρ) = ρ_t(φ_t(·))
    # Input: t ∈ {0,1,...,T-1}, rho ∈ M, a measure.
    # Output: an element of H_t, an 1xK numpy array with K the dimension of H_t.
    return rho.spatial_integrate(t, lambda x: test_func(t,x))

def K_t_star_full(rho):
    assert isinstance(rho, curv.measure)
    # K_t_star: M(Ω) -> H
    # same as K_t_star, but it returns an element in H, not H_t
    # Input: rho ∈ M, a measure.
    # Output: an element of H, it is a numpy list of T elements, each a numpy
    #         array of size K[t], which is the dimension of H_t.
    output = []
    for t in range(config.T):
        output.append(K_t_star(t,rho)[0])
    return np.array(output)

class w_t:
    ' Class defining the dual variable in this problem '
    def __init__(self, rho_t):
        assert isinstance(rho_t, curv.measure)
        # take the difference between the current curve and the problem's data.
        if rho_t.intensities.size == 0:
            if config.f_t is None:
                # Case in which the data has not yet been set
                self.data = None
            else:
                self.data = -config.f_t
        else:
            self.data = K_t_star_full(rho_t)-config.f_t
        self.maximums = [np.nan for t in range(config.T)]
        self.sum_maxs = np.nan
        self.density_support = [np.nan for t in range(config.T)]
        self.as_predensity_mass = [np.nan for t in range(config.T)]
        self.density_max = [np.nan for t in range(config.T)]
        # the following member is for the rejection sampling algorithm
        self.size_epsilon_support = [np.nan for t in range(config.T)]
    def eval(self,t,x):
        assert checker.is_valid_time(t) and checker.is_in_space_domain(x)
        # Input: t ∈ {0,1,..., T-1}, x ∈ NxD numpy array, N number of points.
        # Output:  Nx1 numpy array
        return -K_t(t,self.data)(x)

    def grad_eval(self,t,x):
        assert checker.is_valid_time(t) and checker.is_in_space_domain(x)
        # Input: t ∈ {0,1,..., T-1}, x ∈ NxD numpy array, N number of points.
        # Output: 2xNx1 numpy array
        return -grad_K_t(t,self.data)(x)

    def animate(self, measure = None,
                resolution = 0.01, filename = None, show = True, block = False):
        """Animate the dual function w_t.

        This function uses matplotlib.animation.FuncAnimation to create an
        animation representing the dual variable w_t. Since the dual variable
        is a continuous function in Ω, it can be represented by evaluating
        it in some grid and plotting this in time.
        This method also supports a measure class input, to be overlayed on top
        of this animation. This option is helpful if one wants to see the
        current iterate μ^n overlayed on its dual variable,
        the solution curve of the insertion step or, at the first iteration,
        the backprojection of the data with the ground truth overlayed.
        ---------------------
        Arguments: None
        Output:    FuncAnimation object.
        ----------------------
        Keyword arguments:
            measure (measure class, default None):
                Measure class object to be overlayed in the animation.
            resolution (double, default 0.01):
                Resolution of the evaluation 2-dimensional grid to represent
                the dual variable
            filename (string, default None):
                If given, will save the output to a file <filename>.mp4.
            show (boolean, default True):
                Boolean to indicate if the animation should be shown.
        ---------------------
        small comment: the method returns a FuncAnimation object because it is
        required by matplotlib, else the garbage collector will eat it up and
        no animation would display. Reference:
        https://stackoverflow.com/questions/48188615/funcanimation-doesnt-show-outside-of-function
        """
        return misc.animate_dual_variable(self, measure , resolution = resolution,
                           filename = filename, show = show, block=block)

    def grid_evaluate(self, t, resolution = 0.01):
        evaluations = misc.grid_evaluate(lambda x: self.eval(t,x),
                                         resolution = resolution)
        maximum_at_t = np.max(evaluations)
        self.maximums[t] = maximum_at_t
        return evaluations, maximum_at_t

    def get_sum_maxs(self):
        # get the sum of the maximums of the dual variable. This value is a 
        # bound on the length of the inserted curves in the insertion step
        # the bound is β∫|γ'|^2/2 + α <= sum_t (ω_t * max_{x∈Ω} w_t(x) )
        if np.isnan(self.sum_maxs):
            self.sum_maxs = np.sum([config.time_weights[t]*self.maximums[t] for
                                    t in range(config.T)])
        return self.sum_maxs

    def density_transformation(self, x):
        # To consider this function as a density we apply the same transformation
        # at all times, for x a np.array, this is
        epsi = config.rejection_sampling_epsilon
        # it has to be an increasing function, that kills any value below -epsi
        return np.exp(np.maximum(x + epsi,0))-1

    def as_density_get_params(self, t):
        if np.isnan(self.as_predensity_mass[t]):
            # Produce, and store, the parameters needed to define a density with
            # the dual variable. These parameters change for each time t.
            evaluations = misc.grid_evaluate(lambda x:self.eval(t,x),
                                             resolution = 0.01)
            # extracting the epsilon support for rejection sampling
            # # eps_sup = #{x_i : w_n^t(x_i) > -ε}
            epsi = config.rejection_sampling_epsilon
            eps_sup = np.sum(evaluations > -epsi)
            # # density_support: #eps_sup / #{x_i in evaluations}
            # # i.e. the proportion of the support that evaluates above -ε
            self.density_support[t] = eps_sup/np.size(evaluations)
            # The integral of the distribution
            pre_density_eval = self.density_transformation(evaluations)
            mass = np.sum(pre_density_eval)*0.01**2
            self.as_predensity_mass[t] = mass
            self.density_max[t] = np.max(pre_density_eval)/mass
            return self.density_support[t], self.density_max[t]
        else:
            return self.density_support[t], self.density_max[t]

    def as_density_eval(self, t, x):
        # Considers the dual variable as a density, that is obtained by
        # discarding all the elements below epsilon, defined at the config file.
        # It interally stores the computed values, for later use.
        # Input: t ∈ {0,1,..., T-1}, x ∈ Ω numpy array.
        mass = self.as_predensity_mass[t]
        return self.density_transformation(self.eval(t,x))/mass

def overpenalization(s, M_0):
    assert isinstance(s, float) and isinstance(M_0, float)
    # the φ(t) function that is applied to the Benamou-Brenier energy
    # Input: s, M_0 ∈ R, real numbers.
    # Output: real number 
    if s <= M_0:
        return s
    else:
        return (s**2 + M_0**2)/2/M_0

def main_energy(measure, f):
    assert isinstance(measure, curv.measure) and checker.is_in_H(f)
    # Computes the main energy, the one we seek to minimize
    # Input: measure ∈ M, a measure type object.
    #        f ∈ H, an element of H. 
    # Output: positive number
    forward = K_t_star_full(measure)
    diff = forward - f
    return int_time_H_t_product(diff, diff)/2 + sum(measure.intensities)

