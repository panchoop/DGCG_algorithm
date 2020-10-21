import config
import numpy as np
import curves as curv
import misc
import optimization as opt
import checker
import itertools

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
global K
global sampling_method
K = np.ones(config.T, dtype=int)*18
available_samples = np.array([misc.Archimedian_spiral(t,0,0.2) for t in
                                            np.arange(K[0])]) # K[i] are equal.
sampling_method = [available_samples for t in range(config.T)]


# Method to change to other sampling strategies the frequencies
def change_sampling_method(sampling_meth, *args):
    global sampling_method
    # Store parameters for later plotting
    logger = misc.logger()
    logger.store_parameters(config.T, sampling_meth, list(args))
    global K
    if sampling_meth == 1 or sampling_meth == 'constant_spiral':
        # time-constant spiral sampling
        # 1st input, the highest frequency
        constant_frequency = args[0]
        available_samples = np.array([misc.Archimedian_spiral(t,0,0.2) for t in
                                            np.arange(constant_frequency)])
        sampling_method = [available_samples for t in range(config.T)]
        K = np.ones(config.T, dtype=int)*constant_frequency
    if sampling_meth == 2 or sampling_meth == 'alternating_spiral':
        # This method takes the spiral sampling up to a maximal frequency
        # then subdivides it in different subsamples, that are executed 
        # cyclically and that their definition is also cyclical in the sampling
        # points, such that no sampling point is shared by any of the 
        # subsamples.
        # Inputs: Maximal frequency, number of cycles
        maximal_frequency = args[0]
        cycles = args[1]
        assert isinstance(cycles, int) and isinstance(maximal_frequency, int)
        spiral_samples = np.array([misc.Archimedian_spiral(t,0,0.2) for t
                                        in np.arange(maximal_frequency)])
        subsamples = []
        for i in range(cycles):
            subsample = np.array([spiral_samples[j] for j in
                                      np.arange(i, maximal_frequency, cycles)])
            subsamples.append(subsample)
        subsamples_cycler = itertools.cycle(subsamples)
        sampling_method = [next(subsamples_cycler) for t in range(config.T)]
        # setting the number of frequencies per dimension
        K = np.ones(config.T, dtype=int)
        for t in range(config.T):
            K[t] = len(sampling_method[t])
    if sampling_meth == 3 or sampling_meth == 'lines_sampling':
        # time-constant line sampling
        # Inputs 0: the max number of frequencies
        #        1: Number of lines
        #        2?: Maybe add spacing of the samples
        max_frequencies = args[0]
        line_number = args[1]
        if len(args)==3:
            spacing = args[2]
        else:
            spacing = 0.3
        # Getting the number of frequencies per line: add cyclicly one per line
        K_L = [0]*line_number
        for i in range(max_frequencies-1):
            K_L[i%line_number] += 1
        # A function to sample, excluding the zero, at any line
        angles = np.linspace(0,np.pi, line_number+1)[:-1]
        available_samples = [np.array([0,0])]
        # Put everything together in the available_samples array employinh
        # the sample_line function in the misc module
        for angle, freq in zip(angles,K_L):
            available_samples.extend(misc.sample_line(freq,angle,spacing))
        available_samples = np.array(available_samples)
        # Exporting the samples to the global method
        sampling_method = [available_samples for t in range(config.T)]
        K = np.ones(config.T, dtype=int)*max_frequencies
    if sampling_meth == 4 or sampling_meth == 'single_line':
        # Just to show what happens when we measure a single line
        # time-constant line sampling
        # Inputs 0: max number of frequencies
        #        1: angle of the line (radians)
        #        2?: maybe add the spacing of the samples
        max_frequency = args[0]
        angle = args[1]
        spacing = 0.3
        available_samples = [np.array([0,0])]
        # Put everything together in the available_samples array
        # using the sample_line method of the misc module
        available_samples.extend(misc.sample_line(max_frequency-1,angle,spacing))
        available_samples = np.array(available_samples)
        # Exporting the samples to the global method
        sampling_method = [available_samples for t in range(config.T)]
        K = np.ones(config.T, dtype=int)*max_frequency
    if sampling_meth == 5 or sampling_meth == 'rotating_line':
        # Single line measurement, but rotating in time
        # inputs 0: max frequency per line
        #        1: max_angles
        #        2?: spacing
        #        3?: time samples per line
        max_frequency = args[0]
        max_angles = args[1]
        if len(args)<=2:
            spacing = 0.3
        elif len(args)>= 3:
            spacing = args[2]
        # Same proces as in sampling_meth == 4, the fixed line. But rotating.
        def available_samples(angle):
            av_samps = [np.array([0,0])]
            av_samps.extend(misc.sample_line(max_frequency-1,angle,spacing))
            return np.array(av_samps)
        angles = np.linspace(0,np.pi,max_angles)[:-1]
        angle_cycler = itertools.cycle(angles)
        sampling_method = [available_samples(next(angle_cycler)) for t in
                           range(config.T)]
        K = np.ones(config.T, dtype=int)*max_frequency

def H_t_product(t,f_t,g_t):
    assert checker.is_in_H_t(t,f_t) and checker.is_in_H_t(t,g_t)
    # Computes the H_t product at between two elements in H_t
    # Input: t ∈ {0,1,...,T-1}
    #        f_t, g_t ∈ H_t = 1d numpy array with size K[t] 
    # Output: real value <f_t,g_t>_{H_t}.
    return np.real(np.dot(f_t, np.conj(g_t)))/K[t]

def H_t_product_set_vector(t, f_t, g_t):
    assert checker.set_in_H_t(t,f_t) and checker.is_in_H_t(t,g_t)
    # Computes the H_t product at between a numpy array of H_t elements with 
    # a sigle element in H_t
    # Input: t ∈ {0,1,...,T-1}
    #        f_t ∈ np.array of H_t elements,  g_t ∈ H_t = 1d numpy array
    # Output: Nx1 numpy array with values <f_t[i],g_t>_{H_t} for each H_t 
    #         element in H_t
    return np.real(np.dot(f_t, np.conj(g_t))).reshape(-1,1)/K[t]


def H_t_product_full(f,g):
    assert checker.is_in_H(f) and checker.is_in_H(g)
    # Computes the H_t product at all time samples between f,g ∈ H.
    # Input: f,g ∈ H = [H_t]_{t} = numpy array of 1d numpy arrays.
    # Output: Tx1 sized numpy vector (<f_t,g_t>_{H_t})_{t=1,..}
    # OPTIMIZABLE: if instead a 2d numpy array is considered.
    output = np.zeros((config.T,1))
    for t in range(config.T):
        output[t] = H_t_product(t,f[t],g[t])
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

def test_func(t,x): # φ_t(x)
    assert checker.is_valid_time(t) and checker.is_in_space_domain(x)
    # Family of functions used to test against, to define the K_t operators.
    # Input: t∈[0,1,2,...,T-1]
    #        x numpy array of size Nx2, representing a list of spatial points
    #            in R2.
    # Output: NxK numpy array, corresponding to the  test function evaluated in
    #         the set of spatial points.

    # # complex exponential test functions
    expo = lambda s: np.exp(-2*np.pi*1j*s)
    # # The evaluation points for the expo functions, size NxK.
    evals = x@sampling_method[t].T
    # # The considered cutoff, as a tensor of 1d cutoffs (output: Nx1 vector)
    h = 0.1
    cutoff = misc.cut_off(x[:,0:1],h)*misc.cut_off(x[:,1:2],h)
    # return a np.array of vectors in H_t, i.e. NxK numpy array.
    return expo(evals)*cutoff

def grad_test_func(t,x): # ∇φ_t(x)
    assert checker.is_valid_time(t) and checker.is_in_space_domain(x)
    # Gradient of the test functions before defined. Same inputs.
    # Output: 2xNxK numpy array, where the first two variables correspond to
    #         the dx part and dy part respectively.
    # #  Test function to consider
    expo = lambda s: np.exp(-2*np.pi*1j*s)
    # # The sampling locations defining H_t
    S = sampling_method[t]
    # # Cutoffs
    h = 0.1
    cutoff_1 = misc.cut_off(x[:,0:1],h)
    cutoff_2 = misc.cut_off(x[:,1:2],h)
    D_cutoff_1 = misc.D_cut_off(x[:,0:1],h)
    D_cutoff_2 = misc.D_cut_off(x[:,1:2],h)
    # # preallocating
    N = x.shape[0]
    output = np.zeros((2,N,K[t]), dtype = 'complex')
    # # Derivative along each direction
    output[0] = expo(x@S.T)*cutoff_2*(
                                -2*np.pi*1j*cutoff_1@S[:,0:1].T + D_cutoff_1)
    output[1] = expo(x@S.T)*cutoff_1*(
                                -2*np.pi*1j*cutoff_2@S[:,1:2].T + D_cutoff_2)
    return output

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

    def show(self):
        'Print the target dual function'
        misc.plot_2d_time(lambda t,x: self.eval(t,x), total_animation_time = 2)

    def grid_evaluate(self, t, resolution = config.max_curve_x_res):
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
                                             resolution = config.max_curve_x_res)
            # extracting the epsilon support for rejection sampling
            # # eps_sup = #{x_i : w_n^t(x_i) > -ε}
            epsi = config.rejection_sampling_epsilon
            eps_sup = np.sum(evaluations > -epsi)
            # # density_support: #eps_sup / #{x_i in evaluations}
            # # i.e. the proportion of the support that evaluates above -ε
            self.density_support[t] = eps_sup/np.size(evaluations)
            # The integral of the distribution
            pre_density_eval = self.density_transformation(evaluations)
            mass = np.sum(pre_density_eval)*config.max_curve_x_res**2
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

def int_time_H_t_product(f, g):
    assert checker.is_in_H(f) and checker.is_in_H(g)
    # Computes ∫<f_t, g_t>_{H_t} dt
    # Input : f,g ∈ H.
    # Output: real number.
    output = 0
    time_weights = config.time_weights
    for t in range(config.T):
        output += time_weights[t]*H_t_product(t,f[t], g[t])
    return output

def main_energy(measure, f):
    assert isinstance(measure, curv.measure) and checker.is_in_H(f)
    # Computes the main energy, the one we seek to minimize
    # Input: measure ∈ M, a measure type object.
    #        f ∈ H, an element of H. 
    # Output: positive number
    forward = K_t_star_full(measure)
    diff = forward - f
    return int_time_H_t_product(diff, diff)/2 + sum(measure.intensities)

