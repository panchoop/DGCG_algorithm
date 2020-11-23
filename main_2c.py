import numpy as np
import pickle
import itertools
import DGCG


# Simulation parameters
T = 51
time_samples = np.linspace(0,1,T)
K = np.ones(T, dtype=int)*18

# Kernels to define the forward operators

def cut_off(s):
    # A one dimentional cut-off function that is twice differentiable, monoto-
    # nous and fast to compute.
    # Input: s in Nx1 numpy array, representing 1-D evaluations.
    # Output: a Nx1 numpy array evaluating the 1-D cutoff along the input s.
    # the cut-off threshold is the width of the transition itnerval from 0 to 1.
    cutoff_threshold = 0.1
    transition = lambda s: 10*s**3 - 15*s**4 + 6*s**5
    val = np.zeros(s.shape)
    for i in range(len(s)):
        if s[i]>= cutoff_threshold and s[i]<= 1-cutoff_threshold:
            val[i]=1
        elif s[i]<cutoff_threshold and s[i]>= 0:
            val[i]= transition(s[i]/cutoff_threshold)
        elif s[i]>1-cutoff_threshold and s[i]<=1:
            val[i]= transition(((1-s[i]))/cutoff_threshold)
        else:
            val[i] = 0
    return val

def D_cut_off(s):
    # The derivative of the defined cut_off function.
    # Same input and output sizes.
    cutoff_threshold = 0.1
    D_transition = lambda s: 30*s**2 - 60*s**3 + 30*s**4
    val = np.zeros(s.shape)
    for i in range(len(s)):
        if s[i]< cutoff_threshold and s[i]>=0:
            val[i]= D_transition(s[i]/cutoff_threshold)/cutoff_threshold
        elif s[i]<=1 and s[i] >= 1-cutoff_threshold:
            val[i]= -D_transition(((1-s[i]))/cutoff_threshold)/cutoff_threshold
        else:
            val[i]=0
    return val

# Implementation of the test functions

max_frequency = 15
max_angles = 5
spacing = 1
def sample_line(num_samples, angle, spacing):
    # To sample along a line.
    rotation_mat = np.array([[np.cos(angle), np.sin(angle)],
    [-np.sin(angle), np.cos(angle)]])
    x = [-spacing*(i+1) for i in range(num_samples//2)]
    x.extend([spacing*(i+1) for i in range(num_samples - num_samples//2)])
    horizontal_samps = [ np.array([xx, 0]) for xx in x]
    rot_samples = [samp@rotation_mat for samp in horizontal_samps]
    return rot_samples
def available_samples(angle):
    # To set the available samples per angle
    av_samps = [np.array([0,0])]
    av_samps.extend(sample_line(max_frequency-1,angle,spacing))
    return np.array(av_samps)
# 
angles = np.linspace(0,np.pi,max_angles)[:-1]
angle_cycler = itertools.cycle(angles)
# Set the sampling method
sampling_method = [available_samples(next(angle_cycler)) for t in
                   range(T)]
K = np.ones(T, dtype=int)*max_frequency

def test_func(t,x): # φ_t(x)
    # Input: t∈[0,1,2,...,T-1]
    #        x numpy array of size Nx2, representing a list of spatial points
    #            in R^2.
    # Output: NxK numpy array, corresponding to the  test function evaluated in
    #         the set of spatial points.

    # # complex exponential test functions
    expo = lambda s: np.exp(-2*np.pi*1j*s)
    # # The evaluation points for the expo functions, size NxK.
    evals = x@sampling_method[t].T
    # # The considered cutoff, as a tensor of 1d cutoffs (output: Nx1 vector)
    h = 0.1
    cutoff = cut_off(x[:,0:1])*cut_off(x[:,1:2])
    # return a np.array of vectors in H_t, i.e. NxK numpy array.
    return expo(evals)*cutoff

def grad_test_func(t,x): # ∇φ_t(x)
    # Gradient of the test functions before defined. Same inputs.
    # Output: 2xNxK numpy array, where the first two variables correspond to
    #         the dx part and dy part respectively.
    # #  Test function to consider
    expo = lambda s: np.exp(-2*np.pi*1j*s)
    # # The sampling locations defining H_t
    S = sampling_method[t]
    # # Cutoffs
    h = 0.1
    cutoff_1 = cut_off(x[:,0:1])
    cutoff_2 = cut_off(x[:,1:2])
    D_cutoff_1 = D_cut_off(x[:,0:1])
    D_cutoff_2 = D_cut_off(x[:,1:2])
    # # preallocating
    N = x.shape[0]
    output = np.zeros((2,N,K[t]), dtype = 'complex')
    # # Derivative along each direction
    output[0] = expo(x@S.T)*cutoff_2*(
                                -2*np.pi*1j*cutoff_1@S[:,0:1].T + D_cutoff_1)
    output[1] = expo(x@S.T)*cutoff_1*(
                                -2*np.pi*1j*cutoff_2@S[:,1:2].T + D_cutoff_2)
    return output

DGCG.set_parameters(time_samples, K, test_func, grad_test_func)

# Generate data. 
params = {
    'energy1':1,
    'energy2':1,
    'energy3':1,
    'save_data':True,
}
# Curves defining the ground truth
# # middle curve
gt_curve_1 = DGCG.curves.curve(np.linspace(0,1,2), np.array([[0.1,0.1],
                                                        [0.7,0.8]]))
# # top curve: a circle segment
times_2 = np.linspace(0,1,101)
center = np.array([0.1,0.9])
radius = np.linalg.norm(center-np.array([0.5,0.5]))-0.1
x_2 = radius*np.cos( 3*np.pi/2 + times_2*np.pi/2) + center[0]
y_2 = radius*np.sin( 3*np.pi/2 + times_2*np.pi/2) + center[1]
positions_2 = np.array([x_2,y_2]).T
gt_curve_2 = DGCG.curves.curve(times_2, positions_2)
# # bottom curve: circular segment + straight segment
tangent_time = 0.5
tangent_point = gt_curve_1.eval(tangent_time)
tangent_direction = gt_curve_1.eval(tangent_time)-gt_curve_1.eval(0)
normal_direction = np.array([tangent_direction[0,1],
                             -tangent_direction[0,0]])/ \
        np.linalg.norm(tangent_direction)
radius = 0.3
init_angle = 4.5*np.pi/4
end_angle  = np.pi*6/16/2
middle_time = 0.8
center_circle = tangent_point + radius*normal_direction
increase_factor = 1 # this has to be < 1
increase = lambda t: increase_factor*t**2 + (1-increase_factor)*t
times_3 = np.arange(0,middle_time,0.01)
times_3 = np.append(times_3, middle_time)
x_3 = np.cos( init_angle -
             increase(times_3/middle_time)*(init_angle-end_angle))
x_3 = radius*x_3 + center_circle[0,0]
y_3 = np.sin( init_angle -
             increase(times_3/middle_time)*(init_angle-end_angle))
y_3 = radius*y_3 + center_circle[0,1]
# # # straight line
times_3 = np.append(times_3, 1)
middle_position = np.array([x_3[-1], y_3[-1]])
last_speed = 1
last_position = middle_position*(1-last_speed) + \
                last_speed*center_circle
x_3 = np.append(x_3,last_position[0,0])
y_3 = np.append(y_3,last_position[0,1])
positions_3 = np.array([x_3, y_3]).T
gt_curve_3 = DGCG.curves.curve(times_3, positions_3)
# Setting up the ground truth
ground_truth = DGCG.curves.measure()
ground_truth.add(gt_curve_1, gt_curve_1.energy()*params['energy1'])
ground_truth.add(gt_curve_2, gt_curve_2.energy()*params['energy3'])
ground_truth.add(gt_curve_3, gt_curve_3.energy()*params['energy2'])
### uncomment to see the animated curve
#ground_truth.animate()

# Simulate the measurements generated by this curve
data = DGCG.operators.K_t_star_full(ground_truth)
## uncomment to see the backprojected data
#DGCG.config.f_t = data
#dual_variable = DGCG.operators.w_t(DGCG.curves.measure())
#ani_1 = dual_variable.animate(measure = ground_truth, block = True)

# Add noise to the measurements
noise_level = 0.4 # 20% of noise
# Load noise vector used in paper
noise_vector = pickle.load(open('annex/noise_vector.pickle', 'rb'))
noise_vector = noise_vector/np.sqrt(DGCG.operators.int_time_H_t_product(noise_vector,
                                                            noise_vector))
data_H_norm = np.sqrt(DGCG.operators.int_time_H_t_product(data,data))
data_noise = data + noise_vector*noise_level*data_H_norm

import code; code.interact(local=dict(globals(), **locals()))
## uncomment to see the backprojected data
#DGCG.config.f_t = data_noise
#dual_variable = DGCG.operators.w_t(DGCG.curves.measure())
#ani_2 = dual_variable.animate(measure = ground_truth, block = True)

# Use the DGCG solver
alpha = 0.1
beta = 0.1

# <+TODO+> officialize these settings
DGCG.config.temp_folder = 'temp_2c'
DGCG.config.step3_min_attempts_to_find_better_curve = 10000
DGCG.config.step3_max_attempts_to_find_better_curve = 10005

current_measure = DGCG.solve(data_noise, alpha, beta)
