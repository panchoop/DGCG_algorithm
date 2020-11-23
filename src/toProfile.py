# Standard python imports
import numpy as np
import matplotlib.pyplot as plt

# Not so standard python imports
import copy

# Local imports
import curves
import operators as op
import optimization as opt
import misc

# Regularization parameters
alpha = 1
beta  = 1
curves.alpha = opt.alpha = alpha
curves.beta  = opt.beta  = beta

# Target curve to track
gt_times = np.linspace(0,1,3)
gt_locations = np.array([[0.2,0.3],
                         [0.45,0.6],
                         [0.6, 0.3]])
gt_curve = curves.curve(gt_times, gt_locations)
ground_truth = curves.measure()
ground_truth.add(gt_curve,1)
# Data generation
f_t = lambda t: op.K_t_star(t,ground_truth)

# Initialize data
# Empty measure
rho = curves.measure()
s_star = 1.9 # Given K_t_star, the value of the associated constant is C = 1.

w_t = lambda t,x: -op.K_t(t,op.K_t_star(t,rho)-f_t(t))(x)
grad_w_t = lambda t,x: -op.grad_K_t(t,op.K_t_star(t,rho)-f_t(t))(x)

# Start iterating with a line as ground truth
#new_times = np.linspace(0,1,100)
#new_locations = np.array([[0.5,0.5],[0.5,0.5]])
#new_curve = curves.curve(np.linspace(0,1,2),new_locations)
#new_curve.set_times(new_times)


# Start iterating with a random guess
random_seed = 1
np.random.seed(random_seed) # no soo random
new_times = np.linspace(0,1,100)
dist = 0.03
new_curve = misc.random_curve(new_times,dist, 10)



#Gradient descent

step = beta/alpha*0.1

# In case of trying a conditional step
best_step = True
N_steps = 5
steps = np.linspace(0,step,N_steps)
steps_memory = [0,0,0] # to dynamically adapt the stepsize
step_change_desition = 4      # when to make a desition
end_iter = np.inf
max_iter = 3000
for i in range(max_iter):
    gradient = opt.step3_gradient(new_curve, w_t, grad_w_t)
    if best_step == True:
        energies = [ opt.step3_energy(new_curve-st*gradient, w_t) for st in steps]
        step_selection = np.argmin(energies)
        this_step = steps[step_selection]
        # step to adaptively change the stepsize reach
        if step_selection==0:
            steps_memory[0]+=1
            steps_memory[1]=steps_memory[2]=0
            step = step/2
            steps = np.linspace(0,step, N_steps)
        elif step_selection>(N_steps-1)/2:
            steps_memory[2]+=1
            steps_memory[0]=steps_memory[1]=0
        elif step_selection<(N_steps-1)/2:
            steps_memory[1]+=1
            steps_memory[0]=steps_memory[2]=0
        if steps_memory[0]==5*step_change_desition:
            end_iter = i
            break
        elif steps_memory[2]==step_change_desition:
            steps_memory[2]=0
            step=step*2
            steps = np.linspace(0,step,N_steps)
        elif steps_memory[1]==step_change_desition:
            steps_memory[1]=0
            step=step/2
            steps = np.linspace(0,step,N_steps)
        print(('iter: {:4}, step x Grad H1 relat. norm: {:12.5E},  Current energy: {:12.5E},'
               +' index step: {}, step: {:4.2E}').format(i,0,
                  #step*gradient.H1_norm()/new_curve.H1_norm(),
                  energies[step_selection],
                  step_selection, this_step))
    else:
        this_step = step
        print(('iter: {:4}, step x Grad H1 relat. norm: {:12.5E},  Current energy: {:12.5E},'
         +'').format(i,step*gradient.H1_norm()/new_curve.H1_norm(),
                  -new_curve.integrate_against(w_t)/new_curve.energy())
                  )
    new_curve = new_curve - this_step*gradient

if end_iter == np.inf:
    end_iter = max_iter

# # Attempt to shoot towards the optimal solution
C = 1/beta/opt.step3_energy(new_curve, w_t)
print('Constant to shoot with: ',C)
shoot_curve = opt.step3_shooting(grad_w_t,  new_curve.eval(0), C)



