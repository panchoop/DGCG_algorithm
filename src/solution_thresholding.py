import curves
import numpy as np
import pickle
import config
import matplotlib.pyplot as plt

config.alpha = 0.8/4
config.beta  = 0.5/3

# Get target solution
filename = 'temp16/iter_007_step_3_grad_flow_saved_variables.pickle'

# load it
saved_files = open(filename, 'rb')
data = pickle.load(saved_files)
saved_files.close()

# Get object of interest
current_measure = data['current_measure']

# Erase the curves with intensity lower than threshold
new_measure = curves.measure()
threshold = 0.05
for idx, intensity in enumerate(current_measure.intensities):
    if intensity > threshold:
        curv = current_measure.curves[idx]
        energy = current_measure.energies[idx]
        new_measure.add(curv, (energy/intensity)**-1)


new_measure.draw()
plt.title('Thresholded (>{}) current solution'.format(threshold))
plt.savefig('current_solution_t{}.pdf'.format(threshold))
plt.close()

new_measure = curves.measure()
threshold = 0.01
for idx, intensity in enumerate(current_measure.intensities):
    if intensity > threshold:
        curv = current_measure.curves[idx]
        energy = current_measure.energies[idx]
        new_measure.add(curv, (energy/intensity)**-1)


new_measure.draw()
plt.title('Thresholded (>{}) current solution'.format(threshold))
plt.savefig('current_solution_t{}.pdf'.format(threshold))
plt.close()

new_measure = curves.measure()
threshold = 0.1
for idx, intensity in enumerate(current_measure.intensities):
    if intensity > threshold:
        curv = current_measure.curves[idx]
        energy = current_measure.energies[idx]
        new_measure.add(curv, (energy/intensity)**-1)


new_measure.draw()
plt.title('Thresholded (>{}) current solution'.format(threshold))
plt.savefig('current_solution_t{}.pdf'.format(threshold))
plt.close()

new_measure.animate(filename='current_solution_t{}.pdf'.format(threshold), show=False)

