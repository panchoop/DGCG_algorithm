"""
Logger class module

Undocumented. It should be replaced by a proper logging method that
uses native python's ``logging`` module.

If the logging is too obnoxious, it is always possible to delete all calls
to the logger in the solver and the solver should works just fine.
"""
# Standard imports
import os
import datetime
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from . import config

class logger:
    def __init__(self):
        self.init_time = datetime.datetime.now()
        self.times = []
        self.steps = []
        self.energies = np.array([])
        self.number_elements = []
        self.dual_gaps = np.array([])
        self.current_iter = 0
        self.aux = None
        self.iter = 0
        if config.log_output:
            self.logtext = ''
            self.logcounter = 0
            f = open('{}/log.txt'.format(config.results_folder), 'w')
            f.write('Logging!')
            f.close()

    def status(self, sect, *args):
        temp = config.results_folder
        if sect == [1] or sect == [2]:
            # [1] means the end of the sliding-step, or right before the
            # insertion.  [2] means the end of the insertion-step or right
            # before sliding.
            if sect == [1]:
                self.iter += 1
            num_iter = args[0]
            self.current_iter = num_iter
            current_measure = args[1]
            current_energy = current_measure.get_main_energy()
            # save the variables
            self.times.append(int((datetime.datetime.now()
                                   - self.init_time).total_seconds()))
            self.steps.append(sect[0])
            self.energies = np.append(self.energies, current_energy)
            if sect == [2]:
                # The dual-gap is repeated since it can only be computed in
                # the insertion step
                self.dual_gaps = np.append(self.dual_gaps, self.dual_gaps[-1])
            else:
                # We append a NaN placeholder, this will be modified after
                # computing the solution of the insertion step
                self.dual_gaps = np.append(self.dual_gaps, np.nan)
            self.number_elements.append(len(current_measure.curves))
            # print status
            steptext = ['insertion', 'sliding']
            text_struct = '* Starting {}'
            text = text_struct.format(steptext[sect[0]-1])
            self.printing(text, current_energy)
            if not(num_iter == 1 and sect[0] == 1):
                # plot the results
                self.generate_plots()
                if sect[0] == 1:
                    num_iter = num_iter - 1
                # Save current solution
                subsubtext = steptext[np.mod(sect[0], 2)]
                text_file = '{}/iter_{:03d}_{}'.format(temp, num_iter,
                                                       subsubtext)
                current_measure.animate(filename=text_file, show=False)
                _ = current_measure.draw()
                plt.title('Current solution')
                plt.savefig(text_file+'.pdf')
                plt.close()
                self.save_variables(current_measure, num_iter,
                                    subfilename=subsubtext)
        if sect == [1, 0, 3]:
            # [1,0,3]
            if self.aux is None:
                i = args[0]
            else:
                self.aux += 1
                i = self.aux
            energy_curve = args[1]
            stepsize = args[2]
            text_structure = '* * * * gradient iter #{:04d}, step3-energy ' + \
                             '{:.7E}, stepsize {:.2E}'
            text = text_structure.format(i, energy_curve, stepsize)
            self.printing(text, end="", init='\r')
        if sect == [1, 0, 4]:
            # [1,0,4]
            print("")
        if sect == [1, 1, 0]:
            # [1,1,0]
            text_struct = "* * Execution multistart gradient descent "
            text = text_struct
            self.printing(text)
        if sect == [1, 1, 1]:
            # [1,1,1]
            tries = args[0]
            stationary_curves = args[1]
            min_attempts = config.insertion_max_restarts
            self.aux = 0
            text_struct_1 = '* * * Descend attempt {:02d} of {:02d},'
            text_struct_2 = 'currently {:02d} minima'
            text_struct = text_struct_1 + text_struct_2
            text = text_struct.format(tries, min_attempts,
                                      len(stationary_curves))
            self.printing(text)
        if sect == [1, 1, 1, 1]:
            # [1,1,1,1]
            energy = args[0]
            text_struct = '* * * * * Inserted random curve with energy {:.3E}'
            text = text_struct.format(energy)
            self.printing(text)
        if sect == [1, 1, 1, 2]:
            # [1,1,1,2]
            considered_times = args[0]
            text_struct = '* * * * * Discarded random curve insertion with' + \
                          ' {:02d} nodes'
            text = text_struct.format(len(considered_times))
            self.printing(text)
        if sect == [1, 1, 2]:
            # [1,1,2]
            text = '* * * * * Checking if close to set of stationary curves'
            self.printing(text)
        if sect == [1, 1, 3]:
            # [1,1,3]
            self.aux = 0
            text = '* * * * * * Close to set of stationary curves, discarding'
            self.printing(text)
        if sect == [1, 1, 3, 1]:
            # [1,1,3,1]
            self.aux = 0
            text = '* * * * * * Curve grew too long, discarding'
            self.printing(text)
        if sect == [1, 1, 4]:
            # [1,1,4]
            self.aux = 0
            new_curve_energy = args[0]
            min_energy = args[1]
            soft_max_iter = config.multistart_descent_soft_max_iter
            text_struct1 = '* * * * * * Not promising long itertaion ' + \
                           '(>{:04d}), discarded.'
            text_struct2 = '* * * * * * * Candidate energy {:.3E},' + \
                           'best energy {:.3E}.'
            text1 = text_struct1.format(soft_max_iter)
            text2 = text_struct2.format(new_curve_energy, min_energy)
            self.printing(text1)
            self.printing(text2)
        if sect == [1, 1, 5]:
            # [1,1,5]
            descent_max_iter = config.multistart_descent_max_iter
            text_struct = '* * * * * * Reached maximum ({:04d}) number of ' + \
                          'allowed iterations. Added to curve set'
            text = text_struct.format(descent_max_iter)
            self.printing(text)
        if sect == [1, 1, 7]:
            # [1,1,7]
            stationary_curves = args[0]
            text_struct = '* * * * * * Found a new stationary curve. ' + \
                          'There are {:02d} now'
            text = text_struct.format(len(stationary_curves))
            self.printing(text)
        if sect == [1, 2, 0]:
            # [1,2,0]
            text = '* * Adding candidate curve to current measure'
            self.printing(text)
            # Plotting the stationary curves
            stationary_curves = args[0]
            energy_curves = np.array(args[1])
            almost_normalized = energy_curves - min(energy_curves)
            if max(almost_normalized) <= 1e-10:
                normalized_energies = np.zeros(np.shape(almost_normalized))
            else:
                normalized_energies = almost_normalized/max(almost_normalized)
            cmap = mpl.cm.get_cmap('brg')
            norm = mpl.colors.Normalize(vmin=min(energy_curves),
                                        vmax=max(energy_curves))
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig, ax = plt.subplots()
            for curve, energy in zip(reversed(stationary_curves),
                                     reversed(normalized_energies)):
                _ = curve.draw(ax=ax, color=cmap(energy)[0:3])
            plt.colorbar(sm)
            plt.title('Found {} local minima'.format(len(stationary_curves)))
            fig.suptitle('iter {:03d} stationary curves'.format(self.iter))
            filename = "{}/iter_{:03d}_insertion_stationary_points.pdf"
            fig.savefig(filename.format(temp, self.iter))
            plt.close()
            # To save text values of the considered variables
            f = open("{}/curve_energies.txt".format(temp), "a+")
            f.write("Iteration {:03d}\n".format(self.iter))
            f.write("    energy curves\n")
            for energy in energy_curves:
                f.write("        "+str(energy)+"\n")
            f.write("    normalized energies\n")
            for n_energy in normalized_energies:
                f.write("        "+str(n_energy)+"\n")
            f.close()
        if sect == [1, 2, 1]:
            # [1,2,1]
            text = '* * * Execution weight optimization step'
            self.printing(text)
        if sect == [1, 2, 2]:
            # [1,2,2]
            coefficients = args[0]
            # Printing a with elements with scientific notation
            pretty_array = '[{:.2E}'
            for i in range(len(coefficients)-1):
                pretty_array += ', {:.2E}'
            pretty_array += ']'
            text_struct = '* * * coefficients: '+pretty_array
            text = text_struct.format(*coefficients)
            self.printing(text)
        if sect == [1, 2, 3]:
            # [1,2,3]
            candidate_energy = args[0]
            current_energy = args[1]
            text_1 = '* * * * Curve candidate rejected'
            text_2 = '* * * * * Current energy {:.2E}'
            text_3 = '* * * * * Candidate energy {:.2E}'
            self.printing(text_1)
            self.printing(text_2.format(current_energy))
            self.printing(text_3.format(candidate_energy))
        if sect == [1, 2, 4]:
            # [1,2,4]
            text = '* * * * dual gap below input threshold {:.2E}'
            self.printing(text.format(config.insertion_eps))
            text2 = ' The algorithm finished its execution '
            self.printing(text2)
        if sect == [1, 2, 5]:
            # [1,2,5]
            dual_gap = args[0]
            self.dual_gaps[-1] = dual_gap
            text_struct_1 = '* * * Dual gap {:.2E}'
            text_1 = text_struct_1.format(dual_gap)
            self.printing(text_1)
        if sect == [2, 0, 0]:
            # [2,0,0]
            nearby_index = args[0]
            text_struct = "* * {:2d} candidates to merge"
            text = text_struct.format(len(nearby_index))
            self.printing(text)
        if sect == [2, 0, 1]:
            # [2,0,1]
            new_energy = args[0]
            text = '* * * Successful merging, energy decreased, repeat'
            self.printing(text, new_energy)
        if sect == [2, 0, 2]:
            # [2,0,2]
            text = '* * * Unsuccessful merging, energy mantained'
            self.printing(text)
        if sect == [3, 0, 0]:
            # [3,0,0]
            new_measure = args[0]
            stepsize = args[1]
            iters = args[2]
            current_energy = new_measure.get_main_energy()
            text_struct = '* * gradient iter #{:03d}, stepsize {:.2E}'
            text = text_struct.format(iters, stepsize)
            self.printing(text, current_energy, init='\r', end='')
        if sect == [3, 0, 1]:
            # [3,0,0]
            print('')

    def printing(self, text, *args, init='', end="\n"):
        time_diff = datetime.datetime.now() - self.init_time
        total_seconds = round(time_diff.total_seconds())
        hours = total_seconds // 3600
        total_seconds = total_seconds - hours*3600
        minutes = total_seconds // 60
        seconds = total_seconds - minutes*60
        diff_str = '{:03d}h:{:02d}m:{:02d}s'.format(hours, minutes, seconds)
        if len(args) == 1:
            current_energy = args[0]
            c_e = '{:.7E}'.format(current_energy)
        else:
            c_e = '-------------'
        prepend_string = '['+diff_str+'] '+'current energy:'+c_e+' '
        print(init+prepend_string + text, end=end)
        # test if the log file is too big <+todo+> smart logging.
        if config.log_output:
            if os.path.isfile('{}/log.txt'.format(config.results_folder)):
                f = open('{}/log.txt'.format(config.results_folder), 'rb')
                f_size = sum(1 for i in f)
                f.close()
                if f_size > config.log_maximal_line_size:
                    os.remove('{}/log.txt'.format(config.results_folder))
            self.logtext += init+prepend_string+text + '\n'
            self.logcounter += 1
            if self.logcounter % config.save_output_each_N == 1:
                # Write to file
                f = open('{}/log.txt'.format(config.results_folder), 'a')
                f.write(self.logtext)
                f.close()
                # restart the logtext
                self.logtext = ''

    def time_string(self):
        now = datetime.datetime.now()
        return now.strftime("%d/%m/%Y %H:%M:%S")

    def save_variables(self, current_measure, num_iter, subfilename='', other=None):
        save_dictionary = {
            "current_measure": current_measure,
            "energies": self.energies,
            "steps": self.steps,
            "times": self.times,
            "number_elements": self.number_elements,
            "dual_gaps": self.dual_gaps,
            "other": other
        }
        temp = config.results_folder
        filename = '{}/iter_{:03d}_{}_saved_variables.pickle'
        pickling_on = open(filename.format(temp, num_iter, subfilename), 'wb')
        pickle.dump(save_dictionary, pickling_on)
        pickling_on.close()

    def plotitty(self, data, filename, log=False, start_iter=0, title=None):
        temp = config.results_folder
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
        scattersize = 8
        time = self.times[start_iter:]
        data2 = data[start_iter:]
        if not log:
            ax1.plot(time, data2)
        else:
            ax1.semilogy(time, data2)
        ax1.set_xlabel('time')
        steps = self.steps
        # steps tagging
        insertion_index = [i-start_iter for i in range(start_iter, len(steps))
                           if steps[i] == 1]
        sliding_index = [i-start_iter for i in range(start_iter, len(steps))
                         if steps[i] == 2]
        ax1.scatter([time[i] for i in insertion_index],
                    [data2[i] for i in insertion_index], c='r', s=scattersize)
        ax1.scatter([time[i] for i in sliding_index],
                    [data2[i] for i in sliding_index], c='k', s=scattersize)
        ax1.legend(['', 'insertion step', 'sliding step'])
        if not log:
            ax2.plot(data2)
        else:
            ax2.semilogy(np.arange(len(time)), data2)
        ax2.scatter(insertion_index,
                    [data2[i] for i in insertion_index], c='r', s=scattersize)
        ax2.scatter(sliding_index,
                    [data2[i] for i in sliding_index], c='k', s=scattersize)
        ax2.legend(['', 'insertion step', 'sliding step'])
        ax2.set_xlabel('steps')
        if not title:
            fig.suptitle(filename)
        else:
            fig.suptitle(title)
        fig.savefig("{}/{}.pdf".format(temp, filename))
        plt.close()

    def generate_plots(self):
        self.plotitty(self.number_elements, "number_elements")
        self.plotitty(self.energies - self.energies[-1], "energies", log=True,
                      title="end value = "+str(self.energies[-1]))
        # remember that the dual gap is measured at every insertion step.
        self.plotitty(self.dual_gaps, "dual gaps", log=True,
                      title="end value = "+str(self.dual_gaps[-1]))

    def store_parameters(self, T, sampling_method, sampling_method_arguments):
        dic = {'T': T,
               'sampling_method': sampling_method,
               'sampling_method_arguments': sampling_method_arguments}
        filename = '{}/parameters.pickle'.format(config.results_folder)
        with open(filename, 'wb') as f:
            pickle.dump(dic, f)

    def log_config(self, filename):
        config.self_pickle(filename)



