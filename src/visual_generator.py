import numpy as np
import itertools as it
import matplotlib as mpl
import matplotlib.pyplot as plt
from  matplotlib.collections import LineCollection
from matplotlib import animation

import copy
import logging
import curves
import operators as op
import time
import datetime
import config
import misc
import os

import pickle

# Script to generate all the required images "all at once"


def draw(curve, ax = None, **kwargs):
    default_variables = {
        'tf':1,
        # Line related variables
        'color': [0.0, 0.5,1.0],
        'thickness': 1,
        'max_jump':0.01,
        # Marker related variables
        'marker_times':[0, 0.25, 0.5, 0.75, 1],
        'marker_color':[0,0,0],
        'marker_size':20,
        'marker_type':'->',
        'head_on':True,
        # Annotation related values
        'annotate': False,
        'annotate_offset':[(-20,20)],
        # Other variables
        'output_segments':False,
        'saveas':None

    }
    for key, value in kwargs.items():
        default_variables[key] = value
    var = default_variables
    #First we supersample the whole curve such that there are not jumps higher
    #than a particular value
    supersampl_t, supersampl_x = misc.supersample(curve,
                                                  max_jump = var['max_jump'])
    # Drop all time samples after tf
    value_at_tf = curve.eval(var['tf'])
    index_tf = np.argmax(supersampl_t>=var['tf'])
    supersampl_x = supersampl_x[:index_tf]
    supersampl_t = supersampl_t[:index_tf]
    supersampl_t = np.append(supersampl_t,var['tf'])
    supersampl_x.append(value_at_tf.reshape(-1))
    #Reduce the set of points and times to segments and times, restricted 
    #to the periodic domain.
    _, segments = misc.get_periodic_segments(supersampl_t, supersampl_x)
    #Use the LineCollection class to print using segments and to assign 
    #transparency or colors to each segment
    line_segments = LineCollection(segments, linewidth=var['thickness'])
    # set color 
    lowest_alpha = 0.2
    color = np.array(var['color'])
    rgb_color = np.ones((len(segments),4))
    rgb_color[:,0:3] = color
    rgb_color[:,3] = np.linspace(lowest_alpha,1,len(segments))
    line_segments.set_color(rgb_color)
    start_color = np.zeros((1,4))
    start_color[:,0:3] = color
    if len(segments)<= 1:
        start_color[:,3] = 1
    else:
        start_color[:,3] = lowest_alpha
    #plotting
    ax = ax or plt.gca()
    ax.add_collection(line_segments)
    ax.scatter(curve.x[0,0], curve.x[0,1], c=start_color, marker='x', s=0.4)
    # Markers inclusion
    marker_alpha = lambda t: np.min([0.2 + 0.8*t/var['tf'],1])
    # Marking head
    if var['head_on']==True:
        tf = var['tf']
        marker_pos = curve.eval(tf)
        if tf>= 1-0.01:
            marker_pos_2  = 2*curve.eval(tf)- curve.eval(tf-0.01)
        else :
            marker_pos_2 = curve.eval(tf+0.01)
        ax.annotate('', (marker_pos_2[0,0], marker_pos_2[0,1]),
                    xytext = (marker_pos[0,0], marker_pos[0,1]),
                    arrowprops = {'arrowstyle':'->', 'alpha':marker_alpha(tf),
                                  'color':np.array(var['marker_color'])})
    for idx_t, t in enumerate(var['marker_times']):
        if t<= var['tf']:
            marker_color = np.append(var['marker_color'], marker_alpha(t)).reshape(1,-1)
            marker_pos = curve.eval(t)
            if t==1:
                marker_pos_2  = 2*curve.eval(t)- curve.eval(t-0.01)
            else:
                marker_pos_2 = curve.eval(t+0.01)
            ax.annotate('', (marker_pos_2[0,0], marker_pos_2[0,1]),
                        xytext = (marker_pos[0,0], marker_pos[0,1]),
                        arrowprops = {'arrowstyle':'->', 'alpha':marker_alpha(t),
                                      'color':np.array(var['marker_color'])})
            if var['annotate']==True:
                if len(var['annotate_offset'])==1:
                    a_offset = [ var['annotate_offset'][0] for i in
                                range(len(var['marker_times']))]
                else:
                    a_offset = var['annotate_offset']
                ax.annotate('t='+str(t), (marker_pos[0,0], marker_pos[0,1]),
                           xytext= a_offset[idx_t], textcoords='offset points',
                            arrowprops={'arrowstyle':var['marker_type'],
                                        'alpha':marker_alpha(t)},
                            alpha=marker_alpha(t),
                            size = 10)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    if var['output_segments']==False:
        return ax
    else:
        return ax, (segments, rgb_color)

def generate_frames(measure, subfilename, **kwargs):
    default_variables = {
        'times':np.linspace(0,1,11),
        'cutoff':False,
        'cutoff_val':0.1,
        'gray_cutoff':True,

    }
    for key, value in kwargs.items():
        default_variables[key] = value
    var = default_variables
    times = var['times']
    times[0]=0.001
    for idx, t in enumerate(times):
        fig = plt.figure()
        ax = fig.add_subplot(111,aspect='equal', autoscale_on=False,
                                 xlim =(0,1), ylim = (0,1))
        line_collection = LineCollection([])
        total_segments = []
        total_colors = []
        for weight, curvs in zip(measure.intensities, measure.curves):
            if var['cutoff_val']<= weight:
                draw(curvs, ax=ax,tf=t, thickness = 2)
            elif var['cutoff']==False:
                if var['gray_cutoff']==True:
                    draw(curvs, ax=ax,tf=t, thickness = 1, color=[0.2,0.2,0.2])
                else:
                    draw(curvs, ax=ax,tf=t, thickness = 1)
        plt.title('t = {:.2f}'.format(t))
        #if t == times[-1]:
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim([0,1])
        plt.ylim([0,1])
        #plt.xlabel('x')
        #plt.ylabel('y')
        plt.savefig('{}_t{:02d}.png'.format(subfilename,idx),
            bbox_inches='tight', transparent=False)
        plt.close()

if __name__=='__main__':
    # Generate plot of the ground truth with marks on specific times and 
    # arrows pointing to these points.

    # Here we gather the subfolder names and their respective last iteration.
    dataset_names = ['simple_example','complex_example','crossing_example']
    last_iter = ['013', '019', '014']
    alphas = ['0.1','0.5',' 0.1']
    betas = ['0.1',' 0.1', '0.1']
    converged = ['early stopped', 'converged', 'early stopped']

    # desired animation time samples
    frame_samples = np.linspace(0,1,41)

    # Cut-off value for artifactless videos
    cutoff_val = 0.1

    for ind, (it_num, foldername) in enumerate(zip(last_iter, dataset_names)):
        # Load the ground truth
        tot_folder = ('datasets/' + foldername +
                      '/iter_000_ground_truth_saved_variables.pickle')
        ground_truth = pickle.load(open(tot_folder, 'rb'))['other']
        # Create dual variables
        f_t = lambda t: op.K_t_star(t,ground_truth)
        w_t = lambda t,x : -op.K_t(t,f_t(t))(x)
        # Load last iteration
        cur_folder = ('datasets/' + foldername + '/iter_' + it_num +
                      '_step_3_grad_flow_saved_variables.pickle')
        current_measure = pickle.load(open(cur_folder, 'rb'))['current_measure']
        # Load convergence relate variables
        energies = pickle.load(open(cur_folder, 'rb'))['energies']
        steps = pickle.load(open(cur_folder, 'rb'))['steps']
        sav_fold = 'visualizations/'+foldername

        # General info to text data
        if 1:
            filename = sav_fold+'/simulation_info.txt'
            text = open(filename, 'w+')
            text.write('# Relevant parameters for this simulation\n\n')
            text.write('alpha = {}\n'.format(alphas[ind]))
            text.write('beta  = {}\n\n'.format(betas[ind]))
            text.write('Did it converged or it was early stopped?\n {}\n\n'.format(
                converged[ind]))
            text.write('Number of iterations: {}\n\n'.format(it_num))
            text.write('Final energy: {}\n\n'.format(energies[-1]))



        # Video of ground truth
        if 1:
            sub_fold = sav_fold +'/ground_truth'
            os.system('rm -r {}'.format(sub_fold))
            os.system('mkdir {}'.format(sub_fold))
            filenam = sub_fold+'/ground_truth'
            times = frame_samples
            times[0]=0.001
            generate_frames(ground_truth, filenam, times = times)

        # Video of current solution with artifacts
        if 1:
            sub_fold = sav_fold +'/final_solution_full'
            os.system('rm -r {}'.format(sub_fold))
            os.system('mkdir {}'.format(sub_fold))
            filenam = sub_fold+'/final_solution_full'
            times = frame_samples
            times[0]=0.001
            generate_frames(current_measure, filenam, times = times)

        # Video of the the current solution, without artifacts
        if 1:
            sub_fold = sav_fold +'/final_solution_artifactless'
            os.system('rm -r {}'.format(sub_fold))
            os.system('mkdir {}'.format(sub_fold))
            filenam = sub_fold+'/final_solution_artifactless'
            times = frame_samples
            times[0]=0.001
            generate_frames(current_measure, filenam, times = times,
                           cutoff=True, cutoff_val = cutoff_val)
            # write to a text file the values below cut-off and above
            over_cutoff_weights = []
            below_cutoff_weights = []
            for weights in current_measure.intensities:
                if weights >= cutoff_val:
                    over_cutoff_weights.append(weights)
                else:
                    below_cutoff_weights.append(weights)
            text = open(sub_fold+'/weights.txt', 'w+')
            text.write('Not cut-offed weights \n\n')
            for idx, weights in enumerate(over_cutoff_weights):
                text.write('Weight {:02d}: {:.3f} \n'.format(idx, weights))
            text.write('\n cut-off value: {:.2f} \n'.format(cutoff_val))
            for idx, weights in enumerate(below_cutoff_weights):
                text.write('Weight {:02d}: {:.3f} \n'.format(idx, weights))
            text.close()

        # Static plot comparin the obtained reconstruction with the ground truth.
        if 1:
            # Full_reconstruction_comparison
            good_curves = []
            bad_curves = []
            bad_intensities = []
            intensities = []
            for curve, inte in zip(current_measure.curves,
                                   current_measure.intensities):
                if inte>cutoff_val:
                    good_curves.append(curve)
                    intensities.append(inte)
                else:
                    bad_curves.append(curve)
                    bad_intensities.append(inte)
            new_meas = curves.measure()
            for curv, inte in zip(good_curves,intensities):
                new_meas.add(curv,inte)
            fig, ax = plt.subplots()
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(0)
            # plotting each curve
            for curv in ground_truth.curves:
                draw(curv, ax=ax, thickness = 2, marker_color=[0,0,1,1])
            for curv in new_meas.curves:
                draw(curv, ax = ax, color =[1,0,0])
            from matplotlib.lines import Line2D
            proxies = [Line2D([0,1],[0,1], color=np.array([1,0,0,1]), linewidth=1),
                       Line2D([0,1],[0,1], color=np.array([0,0,1,1]), linewidth=2)]
            ax.legend(proxies, ['Reconstruction', 'Ground truth'], loc ='upper left')
            # saving
            filename = sav_fold + '/final_solution_artifactless_comparison.pdf'
            os.system('rm {}'.format(filename))
            plt.savefig(filename, bbox_inches='tight',
                        facecolor=fig.get_facecolor(), edgecolor = 'none')
            plt.close()
            # Plotting with the artifacts
            fig, ax = plt.subplots()
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(0)
            for curv in bad_curves:
                draw(curv, ax=ax, thickness = 0.5, color = [0.2,0.2,0.2],
                     marker_color = [0.2,0.2,0.2,1])
            for curv in ground_truth.curves:
                draw(curv, ax=ax, thickness = 2, marker_color=[0,0,1,1])
            for curv in new_meas.curves:
                draw(curv, ax = ax, color =[1,0,0])
            proxies.append(Line2D([0,1],[0,1], color=np.array([0.2,0.2,0.2,1]), linewidth=0.5))
            ax.legend( proxies, ['Reconstruction', 'Ground truth','Artifacts'],
                     loc = 'upper left')
            plt.gca().set_aspect('equal', adjustable='box')
            # saving
            filename = sav_fold + '/final_solution_full_comparison.pdf'
            os.system('rm {}'.format(filename))
            plt.savefig(filename, bbox_inches='tight',
                        facecolor=fig.get_facecolor(), edgecolor = 'none')
            plt.close()

        # Video of the dual variables at the first iteration.
        if 1:
            sub_fold = sav_fold+'/dual_variable'
            os.system('rm -r {}'.format(sub_fold))
            os.system('mkdir {}'.format(sub_fold))
            filenam = sub_fold+'/dual_variable'
            times = frame_samples
            times[0] = 0.001
            # get all time min/max values for color scaling
            resolution = 0.01
            vals_max = []
            vals_min = []
            for t in np.linspace(0,1,50):
                vals = misc.grid_evaluate(lambda x: w_t(t,x),
                                          resolution=resolution).reshape(-1)
                vals_max.append(max(vals))
                vals_min.append(min(vals))
            val_max = max(vals_max)
            val_min = min(vals_min)
            # Plotting
            for idx, t in enumerate(times):
                fig = plt.figure()
                ax = fig.add_subplot(111,aspect='equal', autoscale_on=False,
                                         xlim =(0,1), ylim = (0,1))
                line_collection = LineCollection([])
                img = plt.imshow(misc.grid_evaluate(lambda x: w_t(t,x), resolution=resolution),
                                extent=[0,1,0,1], vmin=val_min, vmax=val_max,
                                 origin='lower', cmap='RdGy')
                total_segments = []
                total_colors = []
                for curvs in ground_truth.curves:
                    _, (segments, colors) = draw(curvs,tf=t, output_segments=True)
                    total_segments.extend(segments)
                    total_colors.extend(colors)
                line_collection.set_segments(total_segments)
                line_collection.set_color(total_colors)
                ax.add_collection(line_collection)
                plt.title('t = {:.2f}'.format(t))
                #if t == times[-1]:
                plt.colorbar(ticks=np.linspace(val_min, val_max, 5))
                plt.gca().set_aspect('equal', adjustable='box')
                plt.xlim([0,1])
                plt.ylim([0,1])
                #plt.xlabel('x')
                #plt.ylabel('y')
                plt.savefig('{}_t{:02d}.png'.format(filenam, idx),
                    bbox_inches='tight', transparent=False)
                plt.close()

        # Convergence plots
        if 1:
            filename = sav_fold+ '/convergence_plot.pdf'
            os.system('rm {}'.format(filename))
            # Merge the insert with the quadratic step
            new_energies = []
            new_steps = []
            for ener, ste in zip(energies,steps):
                if ste == 2:
                    pass
                else:
                    new_energies.append(ener)
                    new_steps.append(ste)
            fig, ax = plt.subplots()
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(0)
            diff_energies = new_energies-new_energies[-1]
            iterations = np.arange(0,  len(diff_energies)/2, 0.5)
            ax.semilogy(iterations,diff_energies, label='_nolegend_')
            for idx, step in enumerate(new_steps):
                if idx==0:
                    # Start of the error plot, the first value is the initial error.
                    pass
                else:
                    if step == 1:
                        # The before step was the gradient flow
                        ax.scatter(iterations[idx], diff_energies[idx],c =
                                   np.array([[0,0,0]]), marker='x')
                    elif step == 3:
                        # The before step was a insert + optimize one
                        ax.scatter(iterations[idx], diff_energies[idx], c =
                                   np.array([[1,0,0]]), marker='D')
            plt.grid()
            plt.legend(['After insertion + optimization', 'After gradient flow'])
            plt.xlabel('Iterations')
            plt.ylabel('Errors')
            plt.savefig(filename,bbox_inches='tight',
                        facecolor=fig.get_facecolor(), edgecolor ='none')
            plt.close()

        # Plot the Considered sampling points of the Archimedial spire
        if 1:
            filename = sav_fold + '/sampling_points.pdf'
            os.system('rm {}'.format(filename))
            samples = op.samp(0)
            x = samples[:,0]
            y = samples[:,1]
            plt.scatter(x,y, c='r', marker='o')
            plt.xlim([-5,5])
            plt.ylim([-5,5])
            #plt.xlabel('ζ_x')
            #plt.ylabel('ζ_y')
            plt.grid()
            plt.gca().set_aspect('equal', adjustable ='box')
            plt.savefig(filename, bbox_inches = 'tight')
            plt.close()

