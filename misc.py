#A module that contains some technical pieces of code

import math
import numpy as np
import itertools as it
import matplotlib as mpl
import matplotlib.pyplot as plt
from  matplotlib.collections import LineCollection
from matplotlib import animation

import copy
import time
import datetime
import config

class Animate(object):
    #an object made to animate a measure class and declutter it. 
    # based on matplotlib.animation.FuncAnimation 
    def __init__(self, measure, **kwargs):
        default_parameters = {
            'frames': 51,
            'filename':None,
            'show': False
        }
        # Incorporate the input keyworded values
        for key, val in kwargs.items():
            if key in default_parameters:
                default_parameters[key] = val
            else:
                raise KeyError(
                   'The given keyworded argument «{}» is not valid'.format(key))
        # Assign the respective variables
        varnames = ['frames', 'filename', 'show']
        frames, filename, show = [default_parameters[n] for n in varnames]
        # 
        measure.reorder()
        # Define the colors, these depends on the intensities
        total_intensities = measure.intensities/measure.energies
        colors = plt.cm.brg(np.array(total_intensities)/max(total_intensities))
        # Get the family of segments and times
        segments = []
        times = []
        for i in range(len(measure.intensities)):
            supsamp_t, supsamp_x = supersample(measure.curves[i],
                                                    max_jump = 0.01)
            # Get segments and use as time the last part of each segment
            new_times = supsamp_t[1:]
            new_segt = [ [supsamp_x[j], supsamp_x[j+1]]
                                            for j in range(len(supsamp_x)-1)]
            segments.append(new_segt)
            times.append(new_times)
        # Attribute definitions
        self.frames = frames
        # # extended frames to freeze the last image in animations.
        self.frames_ext = int(np.round(1.1)*frames)
        self.filename = filename
        self.segments = segments
        self.times = times
        self.colors = colors.copy()
        self.fig, self.ax = plt.subplots()
        self.lc = LineCollection([])
        self.ax.set_ylim((0,1))
        self.ax.set_xlim((0,1))
        self.text = self.ax.text(0,0,0)
        self.text.set_position((0.5, 1.01))
        self.ax.add_collection(self.lc)
        self.head_ref = []
        self.meas = measure
        self.show = show
        # For colorbar
        norm = mpl.colors.Normalize(vmin=0,vmax=max(measure.intensities))
        cmap = plt.get_cmap('brg',100)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        self.fig.colorbar(sm, ticks=np.linspace(0, max(measure.intensities),9))

    def animate(self, i):
        if i <=self.frames:
            t = i/(self.frames-1)
            new_segments, alpha_colors = self.subsegment(t)
            # It is required to input the segments and colors as a single list
            new_segt = []
            alph_col = []
            for j in range(len(new_segments)):
                # j indexes the number of curves there are
                new_segt.extend(new_segments[j])
                alph_col.extend(alpha_colors[j])
            self.lc.set_segments(new_segt)
            self.lc.set_color(alph_col)
            # If this is not the first iteration and there are no segments
            if new_segments != []:
                for head_refs in self.head_ref:
                    head_refs.remove()
                self.head_ref = []
                t = min(t,1)
                for j in range(len(self.meas.curves)):
                    # Here the heads of the curves are drawn
                    considered_marker = mpl.markers.MarkerStyle(marker='x')
                    if len(self.meas.curves) > 1:
                        considered_marker._transform = \
                        considered_marker.get_transform().rotate_deg(
                            j/(len(self.meas.curves)-1)*80)
                    curv = self.meas.curves[j]
                    self.head_ref.append(self.ax.scatter(curv.eval(t)[0,0],
                                     curv.eval(t)[0,1],
                                     c=np.array(self.colors[j]).reshape(1,-1),
                                     s = 60, marker = considered_marker,
                                     lw = 0.8))
        self.text.set_text('time '+str(np.round(min(i/self.frames,1),2)))
        return self.lc,

    def start(self):
        self.anim = animation.FuncAnimation(self.fig, self.animate,
                                            frames=int(np.ceil(1.1*self.frames)),
                                            interval=40, blit=False, repeat =
                                            True)

    def draw(self):
        self.start()
        if self.filename is not None:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
            self.anim.save(self.filename + '.mp4', writer = writer, dpi = 200)
        plt.show(block=False)
        if self.show==False:
            plt.close()

    def subsegment(self,t):
        #To grab the subsegments, with their colors that has times less than t
        new_segments = []
        new_times = []
        for i in range(len(self.segments)):
            # i represented a curve index
            # j represents the respective segment
            new_segments.append([ self.segments[i][j] for j in
                        range(len(self.segments[i])) if self.times[i][j] <=
                                 t+1e-5 ])
            new_times.append([time for time in self.times[i] if time <= t+
                              1e-5])
        alpha_colors = self.alpha_channel(self.colors, new_times, t)
        return new_segments, alpha_colors


    def alpha_channel(self, colors, new_times, t):
        #How the colors fade for the inserted segments
        lowest_alpha = 0.2
        head_length = 0.1
        power = 4
        alpha_colors = []
        for i in range(len(new_times)):
            if t <= head_length:
                alpha_colors.append([colors[i] for j in
                                     range(len(new_times[i]))])
            else:
                alpha_profile = lambda s: lowest_alpha +(1-lowest_alpha)/(
                                                1-head_length)**power*s**power
                alpha_colors.append([np.append(colors[i][:-1],
                    np.minimum(1,alpha_profile(1-t+new_t)))
                    for new_t in new_times[i]])
        return alpha_colors

def animate_dual_variable(w_t, measure,
                   resolution = 0.01, filename = None, show = True):
    # w_t is a dual variable instance
    # measure is a measure class objects
    # since outside these samples, the dual variable is not defined,
    # the number of frames is not explicited.
    frames = config.T
    # Find the min/max of w_t, for color scheme
    vals_max = []
    vals_min = []
    evals = []
    for t in range(config.T):
        evals.append(w_t.grid_evaluate(t)[0])
        vals_max.append(np.max(evals[-1]))
        vals_min.append(np.min(evals[-1]))
    val_max = max(vals_max)
    val_min = min(vals_min)
    # Persistent elements of the animation
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal', autoscale_on=False,
                             xlim =(0,1), ylim = (0,1))
    line_collection = LineCollection([])
    img = plt.imshow(evals[0], extent=[0,1,0,1], vmin=val_min, vmax=val_max,
                     origin='lower', cmap='RdGy')
    ax.add_collection(line_collection)
    # Create colorbar
    norm = mpl.colors.Normalize(vmin = val_min, vmax = val_max)
    cmap = plt.get_cmap('RdGy', 100)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ticks=np.linspace(val_min, val_max,9))

    def init():
        # Initialize animation
        img.set_data(evals[0])
        line_collection.set_segments([])
        return  img,line_collection,

    def animate(i):
        t = i/frames
        img.set_data(evals[i])
        total_segments = []
        total_colors = []
        if measure != None:
            for curv in measure.curves:
                _, (segments, colors) = curv.draw(tf=t, plot=False)
                total_segments.extend(segments)
                total_colors.extend(colors)
            line_collection.set_segments(total_segments)
            line_collection.set_color(total_colors)
        return  img,line_collection,

    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=40,
                                  blit=True, init_func = init)
    if filename is not None:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(filename + '.mp4', writer = writer, dpi = 200)
    plt.show(block=False)
    import code; code.interact(local=dict(globals(), **locals()))
    if show==False:
        plt.close()


def supersample(curve, max_jump = 0.01):
    #Given a family of spatial points and their respective times, returns
    #a super sampled version of them, following a linear path between the nodes
    samples = len(curve.t)-1
    supersampl_t = []
    supersampl_x = []
    for i in range(samples):
        ti = curve.t[i]
        tf = curve.t[i+1]
        xi = curve.x[i]
        xf = curve.x[i+1]
        diff = np.linalg.norm(xf-xi)
        if diff > max_jump:
           N_samp = math.ceil(diff/max_jump)
           new_t = np.linspace(ti,tf, N_samp)[0:-1]
           new_x = [curve.eval(t)[0] for t in new_t]
           supersampl_t = np.append(supersampl_t, new_t)
           supersampl_x.extend(new_x)
        else:
           supersampl_t = np.append(supersampl_t, ti)
           supersampl_x.extend([xi])
    supersampl_x.extend([curve.x[-1]])
    supersampl_t = np.append(supersampl_t, 1)
    return supersampl_t, supersampl_x

def get_periodic_segments(time, space):
    # Space is a list of 2-dimensional tuples
    return  time[1:], [ [space[j], space[j+1]] for j in range(len(space)-1)]

def cut_off(s,h):
    # A one dimentional cut-off function that is twice differentiable, monoto-
    # nous and fast to compute. 
    # Input: s in Nx1 numpy array, h > 0 the width of the transition interval
    #        from 0 to 1.
    # Output: a Nx1 numpy array evaluating the 1-D cutoff along the input s.
    transition = lambda s: 10*s**3 - 15*s**4 + 6*s**5
    val = np.zeros(s.shape)
    for i in range(len(s)):
        if s[i]>= h and s[i]<= 1-h:
            val[i]=1
        elif s[i]<h and s[i]>= 0:
            val[i]= transition(s[i]/h)
        elif s[i]>1-h and s[i]<=1:
            val[i]= transition(((1-s[i]))/h)
        else:
            val[i] = 0
    return val

def D_cut_off(s,h):
    # The derivative of the defined cut_off function.
    # Same input and output sizes.
    D_transition = lambda s: 30*s**2 - 60*s**3 + 30*s**4
    val = np.zeros(s.shape)
    for i in range(len(s)):
        if s[i]<h and s[i]>=0:
            val[i]= D_transition(s[i]/h)/h
        elif s[i]<=1 and s[i] >= 1-h:
            val[i]= -D_transition(((1-s[i]))/h)/h
        else:
            val[i]=0
    return val

def plot_2d_time(w_t, total_animation_time = 2):
    # function to plot a two variable function on given times
    # total_animation_time on seconds.
    times = config.time
    # first we scan w_t to get the maximum and minimum values, for colors
    min_val = np.inf
    max_val = -np.inf
    for t_idx in range(len(times)):
        new_data = grid_evaluate(lambda x: w_t(t_idx, x))
        min_val = min(min_val, np.min(new_data))
        max_val = max(max_val, np.max(new_data))
    # Now we start generating the frames
    t_idx = 0
    t = times[0]
    fig, ax = plt.subplots()
    data = grid_evaluate(lambda x: w_t(t_idx,x))
    plot = ax.imshow(data, extent=[0,1,0,1], origin='lower', cmap='RdGy')
    # create colorbar
    cbar = plt.colorbar(plot)
    cbar.set_clim(min_val, max_val)
    plt.title('time t : '+str(t))
    plt.show(block=False)
    plt.pause(total_animation_time/len(times))
    for t_idx in range(1,len(times)):
        new_data = grid_evaluate(lambda x: w_t(t_idx,x))
        plot.set_data(new_data)
        ax.set_title('time t : '+str(times[t_idx]))
        cbar.draw_all()
        plt.draw()
        plt.pause(total_animation_time/len(times))
    plt.close()

def grid_evaluate(w_t, resolution = 0.01):
    # Function to evaluate a function w_t: x \in np.array(1,2) -> R over the 
    # whole grid. 
    # Output: NxN matrix with all the evaluations.
    x = np.linspace(0,1,round(1/resolution))
    y = np.linspace(0,1,round(1/resolution))
    X,Y = np.meshgrid(x,y)
    XY = np.array([ np.array([xx,yy]) for yy,xx in it.product(y,x)])
    return w_t(XY).reshape(X.shape)

def is_inside_domain(x0):
    # To test if the selected point is inside or outside of the domain
    # Input: x0 a 1x2 numpy array
    # Output: boolean
    if x0[0,0] >= 0 and x0[0,0] <= 1 and x0[0,1] >= 0 and x0[0,1] <= 1:
        return True
    else:
        return False

def Archimedian_spiral(t,a,b):
    return np.array([(a+b*t)*np.cos(t), (a+b*t)*np.sin(t)])

def sample_line(num_samples, angle, spacing):
    rotation_mat = np.array([[np.cos(angle), np.sin(angle)],
    [-np.sin(angle), np.cos(angle)]])
    x = [-spacing*(i+1) for i in range(num_samples//2)]
    x.extend([spacing*(i+1) for i in range(num_samples - num_samples//2)])
    horizontal_samps = [ np.array([xx, 0]) for xx in x]
    rot_samples = [samp@rotation_mat for samp in horizontal_samps]
    return rot_samples

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
        if config.log_output == True:
            self.logtext = ''
            self.logcounter = 0
            f = open('{}/log.txt'.format(config.temp_folder),'w')
            f.write('Logging!')
            f.close()

    def status(self, sect, *args):
        temp = config.temp_folder
        if sect == [0]:
            # [0]
            ground_truth = args[0]
            self.printing('Saving the generated ground truth')
            self.save_variables([], subfilename = 'ground_truth', other =
                                ground_truth)
            text_file = '{}/iter_000_ground_truth'.format(temp)
            ground_truth.animate(filename= text_file, show = False)
            _ = ground_truth.draw()
            plt.title('Ground truth')
            plt.savefig('{}/ground_truth.pdf'.format(temp))
            plt.close()
            ## To plot the considered considered frequencies
            #  We erase the duplicated frequencies
            from operators import sampling_method
            samples = sampling_method.copy()
            unique_samps = []
            for samps in samples:
                is_inside = False
                for samps2 in unique_samps:
                    if np.linalg.norm(samps - samps2) < 1e-10:
                        is_inside = True
                        break
                if is_inside == False:
                    unique_samps.append(samps)
            # Get maximums of x and y
            min_x = 0
            max_x = 0
            min_y = 0
            max_y = 0
            for samp in unique_samps:
                x = samp[:,0]
                max_x = max(max(x), max_x)
                min_x = min(min(x), min_x)
                y = samp[:,1]
                max_y = max(max(y), max_y)
                min_y = min(min(y), min_y)
            for idx, samp in enumerate(unique_samps):
                x = samp[:,0]
                y = samp[:,1]
                plt.plot(x,y,'o')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(min_x*1.1, max_x*1.1)
                plt.ylim(min_y*1.1, max_y*1.1)
                plt.title('Sample frequencies {}'.format(idx))
                filename = 'sample_{:02d}'.format(idx)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig("{}/{}.pdf".format(config.temp_folder, filename),
                           bbox_inches='tight', transparent=False)
                plt.close()
        if sect == [1]:
            # [1]
            self.iter += 1
            num_iter = args[0]
            self.current_iter = num_iter
            current_measure = args[1]
            current_energy = current_measure.get_main_energy()
            self.printing('Iteration {:03d}'.format(num_iter), current_energy)
            self.save_variables(current_measure, subfilename = 'step_1_')
        if sect == [1] or sect == [3]:
            # [1], [2], [3]
            num_iter = args[0]
            current_measure = args[1]
            current_energy = current_measure.get_main_energy()
            # save the variables
            self.times.append(int((datetime.datetime.now()
                                   - self.init_time).total_seconds()))
            self.steps.append(sect[0])
            self.energies = np.append(self.energies, current_energy)
            #TOERRASE if sect !=[2] and not(sect == [1] and (num_iter == 1)):
            if sect == [3]:
                # The dual-gap is repeated since it can only be computed in 
                # the insertion step
                self.dual_gaps = np.append(self.dual_gaps, self.dual_gaps[-1])
            else:
                # We append a NaN placeholder
                self.dual_gaps = np.append(self.dual_gaps, np.nan)
            self.number_elements.append(len(current_measure.curves))
            # print status
            steptext = ['step 1 insertion', 'step 2 merging', 'step 3 grad flow']
            text_struct = '* Starting {}'
            text = text_struct.format(steptext[sect[0]-1])
            self.printing(text, current_energy)
            if not(num_iter == 1 and sect[0] == 1):
                # plot the results
                self.generate_plots('iter_{:03d}'.format(num_iter))
                # Save current solution
                subsubtext = steptext[np.mod(sect[0]+1,3)].replace(" ","_")
                if sect[0]==1:
                    text_file = '{}/iter_{:03d}_{}'.format(temp,num_iter-1,
                                                         subsubtext)
                else:
                    text_file = '{}/iter_{:03d}_{}'.format(temp, num_iter,
                                                         subsubtext)
                current_measure.animate(filename= text_file, show = False)
                _ = current_measure.draw()
                plt.title('Current solution')
                plt.savefig(text_file+'.pdf')
                plt.close()
                self.save_variables(current_measure, subfilename=subsubtext)
        if sect == [1,0,3]:
            # [1,0,3]
            if self.aux is None:
                i = args[0]
            else:
                self.aux +=1
                i = self.aux
            energy_curve = args[1]
            stepsize = args[2]
            text_structure = '* * * * gradient iter #{:04d}, step3-energy '+\
                             '{:.7E}, stepsize {:.2E}'
            text = text_structure.format(i, energy_curve, stepsize)
            self.printing(text, end = "", init='\r')
        if sect == [1,0,4]:
            # [1,0,4]
            print("")
        if sect == [1,1,0]:
            # [1,1,0]
            text_struct = "* * Searching better curves via tabu search. "
            text = text_struct
            self.printing(text)
        if sect == [1,1,1]:
            # [1,1,1]
            tries = args[0]
            tabu_curves = args[1]
            min_attempts = config.step3_min_attempts_to_find_better_curve
            self.aux = 0
            text_struct = '* * * Descend attempt {:02d} of {:02d}, currently {:02d} minima'
            text = text_struct.format(tries, min_attempts, len(tabu_curves))
            self.printing(text)
        if sect == [1,1,1,1]:
            # [1,1,1,1]
            considered_times = args[0]
            text_struct = '* * * * * Inserted random curve with {:02d} nodes'
            text = text_struct.format(len(considered_times))
            self.printing(text)
        if sect == [1,1,1,2]:
            # [1,1,1,2]
            considered_times = args[0]
            text_struct='* * * * * Discarded random curve insertion with' + \
                        ' {:02d} nodes'
            text = text_struct.format(len(considered_times))
            self.printing(text)
        if sect == [1,1,2]:
            # [1,1,2]
            text = '* * * * * Checking if close to tabu set'
            self.printing(text)
        if sect == [1,1,3]:
            # [1,1,3]
            self.aux = 0
            text = '* * * * * * Close to tabu set, discarding'
            self.printing(text)
        if sect == [1,1,3,1]:
            # [1,1,3,1]
            self.aux = 0
            text = '* * * * * * Curve grew too long, discarding'
            self.printing(text)
        if sect == [1,1,4]:
            # [1,1,4]
            self.aux = 0
            new_curve_energy = args[0]
            min_energy = args[1]
            soft_max_iter = config.step3_descent_soft_max_iter
            text_struct1 = '* * * * * * Not promising long itertaion (>{:04d})'+\
                           ', discarded.'
            text_struct2 = '* * * * * * * Candidate energy {:.3E},'+\
                           'best energy {:.3E}.'
            text1 = text_struct1.format(soft_max_iter)
            text2 = text_struct2.format(new_curve_energy, min_energy)
            self.printing(text1)
            self.printing(text2)
        if sect == [1,1,5]:
            # [1,1,5]
            descent_max_iter = config.step3_descent_max_iter
            text_struct = '* * * * * * Reached maximum ({:04d}) number of '+\
                          'allowed iterations. Added to tabu set'
            text = text_struct.format(descent_max_iter)
            self.printing(text)
        if sect == [1,1,6]:
            # [1,1,6]
            energy_curves = np.array(args[0])
            min_energy = np.min(energy_curves)
            argmin_energy = np.argmin(energy_curves)
            min_threshold = args[1]
            text_struct = '* * * The curve #{:02d} is candidate'+\
                          ' with {:.2E} step3-energy below required.'
            text = text_struct.format(argmin_energy, min_threshold - min_energy)
            self.printing(text)
        if sect == [1,1,7]:
            # [1,1,7]
            tabu_curves = args[0]
            text_struct = '* * * * * * Found a new stationary curve. There are'+\
                          ' {:02d} now'
            text = text_struct.format(len(tabu_curves))
            self.printing(text)
        if sect == [1,2,0]:
            # [1,2,0]
            text = '* * Adding candidate curve to current measure'
            self.printing(text)
            # Plotting the tabu curves
            tabu_curves = args[0]
            energy_curves = np.array(args[1])
            almost_normalized = energy_curves - min(energy_curves)
            if max(almost_normalized) <= 1e-10:
                normalized_energies = np.zeros(np.shape(almost_normalized))
            else:
                normalized_energies = almost_normalized/max(almost_normalized)
            cmap = mpl.cm.get_cmap('brg')
            norm = mpl.colors.Normalize(vmin = min(energy_curves),
                                        vmax = max(energy_curves))
            sm = plt.cm.ScalarMappable(cmap = cmap, norm= norm)
            sm.set_array([])
            fig, ax = plt.subplots()
            for curve,energy in zip(reversed(tabu_curves),
                                    reversed(normalized_energies)):
                _ = curve.draw(ax=ax, color = cmap(energy)[0:3] )
            plt.colorbar(sm)
            plt.title('Found {} local minima'.format(len(tabu_curves)))
            fig.suptitle('iter {:03d} tabu curves'.format(self.iter))
            filename="{}/iter_{:03d}_step_1_insertion_tabu_set.pdf"
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
        if sect == [1,2,1]:
            # [1,2,1]
            text = '* * * Optimizing coefficients target measure'
            self.printing(text)
        if sect == [1,2,2]:
            # [1,2,2]
            coefficients = args[0]
            # Printing a with elements with scientific notation
            pretty_array = '[{:.2E}'
            for i in range(len(coefficients)-1):
                pretty_array+= ', {:.2E}'
            pretty_array+= ']'
            text_struct = '* * * coefficients: '+pretty_array
            text = text_struct.format(*coefficients)
            self.printing(text)
        if sect == [1,2,3]:
            # [1,2,3]
            candidate_energy = args[0]
            current_energy = args[1]
            text_1 = '* * * * Curve candidate rejected'
            text_2 = '* * * * * Current energy {:.2E}'.format(current_energy)
            text_3 = '* * * * * Candidate energy {:.2E}'.format(candidate_energy)
            self.printing(text_1)
            self.printing(text_2)
            self.printing(text_3)
        if sect == [1,2,4]:
            # [1,2,4]
            text = '* * * * Curve candidate accepted'
            self.printing(text)
        if sect == [1,2,5]:
            # [1,2,5]
            from optimization import dual_gap as opt_dual_gap
            current_measure = args[0]
            tabu_curves = args[1]
            energies = args[2]
            dual_gap  = opt_dual_gap(current_measure, tabu_curves)
            self.dual_gaps[-1] = dual_gap
            text_struct_1 = '* * * Dual gap {:.2E}'
            text_1 = text_struct_1.format(dual_gap)
            self.printing(text_1)
        if sect == [2,0,0]:
            # [2,0,0]
            nearby_index = args[0]
            text_struct = "* * {:2d} candidates to merge"
            text = text_struct.format(len(nearby_index))
            self.printing(text)
        if sect == [2,0,1]:
            # [2,0,1]
            new_energy = args[0]
            text = '* * * Successful merging, energy decreased, repeat'
            self.printing(text, new_energy)
        if sect == [2,0,2]:
            # [2,0,2]
            text = '* * * Unsuccessful merging, energy mantained'
            self.printing(text)
        if sect == [3,0,0]:
            # [3,0,0]
            new_measure = args[0]
            stepsize = args[1]
            iters = args[2]
            current_energy = new_measure.get_main_energy()
            text_struct = '* * gradient iter #{:03d}, stepsize {:.2E}'
            text = text_struct.format(iters, stepsize)
            self.printing(text, current_energy, init='\r', end ='')
        if sect == [3,0,1]:
            #[3,0,0]
            print('')

    def printing(self,text, *args, init = '', end = "\n"):
        time_diff = datetime.datetime.now() - self.init_time
        total_seconds = round(time_diff.total_seconds())
        hours = total_seconds // 3600
        total_seconds = total_seconds - hours*3600
        minutes = total_seconds //60
        seconds = total_seconds - minutes*60
        diff_str = '{:03d}h:{:02d}m:{:02d}s'.format(hours, minutes, seconds)
        if len(args) == 1:
            current_energy = args[0]
            c_e = '{:.7E}'.format(current_energy)
        else:
            c_e = '-------------'
        prepend_string = '['+diff_str+'] '+'current energy:'+c_e+' '
        print(init+prepend_string + text, end = end)
        if config.log_output==True:
            self.logtext += init+prepend_string+text +'\n'
            self.logcounter += 1
            if self.logcounter % config.save_output_each_N==1:
                # Write to file
                f = open('{}/log.txt'.format(config.temp_folder),'a')
                f.write(self.logtext)
                f.close()
                # restart the logtext
                self.logtext = ''

    def time_string(self):
        now = datetime.datetime.now()
        return now.strftime("%d/%m/%Y %H:%M:%S")

    def save_variables(self, current_measure, subfilename='', other=None):
        save_dictionary = {
        "current_measure": current_measure,
        "energies": self.energies,
        "steps": self.steps,
        "times": self.times,
        "number_elements": self.number_elements,
        "dual_gaps": self.dual_gaps,
        "other": other
        }
        temp = config.temp_folder
        import pickle
        filename = '{}/iter_{:03d}_{}_saved_variables.pickle'
        pickling_on = open(filename.format(temp, self.iter, subfilename), 'wb')
        pickle.dump(save_dictionary, pickling_on)
        pickling_on.close()

    def plotitty(self, data, filename, log=False, start_iter=0, title = None):
        temp = config.temp_folder
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(30,15))
        scattersize = 8
        time = self.times[start_iter:]
        data2  = data[start_iter:]
        if log==False:
            ax1.plot(time, data2)
        else:
            ax1.semilogy(time, data2)
        ax1.set_xlabel('time')
        steps = self.steps
        # steps tagging
        step3_index = [i-start_iter for i in range(start_iter,len(steps))
                       if steps[i]==1]
        step6_index = [i-start_iter for i in range(start_iter,len(steps))
                       if steps[i]==3]
        ax1.scatter([time[i] for i in step3_index],
                   [data2[i] for i in step3_index], c='r', s=scattersize)
        ax1.scatter([time[i] for i in step6_index],
                    [data2[i] for i in step6_index], c='k', s=scattersize)
        ax1.legend(['','insertion step','merging step','flow step'])
        if log == False:
            ax2.plot(data2)
        else:
            ax2.semilogy(np.arange(len(time)),data2)
        ax2.scatter([i for i in step3_index],
                    [data2[i] for i in step3_index], c='r', s=scattersize)
        ax2.scatter([i for i in step6_index],
                    [data2[i] for i in step6_index], c='k', s=scattersize)
        ax2.legend(['','insertion step','flow step'])
        ax2.set_xlabel('steps')
        if title == None:
            fig.suptitle(filename)
        else:
            fig.suptitle(title)
        fig.savefig("{}/{}.pdf".format(temp, filename))
        plt.close()

    def generate_plots(self, filename):
        self.plotitty(self.number_elements, "number_elements")
        self.plotitty(self.energies - self.energies[-1], "energies", log=True,
                 title="end value = "+str(self.energies[-1]))
        self.plotitty(self.dual_gaps, "dual gaps", log=True,
                 title="end value = "+str(self.dual_gaps[-1]))

    def store_parameters(self, T, sampling_method, sampling_method_arguments):
       import pickle
       dic = {'T': T,
              'sampling_method': sampling_method,
              'sampling_method_arguments': sampling_method_arguments}
       with open('{}/parameters.pickle'.format(config.temp_folder), 'wb') as f:
           pickle.dump(dic, f)


