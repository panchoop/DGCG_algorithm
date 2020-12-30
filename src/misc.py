"""
Module with Animation helper methods.

Undocumented.
"""
# Standard imports
import os
import math
import itertools as it
import datetime
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import animation
import numpy as np

# Local imports
from . import config

# Global variables

class Animate(object):
    # an object made to animate a measure class and declutter it
    # based on matplotlib.animation.FuncAnimation 
    def __init__(self, measure, **kwargs):
        default_parameters = {
            'frames': 51,
            'filename': None,
            'show': False,
            'block': False,
        }
        # Incorporate the input keyworded values
        for key, val in kwargs.items():
            if key in default_parameters:
                default_parameters[key] = val
            else:
                text = 'The given keyworded argument «{}» is not valid'
                raise KeyError(text.format(key))
        # Assign the respective variables
        varnames = ['frames', 'filename', 'show', 'block']
        frames, filename, show, block = [default_parameters[n]
                                         for n in varnames]
        #
        measure.reorder()
        # Define the colors, these depends on the intensities
        total_intensities = measure.weights/measure._energies
        brg_cmap = plt.cm.get_cmap('brg')
        colors = brg_cmap(np.array(total_intensities)/max(total_intensities))
        # Get the family of segments and times
        segments = []
        times = []
        for i in range(len(measure.weights)):
            supsamp_t, supsamp_x = supersample(measure.curves[i],
                                               max_jump=0.01)
            # Get segments and use as time the last part of each segment
            new_times = supsamp_t[1:]
            new_segt = [[supsamp_x[j], supsamp_x[j+1]]
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
        self.ax.set_ylim((0, 1))
        self.ax.set_xlim((0, 1))
        self.text = self.ax.text(0, 0, 0)
        self.text.set_position((0.5, 1.01))
        self.ax.add_collection(self.lc)
        self.head_ref = []
        self.meas = measure
        self.show = show
        self.block = block
        # For colorbar
        norm = mpl.colors.Normalize(vmin=0, vmax=max(measure.weights))
        cmap = plt.get_cmap('brg', 100)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        self.fig.colorbar(sm,
                          ticks=np.linspace(0, max(measure.weights), 9))


    def animate(self, i):
        if i <= self.frames:
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
                t = min(t, 1)
                for j in range(len(self.meas.curves)):
                    # Here the heads of the curves are drawn
                    marker = mpl.markers.MarkerStyle(marker='x')
                    if len(self.meas.curves) > 1:
                        rotation = marker.get_transform().rotate_deg
                        angle = (j/len(self.meas.curves)-1)*80
                        marker._transform = rotation(angle)
                    curv = self.meas.curves[j]
                    self.head_ref.append(
                        self.ax.scatter(
                            curv.eval(t)[0, 0], curv.eval(t)[0, 1],
                            c=np.array(self.colors[j]).reshape(1, -1), s=60,
                            marker=marker, lw=0.8))
        self.text.set_text('time '+str(np.round(min(i/self.frames, 1), 2)))
        return self.lc

    def start(self):
        self.anim = animation.FuncAnimation(
                        self.fig, self.animate,
                        frames=int(np.ceil(1.1*self.frames)),
                        interval=40, blit=False, repeat=True)

    def draw(self):
        self.start()
        if self.filename is not None and config.use_ffmpeg is True:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
            self.anim.save(self.filename + '.mp4', writer=writer, dpi=200)
        plt.show(block=self.block)
        if not self.show:
            plt.close()

    def subsegment(self, t):
        # To grab the subsegments, with their colors that has times less than t
        new_segments = []
        new_times = []
        for i in range(len(self.segments)):
            # i represented a curve index
            # j represents the respective segment
            new_segments.append(
                [self.segments[i][j] for j in range(len(self.segments[i]))
                 if self.times[i][j] <= t+1e-5])
            new_times.append([time for time in self.times[i]
                              if time <= t + 1e-5])
        alpha_colors = self.alpha_channel(self.colors, new_times, t)
        return new_segments, alpha_colors

    def alpha_channel(self, colors, new_times, t):
        # How the colors fade for the inserted segments
        lowest_alpha = 0.2
        head_length = 0.1
        power = 4

        def alpha_profile(s):
            increase = (1-lowest_alpha)/(1-head_length)**power*s**power
            return increase + lowest_alpha

        alpha_colors = []
        for i in range(len(new_times)):
            if t <= head_length:
                alpha_colors.append([colors[i] for j in
                                     range(len(new_times[i]))])
            else:
                alpha_colors.append(
                    [np.append(colors[i][:-1],
                     np.minimum(1, alpha_profile(1-t+new_t)))
                     for new_t in new_times[i]])
        return alpha_colors


def animate_dual_variable(w_t, measure, **kwargs):
    default_parameters = {
        'resolution': 0.01,
        'filename': None,
        'show': True,
        'block': False,
    }
    # Incorporate the input keyworded values
    for key, val in kwargs.items():
        if key in default_parameters:
            default_parameters[key] = val
        else:
            raise KeyError(
               'The given keyworded argument «{}» is not valid'.format(key))
    # Assign the respective variables
    varnames = ['resolution', 'filename', 'show', 'block']
    resolution, filename, show, block, = [default_parameters[n]
                                          for n in varnames]
    #
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
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(0, 1), ylim=(0, 1))
    line_collection = LineCollection([])
    img = plt.imshow(evals[0], extent=[0, 1, 0, 1], vmin=val_min, vmax=val_max,
                     origin='lower', cmap='RdGy')
    ax.add_collection(line_collection)
    # Create colorbar
    norm = mpl.colors.Normalize(vmin=val_min, vmax=val_max)
    cmap = plt.get_cmap('RdGy', 100)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ticks=np.linspace(val_min, val_max, 9))

    def init():
        # Initialize animation
        img.set_data(evals[0])
        line_collection.set_segments([])
        return img, line_collection

    def animate(i):
        t = i/frames
        img.set_data(evals[i])
        total_segments = []
        total_colors = []
        if measure is not None:
            for curv in measure.curves:
                _, (segments, colors) = curv.draw(tf=t, plot=False)
                total_segments.extend(segments)
                total_colors.extend(colors)
            line_collection.set_segments(total_segments)
            line_collection.set_color(total_colors)
        return img, line_collection

    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=40,
                                  blit=True, init_func=init)
    if filename is not None and config.use_ffmpeg is True:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(filename + '.mp4', writer=writer, dpi=200)
    plt.show(block=block)
    if not show:
        plt.close()
    return ani


def supersample(curve, max_jump=0.01):
    # Given a family of spatial points and their respective times, returns a
    # super sampled version of them, following a linear path between the nodes
    samples = len(curve.time_samples)-1
    supersampl_t = []
    supersampl_x = []
    for i in range(samples):
        ti = curve.time_samples[i]
        tf = curve.time_samples[i+1]
        xi = curve.spatial_points[i]
        xf = curve.spatial_points[i+1]
        diff = np.linalg.norm(xf-xi)
        if diff > max_jump:
            N_samp = math.ceil(diff/max_jump)
            new_t = np.linspace(ti, tf, N_samp)[0:-1]
            new_x = [curve.eval(t)[0] for t in new_t]
            supersampl_t = np.append(supersampl_t, new_t)
            supersampl_x.extend(new_x)
        else:
            supersampl_t = np.append(supersampl_t, ti)
            supersampl_x.extend([xi])
    supersampl_x.extend([curve.spatial_points[-1]])
    supersampl_t = np.append(supersampl_t, 1)
    return supersampl_t, supersampl_x
