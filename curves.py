#
import numpy as np
import copy

# Plotting imports
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Local imports
import misc
import optimization as opt
import operators as op
import config
import checker

# Global shared parameters
alpha = config.alpha
beta = config.beta

# Module methods

class curve:
    ' Class defining curves, their elements and methods'
    def __init__(self, *args):
        assert len(args)<=2
        if len(args)==1:
            # case in which just positions were used
            space = args[0]
            assert checker.is_in_space_domain(space)
            time = np.linspace(0,1,len(space))
            self.t = time
            self.x = space
            if len(space) != len(config.time):
                self.set_times(config.time)
        else:
            # case in which time and positions were used
            space = args[1]
            time = args[0]
            assert checker.is_in_space_domain(space) \
                   and all(time>=0) and all(time<=1)
            self.t = time
            self.x = space
            self.set_times(config.time)

    def __add__(self,curve2):
        # Warning: implemented just for curves with same time samples
        new_curve = curve(self.t, self.x+curve2.x)
        return new_curve
    def __sub__(self,curve2):
        new_curve = curve(self.t, self.x-curve2.x)
        return new_curve
    def __mul__(self,factor):
        new_curve = curve(self.t, self.x*factor)
        return new_curve
    def __rmul__(self, factor):
        new_curve = curve(self.t, self.x*factor)
        return new_curve

    def draw(self, tf=1, ax = None, color=[0.0, 0.5, 1.0], plot=True):
        #First we supersample the whole curve such that there are not jumps higher
        #than a particular value
        supersampl_t, supersampl_x = misc.supersample(self, max_jump = 0.01)
        # Drop all time samples after tf
        value_at_tf = self.eval(tf)
        index_tf = np.argmax(supersampl_t>=tf)
        supersampl_x = supersampl_x[:index_tf]
        supersampl_t = supersampl_t[:index_tf]
        supersampl_t = np.append(supersampl_t,tf)
        supersampl_x.append(value_at_tf.reshape(-1))
        #Reduce the set of points and times to segments and times, restricted 
        #to the periodic domain.
        _, segments = misc.get_periodic_segments(supersampl_t, supersampl_x)
        #Use the LineCollection class to print using segments and to assign 
        #transparency or colors to each segment
        line_segments = LineCollection(segments)
        # set color 
        lowest_alpha = 0.2
        color = np.array(color)
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
        if plot == True:
            ax = ax or plt.gca()
            ax.add_collection(line_segments)
            ax.scatter(self.x[0,0], self.x[0,1], c=start_color, marker='x', s=0.4)
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            return ax, (segments, rgb_color)
        else:
            return None, (segments, rgb_color)

    def eval(self, t):
        assert t<=1 and t>=0
        # Evaluate the curve at a certain time
       # if t<0 or t >1:
       #     raise Exception('Attempted to evaluate a curve outside the interval'
       #                     + '[0,1]')
        if isinstance(t, (float,int)):
            t = np.array(t).reshape(1)
        N = len(t)
        evals = np.zeros((N,2))
        for i, tt in enumerate(t):
            if tt == 0:
                evals[i,:] = self.x[0,:]
            else:
                index = np.argmax(np.array(self.t)>=t[i])
                ti = self.t[index-1]
                tf = self.t[index]
                xi = self.x[index-1,:]
                xf = self.x[index,:]
                evals[i,:] = (tt-ti)/(tf-ti)*xf + (tf-tt)/(tf-ti)*xi
        return evals

    def eval_discrete(self,t):
        assert checker.is_valid_time(t)
        # The evaluation function, defined on the discrete setting, in which 
        # evaluation happens exclusively at the definition points.
        # The argument value chantes, t becomes an index in 0,1,...,T-1. 
        return self.x[t:t+1]

    def integrate_against(self, w_t):
        assert isinstance(w_t, op.w_t)
        # Method integrate against this curve as a measure δ_(t_i) × ð_\gamma(t)
        # against a dual variable type function
        # Input: target is a w_t type function
        # Output: a real value.
        output = 0
        weights = config.time_weights
        for t in range(config.T):
            output += weights[t]*w_t.eval(t,self.eval_discrete(t))
        return output.reshape(1)[0]

    def H1_seminorm(self):
        diff_t = np.diff(config.time)
        diff_points = np.diff(self.x, axis=0)
        squares_divided_times = (1/diff_t)@diff_points**2
        return np.sqrt(np.sum(squares_divided_times))

    def H1_norm(self):
        diff_times = self.t[1:]-self.t[:-1]
        L2_norm_x = np.sum( diff_times/3*(self.x[:-1,0]**2 +
                        self.x[:-1,0]*self.x[1:,0] + self.x[1:,0]**2))
        L2_norm_y = np.sum( diff_times/3*(self.x[:-1,1]**2 +
                        self.x[:-1,1]*self.x[1:,1] + self.x[1:,1]**2))
        return np.sqrt(L2_norm_x + L2_norm_y) +self.H1_seminorm()

    def energy(self):
            # To compute the Benamou-Brenier energy + total variation = Energy
            return beta/2*self.H1_seminorm()**2 + alpha

    def set_times(self, new_times):
        assert isinstance(new_times, np.ndarray) and \
                np.min(new_times)>=0 and np.max(new_times)<=1
        # Changes the time vector to a new one, just evaluates and set the new 
        # nodes, nothing too fancy.
        new_locations = np.array([self.eval(t).reshape(-1) for t in new_times])
        self.t = new_times
        self.x = new_locations
        self.quads = None

class curve_product:
    # Object describing an element of a weighted product space of curve type 
    # objects.
    def __init__(self, curve_list=None, weights=None):
        if curve_list is None or weights is None:
            self.weights = []
            self.curve_list = []
        else:
            if len(curve_list) == len(weights) and all( w>0 for w in weights):
                self.weights = weights
                self.curve_list = curve_list
            else:
                raise Exception("Wrong number of weights and curves in the curve"+
                                "list, or weights non-positive")
    def __add__(self,curve_list2):
        # Warning, implemented for curve_lists with the same weights
        new_curve_list = [curve1+curve2 for curve1,curve2 in
                          zip(self.curve_list, curve_list2.curve_list)]
        return curve_product(new_curve_list, self.weights)
    def __sub__(self,curve_list2):
        new_curve_list = [curve1-curve2 for curve1,curve2 in
                          zip(self.curve_list, curve_list2.curve_list)]
        return curve_product(new_curve_list, self.weights)
    def __mul__(self,factor):
        new_curve_list = [curve1*factor for curve1 in self.curve_list]
        return curve_product(new_curve_list, self.weights)
    def __rmul__(self,factor):
        new_curve_list = [curve1*factor for curve1 in self.curve_list]
        return curve_product(new_curve_list, self.weights)

    def H1_norm(self):
        output = 0
        for weight, curve in zip(self.weights, self.curve_list):
            output += weight*curve.H1_norm()/len(self.curve_list)
        return output

    def to_measure(self):
        # returns the measure equivalent of this object
        new_measure = measure()
        for weight, curve in zip(self.weights, self.curve_list):
            new_measure.add(curve, weight)
        return new_measure

class measure:
    # object describing a measure, composed of atoms and weights
    # We further include the intensity = weight*atom_normalization_coefficient
    def __init__(self):
        self.curves =  []
        self.energies = np.array([])
        self.intensities = np.array([])
        self.main_energy = None

    def add(self, new_curve, new_intensity):
        # Input: new_curve is a curve class object. new_intensity > 0 real.
        if new_intensity > config.measure_coefficient_too_low:
            self.curves.extend([new_curve])
            self.energies = np.append(self.energies,
                                      new_curve.energy())
            self.intensities = np.append(self.intensities, new_intensity)
            self.main_energy = None

    def __add__(self,measure2):
        new_measure = copy.deepcopy(self)
        new_measure.curves.extend(copy.deepcopy(measure2.curves))
        new_measure.energies = np.append(new_measure.energies,measure2.energies)
        new_measure.intensities = np.append(new_measure.intensities,
                                            measure2.intensities)
        new_measure.main_energy = None
        return new_measure

    def __mul__(self,factor):
        if factor <= 0:
            raise Exception('Cannot use a negative factor for a measure')
        new_measure = copy.deepcopy(self)
        new_measure.main_energy = None
        for i in range(len(self.intensities)):
            new_measure.intensities[i] = new_measure.intensities[i]*factor
        return new_measure

    def __rmul__(self, factor):
        return self*factor

    def modify_intensity(self,curve_index,new_intensity):
        self.main_energy = None
        if curve_index >= len(self.curves):
            raise Exception('Trying to modify an unexistant curve! The given'
                            + 'curve index is too high for the current array')
        if new_intensity < config.measure_coefficient_too_low:
            del self.curves[curve_index]
            self.intensities = np.delete(self.intensities,curve_index)
            self.energies    = np.delete(self.energies   ,curve_index)
        else:
            self.intensities[curve_index] = new_intensity

    def integrate_against(self, w_t):
        assert isinstance(w_t, op.w_t)
        # Method to integrate against this measure. 
        integral = 0
        for i,curve in enumerate(self.curves):
            integral+= self.intensities[i]/self.energies[i]* \
                    curve.integrate_against(w_t)
        return integral

    def spatial_integrate(self, t, target):
        # Method to integrate against this measure for a fixed time, target is
        # a function handle
        val = 0
        for i in range(len(self.intensities)):
            val = val + self.intensities[i]/self.energies[i]* \
                    target(self.curves[i].eval_discrete(t))
        return val

    def to_curve_product(self):
        # transforms the measure to a curve product object
        return curve_product(self.curves, self.intensities)

    def get_main_energy(self):
        if self.main_energy is None:
            self.main_energy = op.main_energy(self, config.f_t)
            return self.main_energy
        else:
            return self.main_energy


    def draw(self,ax = None):
        num_plots = len(self.intensities)
        total_intensities = self.intensities/self.energies
        'get the brg colormap for the intensities of curves'
        colors = plt.cm.brg(np.array(total_intensities)/max(total_intensities))
        ax = ax or plt.gca()
        for i in range(num_plots):
            self.curves[i].draw(ax=ax, color=colors[i,:3])
        plt.gca().set_aspect('equal', adjustable='box')
        'setting colorbar'
        norm = mpl.colors.Normalize(vmin=0,vmax=max(total_intensities))
        cmap = plt.get_cmap('brg',100)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ticks=np.linspace(0, max(total_intensities),10))
        return ax

    def animate(self, filename=None, show=True):
        self.reorder()
        'Produce an animation describing the measures.'
        'First, define the colors for each curve; they depend on the intensities'
        total_intensities = self.intensities/self.energies
        colors = plt.cm.brg(np.array(total_intensities)/max(total_intensities))
        'get the family of segments, times and colors'
        segments = []
        times = []
        for i in range(len(self.intensities)):
            supsamp_t, supsamp_x = misc.supersample(self.curves[i],
                                                    max_jump = 0.01)
            new_times, new_segt = misc.get_periodic_segments(supsamp_t, supsamp_x)
            segments.append(new_segt)
            times.append(new_times)
        animation = misc.Animate(segments, times, colors, total_intensities, self,
                                 frames = 51, filename = filename, show=show)
        animation.draw()

    def reorder(self):
        # Script to reorder the curves inside the measure with an increasing
        # total energy.
        total_intensities = self.intensities/self.energies
        new_order = np.argsort(total_intensities)
        new_measure = measure()
        for idx in new_order:
            new_measure.add(self.curves[idx], self.intensities[idx])
        self.curves = new_measure.curves
        self.intensities = new_measure.intensities
        self.energies = new_measure.energies

if __name__=='__main__':
    'Test for curve periodicity and plotting'
    times = np.linspace(0,1,1000)
    'space is a list of 2dimensional locations'
    fx = lambda t: np.sin(4*np.pi*t)+0.5
    fy = lambda t: np.sin(7*np.pi*t)/3 + 1/2
    space = [np.array([fx(t),fy(t)]) for t in times ]
    'bulding a curve'
    c = curve(times,space)
    fig, ax = plt.subplots()
    c.draw(ax=ax, color=[1.0,0,0])
    plt.show()
    'Test for the integration method'
    times2 = np.linspace(0,1,100)
    fx2 = lambda t: np.cos(t*2*np.pi)
    fy2 = lambda t: np.sin(t*2*np.pi)
    space2 = [np.array([fx2(t),fy2(t)]) for t in times2 ]
    c2 = curve(times2,space2)
    meas = measure()
    meas.add(c2,2)
    meas.add(c,1)
    target = lambda x: 1
    print(meas.integrate_against(target))
    meas.draw()
    'Last multiple plots test'
    nums = 10
    meas2 = measure()
    for j in range(nums):
        times = np.random.rand(5)
        times[0]=0
        times[-1]=1
        space = [np.random.rand(2) for i in range(5)]
        c = curve(times,space)
        meas2.add(c,np.random.rand())
    meas2.draw()
    'Animation test'
    meas.animate(filename='wololo')
