import numpy as np
import curves
import operators as op
import checker
import config
import misc
import pickle

# This module implements all the examples considered for data generation.

def generate(case, **new_parameters):
    # to generate predefined examples of ground_truths or data.
    if case == 1:
        # Single straight line example, default parameter values
        params = {
            'x_i'   : 0.2,
            'y_i'   : 0.2,
            'x_f'   : 0.8,
            'y_f'   : 0.8,
            'energy': 1,
            'save_data': True,
        }
        for key, val in new_parameters.items():
            params[key] = val
        # generate the curve
        x_i = np.array([params['x_i'], params['y_i']])
        x_f = np.array([params['x_f'], params['y_f']])
        gt_curve_1 = curves.curve(np.linspace(0,1,2), np.array([x_i, x_f]))
        # producing the measure
        ground_truth = curves.measure()
        ground_truth.add(gt_curve_1, gt_curve_1.energy()*params['energy'])
        # Save animation of ground truth into folder
        if params['save_data']==True:
            save_ground_truth(ground_truth)
        f_t = op.K_t_star_full(ground_truth)
        config.f_t = f_t
        return f_t

    if case == 2:
        # Two crossing lines example
        params = {
            'x1_i'   : 0.2,
            'y1_i'   : 0.2,
            'x1_f'   : 0.8,
            'y1_f'   : 0.8,
            'x2_i'   : 0.8,
            'y2_i'   : 0.2,
            'x2_f'   : 0.2,
            'y2_f'   : 0.8,
            'energy1': 1,
            'energy2': 1,
            'offset' : 0,
            'save_data': True,
        }
        # offset: the curves cross in the middle, offset is a displacement
        #         done so the curves do not touch, by the distance given.
        #         only the second curve is displaced. this is done diagonally.
        for key, val in new_parameters.items():
            params[key] = val
        # generate curve 1
        x1_i = np.array([params['x1_i'], params['y1_i']])
        x1_f = np.array([params['x1_f'], params['y1_f']])
        gt_curve_1 = curves.curve(np.linspace(0,1,2), np.array([x1_i, x1_f]))
        # generate curve 2
        x2_i = np.array([params['x2_i'], params['y2_i']])
        x2_f = np.array([params['x2_f'], params['y2_f']])
        # offsetting
        x2_i = x2_i + params['offset']*np.array([-1, 1])
        x2_f = x2_f + params['offset']*np.array([-1, 1])
        gt_curve_2 = curves.curve(np.linspace(0,1,2), np.array([x2_i, x2_f]))
        # producing the measure
        ground_truth = curves.measure()
        ground_truth.add(gt_curve_1, gt_curve_1.energy()*params['energy1'])
        ground_truth.add(gt_curve_2, gt_curve_2.energy()*params['energy2'])
        # Save animation of ground truth into folder
        if params['save_data']==True:
            save_ground_truth(ground_truth)
        f_t = op.K_t_star_full(ground_truth)
        config.f_t = f_t
        return f_t

    if case == 4:
        # The curved curves, tangential and with a kink.
        # the involved parameters are too precise for tweaking
        params = {
            'energy1':1,
            'energy2':1,
            'energy3':1,
            'save_data':True,
        }
        for key, val in new_parameters.items():
            params[key] = val
        # Curves defining the ground truth
        # # middle curve
        gt_curve_1 = curves.curve(np.linspace(0,1,2), np.array([[0.1,0.1],
                                                                [0.7,0.8]]))
        # # top curve: a circle segment
        times_2 = np.linspace(0,1,101)
        center = np.array([0.1,0.9])
        radius = np.linalg.norm(center-np.array([0.5,0.5]))-0.1
        x_2 = radius*np.cos( 3*np.pi/2 + times_2*np.pi/2) + center[0]
        y_2 = radius*np.sin( 3*np.pi/2 + times_2*np.pi/2) + center[1]
        positions_2 = np.array([x_2,y_2]).T
        gt_curve_2 = curves.curve(times_2, positions_2)
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
        gt_curve_3 = curves.curve(times_3, positions_3)
        # Setting up the ground truth
        ground_truth = curves.measure()
        ground_truth.add(gt_curve_1, gt_curve_1.energy()*params['energy1'])
        ground_truth.add(gt_curve_2, gt_curve_2.energy()*params['energy3'])
        ground_truth.add(gt_curve_3, gt_curve_3.energy()*params['energy2'])
        # saving an returning
        if params['save_data']==True:
            save_ground_truth(ground_truth)
        f_t = op.K_t_star_full(ground_truth)
        config.f_t = f_t
        return f_t

    if case == 5:
        params = {
            'save_data': True,
        }
        # Curves defining the ground truth YOU example
        # # # Global coordinates
        top = 0.7
        bot = 0.3
        # # Y 
        # # # coordinates
        Y_b = [0.2, bot]
        Y_m = [0.2, bot + (top-bot)/3*2]
        Y_l = [0.1, top]
        Y_r = [0.3, top]
        # # O 
        # # # coordinates
        O_tl = [0.4, top]
        O_bl = [0.4, bot]
        O_br = [0.6, bot]
        O_tr = [0.6, top]
        # # U
        # # # coordinates
        U_tl = [0.7, top]
        U_bl = [0.7, bot]
        U_br = [0.9, bot]
        U_tr = [0.9, top]
        # curves
        # # Y curve: two curves coming from bottom to each side
        Y_curve_l = curves.curve(np.linspace(0,1,3), np.array([Y_b, Y_m, Y_l]))
        Y_curve_r = curves.curve(np.linspace(0,1,3), np.array([Y_b, Y_m, Y_r]))
        # # O curve: looping and different speeds
        times = np.array([0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0])
        O_curve = curves.curve(times, np.array([O_tl, O_bl, O_br, O_tr, O_tl, O_bl,
                                                O_br, O_tr]))
        # # U curve: 
        times = np.linspace(0,1,5)
        U_curve = curves.curve(times, np.array([U_tl, U_bl, U_br, U_tr, U_tr]))
        # Measure
        ground_truth = curves.measure()
        ground_truth.add(Y_curve_l,0.5)
        ground_truth.add(Y_curve_r,0.5)
        ground_truth.add(O_curve , 1)
        ground_truth.add(U_curve, 0.7)
        # saving an returning
        if params['save_data']==True:
            save_ground_truth(ground_truth)
        f_t = op.K_t_star_full(ground_truth)
        config.f_t = f_t
        return f_t

def save_ground_truth(ground_truth):
    # Save animation of ground truth into folder
    logger = config.logger
    temp = config.temp_folder
    logger.status([0], ground_truth)
    # Generate data and set it to global in the config module
    f_t = op.K_t_star_full(ground_truth)
    config.f_t = f_t
    # Save visualization of the backprojected data
    new_measure = curves.measure()
    w_t = op.w_t(new_measure)
    w_t.animate(measure = ground_truth,
               filename = '{}/backprojected_data'.format(temp), show = False)

def add_noise(f_t, noise_level, noisevector_file = None):
    # Add some type of noise to the data for reconstruction
    # consider a gaussian model of noise on H = [H_t]_t
    # generate a noise vector in H
    # noise_level ∈ [0,1]
    # noisevector_file ∈ string, if None, a random one is generated.
    if noisevector_file is None:
        noise_vector = []
        for t, freq in enumerate(op.K):
    #        noise_t = np.array([np.exp(1j*2*np.pi*np.random.rand())*np.random.rand()
    #                               for k in range(freq)])
            # Gaussian noise per coordinate and complex coordinate
            noise_t = np.array([np.random.normal(0,1,1)[0] + 1j* np.random.normal(0,1,1)[0]
                                   for k in range(freq)])
            noise_vector.append(noise_t)
        noise_vector = np.array(noise_vector)
        assert checker.is_in_H(noise_vector)
        with open('{}/noise_vector.pickle'.format(config.temp_folder), 'wb') as f:
            pickle.dump(noise_vector, f)
    else:
        noise_vector = pickle.load(open(noisevector_file, "rb"))
        with open('{}/noise_vector.pickle'.format(config.temp_folder), 'wb') as f:
            pickle.dump(noise_vector, f)
    # just multiply it adequeatedly to obtain the desired noise level
    # normalize
    noise_vector = noise_vector/np.sqrt(op.int_time_H_t_product(noise_vector,
                                                                noise_vector))
    noise_vector = noise_vector*np.sqrt(op.int_time_H_t_product(f_t,f_t)*noise_level)
    f_t_noise = f_t+ noise_vector
    config.f_t = f_t_noise
    # save directly the data
    with open('{}/f_t.pickle'.format(config.temp_folder), 'wb') as f:
        pickle.dump(f_t_noise, f)

    if noise_level>0:
        temp = config.temp_folder
        new_measure = curves.measure()
        w_t = op.w_t(new_measure)
        misc.animate_step3(w_t, new_measure, filename =
                           '{}/backprojected_data_noisy'.format(temp),
                           show = False)
        w_t.data = -noise_vector
        misc.animate_step3(w_t, new_measure, filename =
                           '{}/backprojected_data_noisy_part'.format(temp),
                           show = False)

    return f_t_noise





