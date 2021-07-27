""" Kernel that implements the forward operators and its derivative
using OpenGl for acceleration"""

import pyopencl as cl
# Import  PyOpenCL Array (a Numpy array plus an OpenCl buffer object)
import numpy as np
import scipy.io
import config
import os 
# import top level packages

# Set environment variables
# This one to automatically use the first available platform (I only have one)
# os.environ['PYOPENCL_CTX'] = '0'  # Personal computer setting
#os.environ['PYOPENCL_CTX'] = '0:1'
# This one to see the compiler warnings
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

# 
platform = cl.get_platforms()[0]  # Select the first platform [0]
device = platform.get_devices()[0]   # Select the first device on this platform
context = cl.create_some_context([device])  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue

with open("kernel_code.c", 'r', encoding='utf-8') as f:
    kernel_code = ''.join(f.readlines())

# Compile the kernel code into an executable OpenCl program
program = cl.Program(context, kernel_code).build()


# Problem specific variables
DOMAIN_TIME = 50
folder = '../data/'
deps = scipy.io.loadmat(folder+'deps.mat')['deps'].astype(np.float64)  # nu
deps = deps.reshape(-1).astype(np.float64)  # nu
lats = scipy.io.loadmat(folder+'lats.mat')['lats']  # xi
lats = lats.reshape(-1).astype(np.float64)
DOMAIN_WIDTH = len(lats.reshape(-1))
DOMAIN_DEPTH = len(deps.reshape(-1))
#
X_VAR = np.linspace(-0.0049, 0.00505, DOMAIN_TIME).astype(np.float64)
Y_VAR = (np.ones(DOMAIN_TIME)*0.00625).astype(np.float64)
#
K = DOMAIN_TIME*DOMAIN_DEPTH*DOMAIN_WIDTH

def kernel(params):
    """ The continuous kernel, but evaluated in a given grid.
    Parameters
    ----------
    params : numpy.ndarray of shape (M, 12)
        representing M different sets of parameters
        a4, a2, e3, e1, f3, f1, om2, om0, phi2, phi0, th2, th0

    Returns
    -------
    (M, K) numpy.ndarray with K = DOMAIN_TIME*DOMAIN_DEPTH*DOMAIN_WIDTH
    """
    M = params.shape[0]
    full_output = np.zeros((M, K)).astype(np.float64)
    for m in range(M):
        param = params[m].astype(np.float64)
        cl_param = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=param)
        cl_deps = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=deps)
        cl_lats = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=lats)
        cl_X_VAR = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=X_VAR)
        cl_Y_VAR = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=Y_VAR)
        cl_output = cl.Buffer(context, cl.mem_flags.WRITE_ONLY,
                              full_output[0].nbytes)
        program.kern(queue, (DOMAIN_TIME, DOMAIN_DEPTH, DOMAIN_WIDTH), None,
                     cl_param, cl_deps, cl_lats, cl_X_VAR, cl_Y_VAR,
                     cl_output)
        #
        cl.enqueue_copy(queue, full_output[m], cl_output)
        queue.finish()
    return full_output

def gradient_kernel(params):
    """ The continuous kernel, but evaluated in a given grid.
    Parameters
    ----------
    params : numpy.ndarray of shape (M, 12)
        representing M different sets of parameters
        a4, a2, e3, e1, f3, f1, om2, om0, phi2, phi0, th2, th0

    Returns
    -------
    (M, K) numpy.ndarray with K = DOMAIN_TIME*DOMAIN_DEPTH*DOMAIN_WIDTH
    """
    M = params.shape[0]
    bytesample = np.zeros(K).astype(np.float64)
    full_output = np.zeros((M, 12*K)).astype(np.float64)
    for m in range(M):
        param = params[m].astype(np.float64)
        cl_param = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=param)
        cl_deps = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=deps)
        cl_lats = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=lats)
        cl_X_VAR = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=X_VAR)
        cl_Y_VAR = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=Y_VAR)
        # derivative buffers
        cl_a4 = cl.Buffer(context, cl.mem_flags.WRITE_ONLY,
                              bytesample.nbytes)
        program.grada4_kern(queue, (DOMAIN_TIME, DOMAIN_DEPTH, DOMAIN_WIDTH),
                          None, cl_param, cl_deps, cl_lats, cl_X_VAR, cl_Y_VAR,
                          cl_a4)
        cl_a2 = cl.Buffer(context, cl.mem_flags.WRITE_ONLY,
                              bytesample.nbytes)
        program.grada2_kern(queue, (DOMAIN_TIME, DOMAIN_DEPTH, DOMAIN_WIDTH),
                          None, cl_param, cl_deps, cl_lats, cl_X_VAR, cl_Y_VAR,
                          cl_a2)
        cl_e3 = cl.Buffer(context, cl.mem_flags.WRITE_ONLY,
                              bytesample.nbytes)
        program.grade3_kern(queue, (DOMAIN_TIME, DOMAIN_DEPTH, DOMAIN_WIDTH),
                          None, cl_param, cl_deps, cl_lats, cl_X_VAR, cl_Y_VAR,
                          cl_e3)
        cl_e1 = cl.Buffer(context, cl.mem_flags.WRITE_ONLY,
                              bytesample.nbytes)
        program.grade1_kern(queue, (DOMAIN_TIME, DOMAIN_DEPTH, DOMAIN_WIDTH),
                          None, cl_param, cl_deps, cl_lats, cl_X_VAR, cl_Y_VAR,
                          cl_e1)
        cl_f3 = cl.Buffer(context, cl.mem_flags.WRITE_ONLY,
                              bytesample.nbytes)
        program.gradf3_kern(queue, (DOMAIN_TIME, DOMAIN_DEPTH, DOMAIN_WIDTH),
                          None, cl_param, cl_deps, cl_lats, cl_X_VAR, cl_Y_VAR,
                          cl_f3)
        cl_f1 = cl.Buffer(context, cl.mem_flags.WRITE_ONLY,
                              bytesample.nbytes)
        program.gradf1_kern(queue, (DOMAIN_TIME, DOMAIN_DEPTH, DOMAIN_WIDTH),
                          None, cl_param, cl_deps, cl_lats, cl_X_VAR, cl_Y_VAR,
                          cl_f1)
        cl_om2 = cl.Buffer(context, cl.mem_flags.WRITE_ONLY,
                              bytesample.nbytes)
        program.gradom2_kern(queue, (DOMAIN_TIME, DOMAIN_DEPTH, DOMAIN_WIDTH),
                          None, cl_param, cl_deps, cl_lats, cl_X_VAR, cl_Y_VAR,
                          cl_om2)
        cl_om0 = cl.Buffer(context, cl.mem_flags.WRITE_ONLY,
                              bytesample.nbytes)
        program.gradom0_kern(queue, (DOMAIN_TIME, DOMAIN_DEPTH, DOMAIN_WIDTH),
                          None, cl_param, cl_deps, cl_lats, cl_X_VAR, cl_Y_VAR,
                          cl_om0)
        cl_phi2 = cl.Buffer(context, cl.mem_flags.WRITE_ONLY,
                              bytesample.nbytes)
        program.gradphi2_kern(queue, (DOMAIN_TIME, DOMAIN_DEPTH, DOMAIN_WIDTH),
                          None, cl_param, cl_deps, cl_lats, cl_X_VAR, cl_Y_VAR,
                          cl_phi2)
        cl_phi0 = cl.Buffer(context, cl.mem_flags.WRITE_ONLY,
                              bytesample.nbytes)
        program.gradphi0_kern(queue, (DOMAIN_TIME, DOMAIN_DEPTH, DOMAIN_WIDTH),
                          None, cl_param, cl_deps, cl_lats, cl_X_VAR, cl_Y_VAR,
                          cl_phi0)
        cl_th2 = cl.Buffer(context, cl.mem_flags.WRITE_ONLY,
                              bytesample.nbytes)
        program.gradth2_kern(queue, (DOMAIN_TIME, DOMAIN_DEPTH, DOMAIN_WIDTH),
                          None, cl_param, cl_deps, cl_lats, cl_X_VAR, cl_Y_VAR,
                          cl_th2)
        cl_th0 = cl.Buffer(context, cl.mem_flags.WRITE_ONLY,
                              bytesample.nbytes)
        program.gradth0_kern(queue, (DOMAIN_TIME, DOMAIN_DEPTH, DOMAIN_WIDTH),
                          None, cl_param, cl_deps, cl_lats, cl_X_VAR, cl_Y_VAR,
                          cl_th0)
        #
        queue.finish()
        cl.enqueue_copy(queue, full_output[m][0:K], cl_a4)
        cl.enqueue_copy(queue, full_output[m][1*K:2*K], cl_a2)
        cl.enqueue_copy(queue, full_output[m][2*K:3*K], cl_e3)
        cl.enqueue_copy(queue, full_output[m][3*K:4*K], cl_e1)
        cl.enqueue_copy(queue, full_output[m][4*K:5*K], cl_f3)
        cl.enqueue_copy(queue, full_output[m][5*K:6*K], cl_f1)
        cl.enqueue_copy(queue, full_output[m][6*K:7*K], cl_om2)
        cl.enqueue_copy(queue, full_output[m][7*K:8*K], cl_om0)
        cl.enqueue_copy(queue, full_output[m][8*K:9*K], cl_phi2)
        cl.enqueue_copy(queue, full_output[m][9*K:10*K], cl_phi0)
        cl.enqueue_copy(queue, full_output[m][10*K:11*K], cl_th2)
        cl.enqueue_copy(queue, full_output[m][11*K:12*K], cl_th0)
        
    return full_output.reshape(M,K,12)

def random_propose(dual_var):
    """Proposes points that are ideal to descend.

    This function is adhoc to each particular problem.

    Parameters
    ----------
    dual_var : classes.dual_variable

    Returns :
    y_val : numpy.ndarray
        vector of size 1xd, with d the dimension of the lifted space.
    """
    # Get the range of values
    bounds = config.DOMAIN_BOUNDS
    num_tries = config.MULTISTART_RANDOM_PROPOSE_TOTAL_NUMBER_OF_TRIES
    best_y = None
    best_val = -np.inf
    for _ in range(num_tries):
        a4 = bounds[0][0] + np.random.rand()*bounds[0][1]
        a2 = bounds[1][0] + np.random.rand()*bounds[1][1]
        e3 = bounds[2][0] + np.random.rand()*bounds[2][1]
        e1 = bounds[3][0] + np.random.rand()*bounds[3][1]
        f3 = bounds[4][0] + np.random.rand()*bounds[4][1]
        f1 = bounds[5][0] + np.random.rand()*bounds[5][1]
        om2 = bounds[6][0] + np.random.rand()*bounds[6][1]
        om0 = bounds[7][0] + np.random.rand()*bounds[7][1]
        phi2 = bounds[8][0] + np.random.rand()*bounds[8][1]
        phi0 = bounds[9][0] + np.random.rand()*bounds[9][1]
        th2 = bounds[10][0] + np.random.rand()*bounds[10][1]
        th0 = bounds[11][0] + np.random.rand()*bounds[11][1]
        y_val = np.array([[a4, a2, e3, e1, f3, f1, om2, om0, 
                           phi2, phi0, th2, th0]])
        dual_var_eval = dual_var(y_val)
        if dual_var_eval > best_val:
            best_y = y_val
            best_val = dual_var_eval
    return best_y


if __name__ == '__main__':
    pass
#    params = np.array([[0,0,0,1500,0,500, 0,30000, 0, -np.pi/3, 0, np.pi/3]])
#    output = kernel(params).reshape(DOMAIN_TIME, DOMAIN_DEPTH, DOMAIN_WIDTH)
#    print(output)
#    # misc.draw_data(output[0], 'results')
#    grad_output = grad_kernel(params)
#    print(grad_output)
#    os.system('rm -r results')
#    misc.draw_data(grad_output.reshape(DOMAIN_TIME, DOMAIN_DEPTH, DOMAIN_WIDTH,
#                                       12)[:,:,:,1], 'results')
