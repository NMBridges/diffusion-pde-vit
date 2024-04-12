''' Generates a dataset for the 1-D heat equation '''

from fenics import (UnitSquareMesh, UnitIntervalMesh, FunctionSpace,
                    Expression, DirichletBC, interpolate,
                    TrialFunction, TestFunction, Constant,
                    dot, grad, lhs, rhs, dx, Function, solve,
                    plot, div, dof_to_vertex_map)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyvista as pv

T = 3.0            # final time
num_steps = 96     # number of time steps
dt = T / num_steps # time step size
alpha = 0.4          # parameter alpha
beta = 0.9        # parameter beta
gamma = 0.8        # parameter beta

# Create mesh and define function space
nx = 32 - 1
ny = 32 - 1
# mesh = UnitIntervalMesh(nx)
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)

def solve_heat(params, plot_vals = True, verbose = True):
    # Define boundary condition
    u_D = Constant(0)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

    # Define initial value
    init_cond = Expression(f'sin(3.14159 * x[0]) * sin(3.14159 * x[1])', degree=2)
    u_n = interpolate(init_cond, V)
    # u_n = interpolate(u_D, V)
    #u_n = project(u_D, V)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression(f'3 * sin(3.1415 * x[0]) * sin(3.1415 * x[1]) * exp(2-{params[0]}*t) * (exp(-5 * (x[0] - {params[1]}) * (x[0] - {params[1]})) + exp(-5 * (x[1] - {params[2]}) * (x[1] - {params[2]})))', degree=2, t=0) # alpha is parameterization term
    # f = Constant(params[0]) # alpha is parameterization term

    F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
    a, L = lhs(F), rhs(F)

    # Time-stepping
    u = Function(V)
    t = 0

    us = []
    for n in range(num_steps):

        # Update current time
        t += dt
        u_D.t = t

        # TODO: maybe remove
        f.t = t

        # Compute solution
        solve(a == L, u, bc)

        if plot_vals:
            # Plot solution
            plot(u, label=f'{t=}')

        # Compute error at vertices
        u_e = interpolate(u_D, V)
        error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
        if verbose:
            print('t = %.2f: error = %.3g' % (t, error))

        # Update previous solution
        u_n.assign(u)

        xy = np.array(mesh.coordinates())
        nparr = np.array([u(pt[1], pt[0]) for pt in xy]).reshape((ny+1, nx+1))
        
        us.append(nparr)

    if plot_vals:
        plt.xlabel('x[0]')
        plt.ylabel('x[1]')
        plt.legend()
        plt.grid()
        plt.title("u(x) values at x, t=0")
        plt.savefig('test.png')

    return np.array(us)

def solve_navier_stokes(params, plot_vals = True, verbose = True):
    # Define boundary condition
    u_D = Constant(0)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)
    
    # Define initial value
    init_cond = Expression(f'sin(3.14159 * x[0]) * sin(3.14159 * x[1])', degree=2)
    u_n = interpolate(init_cond, V)
    # u_n = interpolate(u_D, V)
    #u_n = project(u_D, V)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression(f'3 * sin(3.1415 * x[0]) * sin(3.1415 * x[1]) * exp(2-{params[0]}*t) * (exp(-5 * (x[0] - {params[1]}) * (x[0] - {params[1]})) + exp(-5 * (x[1] - {params[2]}) * (x[1] - {params[2]})))', degree=2, t=0) # alpha is parameterization term
    # f = Constant(params[0]) # alpha is parameterization term

    F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
    a, L = lhs(F), rhs(F)

    # Time-stepping
    u = Function(V)
    t = 0

    us = []
    for n in range(num_steps):

        # Update current time
        t += dt
        u_D.t = t

        # TODO: maybe remove
        f.t = t

        # Compute solution
        solve(a == L, u, bc)

        if plot_vals:
            # Plot solution
            plot(u, label=f'{t=}')

        # Compute error at vertices
        u_e = interpolate(u_D, V)
        error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
        if verbose:
            print('t = %.2f: error = %.3g' % (t, error))

        # Update previous solution
        u_n.assign(u)

        xy = np.array(mesh.coordinates())
        nparr = np.array([u(pt[1], pt[0]) for pt in xy]).reshape((ny+1, nx+1))
        
        us.append(nparr)

    if plot_vals:
        plt.xlabel('x[0]')
        plt.ylabel('x[1]')
        plt.legend()
        plt.grid()
        plt.title("u(x) values at x, t=0")
        plt.savefig('test.png')

    return np.array(us)

def make_gif(filename, np_arr, fps):
    fig = plt.figure()
    im = plt.imshow(np_arr[0])
    def animate_func(i):
        im.set_array(np_arr[i])
        return [im]
    anim = animation.FuncAnimation(fig, animate_func, frames=num_steps, interval=1000/fps)
    anim.save(filename, fps=fps)

def plot_heat(params, t=None):
    fff = solve_heat(params)
    plt.clf()
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.title("u(x) values at x, t=0")
    if t is None:
        make_gif('2.gif', fff, 16)
    else:
        plt.imshow(fff[t])
    breakpoint()
    plt.savefig('2.png')

dx2 = 1 / (nx - 1)**2

def heat_1d_loss(pred_x_i1, label):
    time_deriv = (pred_x_i1[1:,:] - pred_x_i1[-1,:]) / dt
    second_spat_deriv_x1 = (pred_x_i1[:,2:] - 2 * pred_x_i1[:,1:-1] + pred_x_i1[:,:-2]) / dx2
    f = label
    resid = time_deriv[:,1:-1] - second_spat_deriv_x1[:-1,:] - f # physics loss on terms that have derivatives (loss doesn't apply to some edge values)
    heat_loss = np.linalg.norm(resid) / (32 * 32)
    print(resid)
    return heat_loss

t_tens = (np.arange(num_steps))[None,:,None,None] / (num_steps - 1) * 1
y_tens = (np.arange(ny + 1))[None,None,:,None] / 31 * 1
x_tens = (np.arange(nx + 1))[None,None,None,:] / 31 * 1

def eval_f(params):
    return 3 * np.sin(3.1415 * x_tens) * np.sin(3.1415 * y_tens) * np.exp(2-params[0] * t_tens) * (np.exp(-5 * (x_tens - params[1])**2) + np.exp(-5 * (y_tens - params[2])**2))
    # return params[0] * np.ones((1,32,32,32))
    # return np.zeros((1,32,32,32))

def heat_2d_loss(pred_x_i1, label):
    time_deriv = (pred_x_i1[:,1:,:,:] - pred_x_i1[:,:-1,:,:]) / dt
    second_spat_deriv_x = (pred_x_i1[:,:,:,2:] - 2 * pred_x_i1[:,:,:,1:-1] + pred_x_i1[:,:,:,:-2]) / dx2
    second_spat_deriv_y = (pred_x_i1[:,:,2:,:] - 2 * pred_x_i1[:,:,1:-1,:] + pred_x_i1[:,:,:-2,:]) / dx2
    f = eval_f(label)[:,:-1,1:-1,1:-1]
    resid = time_deriv[:,:,1:-1,1:-1] - second_spat_deriv_x[:,:-1,1:-1,:] - second_spat_deriv_y[:,:-1,:,1:-1] - f # physics loss on terms that have derivatives (loss doesn't apply to some edge values)
    heat_loss = np.linalg.norm(resid) / (num_steps * (nx+1) * (ny+1))**3
    print(resid[0,0])
    print(f[0,0,-1,-1])
    print((time_deriv[:,:,1:-1,1:-1] - second_spat_deriv_x[:,:-1,1:-1,:] - second_spat_deriv_y[:,:-1,:,1:-1])[0,0,-1,-1])
    print(resid.mean())
    print(resid.std())
    return heat_loss, resid

fff = solve_heat([alpha,beta,gamma])
_, k = heat_2d_loss(fff[None,:,:,:], [alpha,beta,gamma])
plot_heat([alpha, beta, gamma])

breakpoint()

dataset_size = 1500

X = np.zeros((dataset_size, num_steps, ny + 1, nx + 1))
Y = np.zeros((dataset_size, 3)) # params for each process

for i in range(dataset_size):
    params = np.random.random(size=3)
    fn = solve_heat(params, plot_vals=False, verbose=False)[:,:,:]
    X[i,:,:,:] = fn
    Y[i,:] = params
    print(f"Iteration {i+1}/{dataset_size} complete...")

np.save(f'datasets/heat/{"x".join([str(dim) for dim in X.shape])}_data.npy', X, allow_pickle=True)
np.save(f'datasets/heat/{"x".join([str(dim) for dim in X.shape])}_label.npy', Y, allow_pickle=True)