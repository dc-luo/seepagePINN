import dolfin as dl 
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.animation import FuncAnimation
dl.set_log_active(False)

def compute_first_derivative(func, x, delta):
    f1 = func(x)
    f2 = func(x+delta)
    return (f2-f1)/delta

def compute_second_derivative(func, x, delta):
    f0 = func(x-delta)
    f1 = func(x)
    f2 = func(x+delta)
    return (f2 - 2*f1 + f0)/delta**2

class InitialConditions(dl.UserExpression):
    def __init__(self, h1, **kwargs):
        self.h1 = h1 
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = self.h1
        values[1] = 0.0

    def value_shape(self):
        return (2,) 

# Define constants
test_name = "test1"
h1 = 1.0
H2 = 0.167
L = 0.667
K = 1.0
eta = 0.4 
bounds = [0.5, 1.0]

test_name = "test2"
h1 = 3.22
H2 = 0.84
L = 1.62
K = 0.2
eta = 0.2 
bounds = [2.0, 3.2]

test_name = "test3"
h1 = 4
H2 = 0.84
L = 1.62
K = 0.2
eta = 0.2 
bounds = [3.0, 4.2]

# Compute flow boundary condition 
q_star = (h1**2 - H2**2)/(2*L) 
print(q_star)

# Define the constants
q_star = dl.Constant(q_star)
K = dl.Constant(K) 
eta = dl.Constant(eta)

# Define discretization
T_max = 2.0
N_steps = 4000
dt = T_max/N_steps
N = 64

# Define function space and boundaries
mesh = dl.IntervalMesh(N, 0, L) 
P1 = dl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
P2 = dl.FiniteElement("Lagrange", mesh.ufl_cell(), 2)
ME = dl.FunctionSpace(mesh, P1*P2) 

def right_boundary(x, on_boundary):
    return x[0] > L - dl.DOLFIN_EPS

def left_boundary(x, on_boundary):
    return x[0] < dl.DOLFIN_EPS

# Define functions
u = dl.Function(ME)
u0 = dl.Function(ME) 
u_test = dl.TestFunction(ME) 

h, q = dl.split(u) 
h0, q0 = dl.split(u0)
v, p = dl.split(u_test)

# Initial conditions
u_init = InitialConditions(h1) 
u.interpolate(u_init)
u0.interpolate(u_init)

n = dl.FacetNormal(mesh) 

f0 = h*h/3 * dl.inner(dl.grad(q), dl.grad(p)) * dl.dx \
        - h/3 * q * dl.inner(dl.grad(h), dl.grad(p)) * dl.dx \
        + q * p * dl.dx \
        + h * dl.div(h * dl.Constant((1.0,))) * p * dl.dx \
        + h/3 * q * dl.inner(dl.grad(h), n) * p * dl.ds \
        - h*h/3 * dl.inner(dl.grad(q), n) * p *dl.ds

# f0 = h*h/(3*K) * dl.inner(dl.grad(q), dl.grad(p)) * dl.dx \
#         - h/(3*K) * q * dl.inner(dl.grad(h), dl.grad(p)) * dl.dx \
#         + (1/K) * q * p * dl.dx \
#         + h * dl.div(h * dl.Constant((1.0,))) * v * dl.dx \
#         + h/(3*K) * q * dl.inner(dl.grad(h), n) * p * dl.ds

f1 = (h - h0) * dl.Constant(eta) / dl.Constant(dt) * v * dl.dx \
          + dl.div(q * dl.Constant((1.0,))) * v * dl.dx 

f = f0 + f1

time = 0.0
count = 0

plot_anim = False
plot_final = True
plot_terms = False
lines = [] 

timestamps = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 2.0]
i_plot = 0

FIGDPI = 100
plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.size"] = 12

grad_color = plt.cm.get_cmap('Blues', 12)
base_color = 0.2
top_color = 1.0
color_incr = (top_color - base_color)/len(timestamps)


plt.figure()

while time < T_max:
    # Define the boundary conditions 
    if time > 0:
        bcq = dl.DirichletBC(ME.sub(1), q_star, right_boundary)
    else:
        bcq = dl.DirichletBC(ME.sub(1), dl.Constant(0.0), right_boundary)

    bch = dl.DirichletBC(ME.sub(0), dl.Constant(h1), left_boundary)

    # Solve the nonlinear problem 
    dl.solve(f == 0, u, bcs=[bch, bcq])

    if plot_anim:
        lines.append(dl.plot(h, label="t = %g" %(time)))
    elif time <= timestamps[i_plot] and timestamps[i_plot] < time + dt:
        dl.plot(h, color=grad_color(base_color+i_plot*(color_incr)), label="t = %g" %(timestamps[i_plot]))
        i_plot += 1

    if count % 10 == 0:
        print("Iteration: ", count, "time = ", time)
    
    time += dt 
    count += 1
    u0.vector().set_local(u.vector().get_local())

if plot_anim:
    plt.close()
else:
    plt.xlabel("x")
    plt.ylabel("h")
    plt.ylim(bounds)
    plt.grid(True)
    plt.legend()
    #plt.savefig(test_name+".pdf")
    plt.show()
    
# plt.close()

if plot_anim:
    fig, ax = plt.subplots()
    ax.set_xlim([0, L])
    ax.set_ylim([0, h1])
    ln, = ax.plot([], [], '-b')
    def update(i):
        line = lines[i][0]
        ln.set_data(line.get_xdata(), line.get_ydata())
        return ln,
    
    ani = FuncAnimation(fig, update, frames=len(lines), interval=1, blit=True, repeat=True)
    ani.save(test_name + "_time_series.mp4")

if plot_final:
    plt.figure()
    plt.subplot(211)
    dl.plot(h)
    plt.xlabel("x")
    plt.ylabel("h")
    plt.ylim([0, None])
    plt.title("t = %g" %(time))

    plt.subplot(212)
    dl.plot(q)
    plt.xlabel("x")
    plt.ylabel("q")
    plt.ylim([0, 1.2*(h1**2 - H2**2)/(2*L)])
    plt.savefig(test_name + "_equilibrium.pdf")

if plot_terms:
    delta = 1/N * 3
    xx = np.linspace(1.1*delta, L-1.1*delta, 200)
    hh = np.zeros(xx.shape)
    dhdx = np.zeros(xx.shape)
    d2hdx2 = np.zeros(xx.shape)
    
    h_func = lambda x : h((x,))
    
    for i,x in enumerate(xx):
        hh[i] = h_func(x) 
        dhdx[i] = compute_first_derivative(h_func, x, delta)
        d2hdx2[i] = compute_second_derivative(h_func, x, delta)
    
    qs = q((0.5,))
    plt.figure()
    plt.plot(xx, -hh*dhdx, label="$-hh_x$") 
    plt.plot(xx, -hh*d2hdx2*qs/3, label="Term 2") 
    plt.plot(xx, -qs/3 * dhdx**2, label="Term 3") 
    plt.plot(xx, -hh*dhdx - hh*d2hdx2*qs/3 - qs/3 * dhdx**2, label="Sum") 
    plt.plot(xx, qs*np.ones(xx.shape), '--k', 'q_star')
    plt.legend()
    plt.show()
