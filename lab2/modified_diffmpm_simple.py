from tracemalloc import start
import taichi as ti
import matplotlib.pyplot as plt
import random

real = ti.f32
ti.init(arch=ti.cuda, default_fp=real, device_memory_GB=1)

# material parameters
N = 60  # reduce to 30 if run out of GPU memory
n_particles = 1000
n_grid = 120
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 3e-4
p_mass = 1
p_vol = 1
E = 50
mu = 1/5*E
la = 1/5*E
#

# creature parameters
max_segs = 8
head_width = 10
head_height = 10
seg_width = 12
origin_x = 40
origin_y = 5
#

# sim parameters
max_steps = 1024
steps = 1024
gravity = 9.8
target = [0.7, 0.2]
dim = 2
num_iterations = 25
#

# taichi stuff
material = ti.field(dtype=ti.i32, shape=n_particles)        # material -- I ADDED
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

x = ti.Vector.field(dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
x_avg = ti.Vector.field(dim, dtype=real, shape=(), needs_grad=True)
v = ti.Vector.field(dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
grid_v_in = ti.Vector.field(dim,
                            dtype=real,
                            shape=(max_steps, n_grid, n_grid),
                            needs_grad=True)
grid_v_out = ti.Vector.field(dim,
                             dtype=real,
                             shape=(max_steps, n_grid, n_grid),
                             needs_grad=True)
grid_m_in = ti.field(dtype=real,
                     shape=(max_steps, n_grid, n_grid),
                     needs_grad=True)
C = ti.Matrix.field(dim,
                    dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
F = ti.Matrix.field(dim,
                    dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
init_v = ti.Vector.field(dim, dtype=real, shape=(), needs_grad=True)
loss = ti.field(dtype=real, shape=(), needs_grad=True)


@ti.kernel
def set_v():
    for i in range(n_particles):
        v[0, i] = init_v[None]


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        F[f + 1, p] = new_F
        J = (new_F).determinant()
        r, s = ti.polar_decompose(new_F)
        cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                 ti.Matrix.diag(2, la * (J - 1) * J)
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + p_mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[f, base + offset] += weight * (p_mass * v[f, p] +
                                                         affine @ dpos)
                grid_m_in[f, base + offset] += weight * p_mass


bound = 3

# material options -- I ADDED
# materials = [0, 1, 2]   # soft, rigid, semi-rigid
# @ti.kernel
# def set_material_properties():
#     for p in range(n_particles):
#         if material[p] == 0:  # Soft material
#             mu[p] = 50
#             la[p] = 50
#         elif material[p] == 1:  # Rigid material
#             mu[p] = 200
#             la[p] = 200
#         elif material[p] == 2:  # semi-rigid matirial
#             mu[p] = 100
#             la[p] = 100

@ti.kernel
def grid_op(f: ti.i32):
    for i, j in ti.ndrange(n_grid, n_grid):
        inv_m = 1 / (grid_m_in[f, i, j] + 1e-10)
        v_out = inv_m * grid_v_in[f, i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
        if j < bound and v_out[1] < 0:
            v_out[1] = 0
        if j > n_grid - bound and v_out[1] > 0:
            v_out[1] = 0
        grid_v_out[f, i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[f, base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        x_avg[None] += (1 / n_particles) * x[steps - 1, i]


@ti.kernel
def compute_loss():
    dist = (x_avg[None] - ti.Vector(target))**2
    loss[None] = 0.5 * (dist[0] + dist[1])


def substep(s):
    p2g(s)
    grid_op(s)
    g2p(s)


# construct creature
def init_creature():
    start_x = origin_x
    start_y = origin_y
    particle_idx = 0  # Keep track of global particle index
    # construct head
    for i in range(head_width):
        for j in range(head_height):
            x[0, particle_idx] = [dx * (i * 1 + start_x), dx * (j * 0.7 + start_y)]
            particle_idx += 1

    # construct segments
    num_segs = 3 # random.randint(1,max_segs)
    max_segh = head_height
    min_segh = int(0.5*max_segh)
    start_x += head_width
    seg_height = head_height
    seg_width = 10

    for i in range(num_segs):
        if seg_height < 0: break
        # seg_width = random.randint(min_segw, max_segw)
        # seg_mat = random.choice(material)
        seg_height = random.randint(min_segh, max_segh)
        x_dens = random.random()
        y_dens = 0.2 + random.random()
        for i in range(seg_width):
            for j in range(seg_height):
                x[0, particle_idx] = [dx * (i * 1 + start_x), dx * (j * y_dens + start_y)]
                # material[particle_idx] = seg_mat
                particle_idx += 1
        max_segh = seg_height
        min_segh = int(0.5*max_segh)
        start_x += seg_width

    # construct tail / left appendage
    seg_width = random.randint(5,20)
    seg_height = random.randint(2,4)
    y_dens = random.random()
    for i in range(seg_width):
        for j in range(seg_height):
            x[0, particle_idx] = [dx * (origin_x - i * 1), dx * (origin_y + seg_height/2 + j*y_dens)]
            particle_idx += 1



init_v[None] = [0, 0]

for i in range(n_particles):
    F[0, i] = [[1, 0], [0, 1]]

# set_material_properties()
init_creature()

set_v()

losses = []
img_count = 0

gui = ti.GUI("Simple Differentiable MPM Solver", (640, 640), 0xAAAAAA)

for i in range(num_iterations):
    grid_v_in.fill(0)
    grid_m_in.fill(0)

    x_avg[None] = [0, 0]
    with ti.ad.Tape(loss=loss):
        set_v()
        for s in range(steps - 1):
            substep(s)

        compute_x_avg()
        compute_loss()

    l = loss[None]
    losses.append(l)
    grad = init_v.grad[None]
    print('loss=', l, '   grad=', (grad[0], grad[1]))
    learning_rate = 10
    init_v[None][0] -= learning_rate * grad[0]
    init_v[None][1] -= learning_rate * grad[1]

    # visualize
    x_np = x.to_numpy()
    for s in range(15, steps, 16):
        scale = 4
        gui.circles(x_np[s], color=0x112233, radius=1.5)
        gui.circle(target, radius=5, color=0xFFFFFF)
        img_count += 1
        gui.show()

plt.title("Optimization of Initial Velocity")
plt.ylabel("Loss")
plt.xlabel("Gradient Descent Iterations")
plt.plot(losses)
plt.show()
