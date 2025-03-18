from tracemalloc import start
import taichi as ti
import matplotlib.pyplot as plt
import random

real = ti.f32
ti.init(arch=ti.cuda, default_fp=real, device_memory_GB=1)

# material parameters
N = 60  # reduce to 30 if run out of GPU memory
n_particles = 1000
n_grid_x = 200
n_grid_y = 120
dx = 1 / max(n_grid_x,n_grid_y)
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
origin_x = 150
origin_y = 5

num_segs = 1
max_segh = head_height
min_segh = int(0.5*max_segh)
start_x = origin_x
start_y = origin_y
seg_height = head_height
seg_width = 10
#

# sim parameters
max_steps = 1024
steps = 1024
gravity = 9.8
target = [0.1, 0.7]
dim = 2
num_iterations = 25
#

# taichi stuff
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
                            shape=(max_steps, n_grid_x, n_grid_y),
                            needs_grad=True)
grid_v_out = ti.Vector.field(dim,
                             dtype=real,
                             shape=(max_steps, n_grid_x, n_grid_y),
                             needs_grad=True)
grid_m_in = ti.field(dtype=real,
                     shape=(max_steps, n_grid_x, n_grid_y),
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


# @ti.kernel
def set_v(freq):
    for i in range(n_particles):
        x_pos = x[0,i][0]
        v_x = ti.sin(x_pos*freq) * 2
        v[0, i] = [v_x, init_v[None][1]]


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

@ti.kernel
def grid_op(f: ti.i32):
    for i, j in ti.ndrange(n_grid_x, n_grid_y):
        inv_m = 1 / (grid_m_in[f, i, j] + 1e-10)
        v_out = inv_m * grid_v_in[f, i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
        if i > n_grid_x - bound and v_out[0] > 0:
            v_out[0] = 0
        if j < bound and v_out[1] < 0:
            v_out[1] = 0
        if j > n_grid_y - bound and v_out[1] > 0:
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

    # reward smoothness of segments
    # for seg in range(snake.get_num_segs() - 1):
    #     loss[None] += 0.1 * (snake.get_seg_height[seg] - snake.get_seg_height[seg + 1])**2

def substep(s):
    p2g(s)
    grid_op(s)
    g2p(s)


class Creature:
    def __init__(self, num_segs, seg_height, ydens):
        self.num_segs = num_segs
        self.seg_height = seg_height
        self.ydens = ydens

    def update_morph(d_seg_height, d_ydens):
        self.seg_height += d_seg_height
        self.ydens += d_ydens
    
    def get_num_segs(self): return self.num_segs
    def get_seg_height(self): return self.seg_height
    def get_ydens(self): return self.ydens


# create Creature object
num_segs = random.randint(1,max_segs)
seg_heights_tc = ti.field(dtype=real, shape=num_segs, needs_grad=True) # I ADDED -- so we can diff segment heights
seg_heights = []
ydens = []
for i in range(num_segs):
    seg_heights_tc[i] = random.randint(min_segh, max_segh)
    seg_heights.append(seg_heights_tc[i])
    ydens.append(0.2 + random.random())
snake = Creature(num_segs, seg_heights, ydens)

# physically construct creature
def constructCreature():
    particle_idx = 0
    start_x1 = start_x
    # construct head
    for i in range(head_width):
        for j in range(head_height):
            x[0, particle_idx] = [dx * (i * 1 + start_x), dx * (j * 0.7 + start_y)]
            particle_idx += 1

    # construct segments
    for k in range(snake.get_num_segs()):
        # print("numsegs: ", snake.get_num_segs())
        # print("k: ",k)
        # print("seg height: ", seg_height)
        for i in range(seg_width):
            for j in range(int(seg_heights_tc[k])):
                x[0, particle_idx] = [dx * (i * 1 + start_x1), dx * (j * snake.ydens[k] + start_y)]
                particle_idx += 1
        max_segh = seg_heights_tc[k]
        min_segh = int(0.5*max_segh)
        start_x1 += seg_width


for i in range(n_particles):
    F[0, i] = [[1, 0], [0, 1]]

# set initial velocities
init_v[None] = [0, 0]
init_freq = 5
freq = init_freq
set_v(freq)

losses = []
img_count = 0

gui = ti.GUI("Differentiable MPM Solver", (n_grid_x*5, n_grid_y*5), 0xAAAAAA)


# MAIN LOOP
for i in range(num_iterations):

    grid_v_in.fill(0)
    grid_m_in.fill(0)
    x_avg[None] = [0, 0]
    
    with ti.ad.Tape(loss=loss):
        constructCreature() # physically build creature
        set_v(freq)
        for s in range(steps - 1):
            substep(s)
        compute_x_avg()
        compute_loss()

    l = loss[None]
    losses.append(l)
    grad = init_v.grad[None]
    print('loss=', l, '   grad=', (grad[0], grad[1]))
    learning_rate = 10
    for seg in range(snake.get_num_segs()):
        seg_heights_tc[seg] -= learning_rate * seg_heights_tc.grad[seg]
        seg_heights[seg] = seg_heights_tc[seg]
    # init_v[None][0] = 0 # -= learning_rate * grad[0]
    # init_v[None][1] -= learning_rate * grad[1]

    # visualize
    x_np = x.to_numpy()
    for s in range(15, steps, 16):
        scale = 4
        gui.circles(x_np[s], color=0x112233, radius=1.5)
        gui.circle(target, radius=5, color=0xFFFFFF)
        img_count += 1
        gui.show()

plt.title("Optimization of Initial Vertical Velocity")
plt.ylabel("Loss")
plt.xlabel("Gradient Descent Iterations")
plt.plot(losses)
plt.show()
