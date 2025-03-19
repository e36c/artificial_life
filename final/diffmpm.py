import taichi as ti
import copy
import os
import math
import numpy as np
import matplotlib.pyplot as plt


dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
# TODO: update
mu = E
la = E
max_steps = 2048
steps = 1024
gravity = 3.8
target = [1, 0.2]



def allocate_fields():
    global real
    real = ti.f32
    ti.init(default_fp=real, arch=ti.cpu, flatten_if=True,device_memory_GB=4)

    global actuator_id
    actuator_id = ti.field(ti.i32)
    global particle_type
    particle_type = ti.field(ti.i32)
    global scalar
    scalar = lambda: ti.field(dtype=real)
    global vec
    vec = lambda: ti.Vector.field(dim, dtype=real)
    global mat
    mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

    global x, v
    x,v = vec(), vec()
    global grid_v_in, grid_m_in
    grid_v_in, grid_m_in = vec(), scalar()
    global grid_v_out
    grid_v_out = vec()
    global C, F
    C,F = mat(), mat()

    global loss
    loss = scalar()

    global n_sin_waves
    n_sin_waves = 4
    global weights
    weights = scalar()
    global bias
    bias = scalar()
    global x_avg
    x_avg = vec()

    global actuation
    actuation = scalar()
    global actuation_omega
    actuation_omega = 20
    global act_strength
    act_strength = 4
    
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)

    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, x_avg)

    ti.root.lazy_grad()


@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]


@ti.kernel
def clear_particle_grad():
    # for all time steps and all particles
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]


@ti.kernel
def clear_actuation_grad():
    for t, i in actuation:
        actuation[t, i] = 0.0


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)

        act_id = actuator_id[p]

        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        # ti.print(act)

        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base +
                          offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass


bound = 3
coeff = 0.5


@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0

        grid_v_out[i, j] = v_out


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
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


@ti.kernel
def compute_actuation(t: ti.i32):
    for i in range(n_actuators):
        act = 0.0
        for j in ti.static(range(n_sin_waves)):
            act += weights[i, j] * ti.sin(actuation_omega * t * dt +
                                          2 * math.pi / n_sin_waves * j)
        act += bias[i]
        actuation[t, i] = ti.tanh(act)


@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])


@ti.kernel
def compute_loss():
    dist = x_avg[None][0]
    loss[None] = -dist


@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    compute_actuation(s)
    p2g(s)
    grid_op()
    g2p(s)


@ti.ad.grad_for(advance)
def advance_grad(s):
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    compute_actuation.grad(s)


def forward(total_steps=steps):
    # simulation
    for s in range(total_steps - 1):
        advance(s)
    x_avg[None] = [0, 0]
    compute_x_avg()
    compute_loss()


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

    def add_rect(self, x, y, w, h, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act

def robot(scene, fins):
    '''
    structure of arguments:
    - scene is scene
    - fins is a list of 4 fins, with the below structure
    - fin data: [pos, length, width]
    - pos should be in the range [0:1]
    '''
    top = fins[0]
    rhs = fins[1]
    bot = fins[2]
    lhs = fins[3]
    offset = [0.1, 0.03]                        # scene offset
    pos = [0.1, 0.1]                           # origin of robot
    ht = 0.1
    wd = ht
    top_edge = pos[1] + ht
    rhs_edge = pos[0] + wd
    bot_edge = pos[1]
    lhs_edge = pos[0]
    scene.set_offset(offset[0], offset[1])                                          # set offset to origin
    scene.add_rect(pos[0], pos[1], wd, ht, -1)                                      # carapace, unactuated
    if (top[1]>0 and top[2]>0): scene.add_rect(top[0]*wd+lhs_edge-top[2]/2, top_edge, top[2], top[1], 0)        # top fin
    if (rhs[1]>0 and rhs[2]>0): scene.add_rect(rhs_edge, -rhs[0]*ht+top_edge-rhs[2]/2, rhs[1], rhs[2], 1)        # right fin
    if (bot[1]>0 and bot[2]>0): scene.add_rect(-bot[0]*wd+rhs_edge-bot[2]/2, bot_edge-bot[1], bot[2], bot[1], 2) # bottom fin
    if (lhs[1]>0 and lhs[2]>0): scene.add_rect(lhs_edge-lhs[1], lhs[0]*ht+bot_edge-lhs[2]/2, lhs[1], lhs[2], 3) # left fin
    scene.set_n_actuators(4)


gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)


def visualize(s, folder):
    aid = actuator_id.to_numpy()
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s]
    actuation_ = actuation.to_numpy()
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            act = actuation_[s - 1, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')


def main():

    # loop variables
    total_ol = 2
    total_il = 15
    
    # outer loop variables
    losses_outer = [0.0]*total_ol
    plateau_counter = 0
    plateau_limit = 250

    # generate seed creature
    # test robot
    # test_fin = [0.3, 0.1, 0.1]
    # test_fin_tl = [0.7, 0.15, 0.03]
    # test_fin_br = test_fin_tl
    # test_fin_br[0] = 1-test_fin_tl[0]
    fint = [0.7, 0.1, 0.1]
    finr = [0.7, 0.1, 0.21]
    finb = [0.7, 0.1, 0.1]
    finl = [0.3,0.1,0.0]
    # fint = [0.7, 0.1, 0.1]
    # finr = [0.7, 0.1, 0.1]
    # finb = [0.7, 0.1, 0.1]
    # finl = [0.7, 0.1, 0.1]
    fint,finr,finb,finl = [0.7, 0.1, 0.1], [0.123452346, 0.1, 0.21], [0.7, 0.1, 0.1], [0.3, 0.1, 0.0]
    fint,finr,finb,finl = [0.7, 0.1, 0.1], [0.7, 0.1, 0.21], [0.7, 0.127824976043278, 0.1], [0.3, 0.1, 0.0]
    # fint,finr,finb,finl = [0.2, 0.052394857, 0.1], [0.7, 0.1, 0.21], [0.7, 0.1, 0.1], [0.3, 0.1, 0.0]

    # list with all fins
    fins = [[[0,0,0] for _ in range(4)] for _ in range(total_ol)]      # fins holds all fins over all iterations of total_ol
    fins[0] = [fint,finr,finb,finl]
    
    #helpful
    fin_names = ["top","rhs","bot","lhs"]

    outer_iter = 0
    # OUTER LOOP START ---------------------------------------------------------------------
    while outer_iter < total_ol:        # outer for loop

        # save new fins!
        print("---------------------------")
        print("fins for outer_iter =",outer_iter)
        for i in fins: print(i)
        # for i in range(4):
        #     print(fin_names[i]+":", fins[outer_iter][i])
        print("---------------------------")

        # reset taichi
        ti.reset()

        # initialization
        scene = Scene()
        robot(scene,fins[outer_iter])
        scene.finalize()
        allocate_fields()

        for i in range(n_actuators):
            for j in range(n_sin_waves):
                weights[i, j] = np.random.randn() * 0.01

        for i in range(scene.n_particles):
            x[0, i] = scene.x[i]
            F[0, i] = [[1, 0], [0, 1]]
            actuator_id[i] = scene.actuator_id[i]
            particle_type[i] = scene.particle_type[i]

        losses_inner = []
        # INNER LOOP START -------------------------------------------------------------
        for inner_iter in range(total_il):    # inner for loop
            with ti.ad.Tape(loss):
                forward()
            l = loss[None]
            losses_inner.append(l)
            print('i=', inner_iter, 'loss=', l)
            learning_rate = 0.1

            for i in range(n_actuators):
                for j in range(n_sin_waves):
                    # print(weights.grad[i, j])
                    weights[i, j] -= learning_rate * weights.grad[i, j]
                bias[i] -= learning_rate * bias.grad[i]

            if (inner_iter+1) % total_il == 0:  # visualize only once at the final inner generation
                # visualize
                forward(1500)
                for s in range(30, 1500, 16):
                    visualize(s, 'diffmpm/inner_iter{:03d}/'.format(inner_iter))

        # INNER LOOP END ----------------------------------------------------------------

        # print loss plot
        print_loss_plot = False
        if print_loss_plot:
            # ti.profiler_print()
            plt.title("Optimization of Initial Velocity")
            plt.ylabel("Loss")
            plt.xlabel("Gradient Descent Iterations")
            plt.plot(losses_inner)
            plt.show()
        
        # log last loss of inner loop
        losses_outer[outer_iter] = (losses_inner[-1])

        # check if got better, oth. throw out
        if (outer_iter!=0) and losses_outer[outer_iter] >= losses_outer[outer_iter-1]:
            print("Bad generation! in outer_iter " + str(outer_iter) + ". Reverting mutation")
            outer_iter -= 1         # throw out this gen, and loop again
            print("outer_iter:",outer_iter)

        ### mutate, if not already the last generation
        # print("fins before assignment")
        # for i in fins: print(i)
        if outer_iter<total_ol-1:
            fins[outer_iter+1] = copy.deepcopy(fins[outer_iter])
            # print("fins after assignment")
            # for i in fins: print(i)
            dice = np.random.rand()
            scale = 0.3
            findex = np.random.randint(0,4)
            dimdex = np.random.randint(0,3)
            if dimdex==0: scale = 1
            if dice<1:      fins[outer_iter+1][findex][dimdex] = scale*np.random.rand()
            elif dice<0.66: fins[outer_iter+1][np.random.randint(0,4)][np.random.randint(0,3)] = scale*np.random.rand()
            else:           fins[outer_iter+1][np.random.randint(0,4)][np.random.randint(0,3)] = scale*np.random.rand()

        plateau_counter += 1        #  plateau counter++, so we don't get stuck forever bc of many bad gens
        if plateau_counter > plateau_limit: break

        outer_iter += 1

    # OUTER LOOP END ------------------------------------------------------------------------

    
    print("---------------------------")
    print("SUMMARY:\n\tlowest loss achieved: "+str(min(losses_outer))+"\n\tfinal evolved robot fins:")
    for i in fins: print("\t\t",i)
    print("---------------------------")

    # print outer loop loss plot
    # ti.profiler_print()
    plt.title("Optimization of Morphology by Selection")
    plt.ylabel("Min(loss)")
    plt.xlabel("Survived Generation")
    plt.plot(losses_outer)
    plt.show()



if __name__ == '__main__':
    main()
