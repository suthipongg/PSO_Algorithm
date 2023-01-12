from cmath import inf
import random
from Test_Function import func_2D
import matplotlib.pyplot as plt
import list_well_initial 
import numpy as np

# set initial position and velocities of each particle
class WE_initial():
    def __init__(self, num_dimensions, bound):
        self.bound = bound
        self.num_dimensions = num_dimensions
    '''
    def pos(self):
        pos = []
        for i in range(self.num_dimensions):
            pos.append(random.uniform(self.bound[i][0], self.bound[i][1]))
        return pos'''


    def vel(self):
        vel = np.zeros((self.num_dimensions, ))
        for i in range(self.num_dimensions):
            vel[i] = random.uniform(self.bound[i][0]*0.2, self.bound[i][1]*0.2)
        return vel

# compute one particle
class WE_Particle:
    def __init__(self, num_dimensions, bound, w, c1, c2, i, we):

        init = WE_initial(num_dimensions, bound) # initial particle
        init_0_1 = we()[i]
        max_x, max_y, min_x, min_y = bound[0][1], bound[1][1], bound[0][0], bound[1][0]
        pos_x = min_x + (max_x-min_x)*init_0_1[0]
        pos_y = min_y + (max_y-min_y)*init_0_1[1]

        self.position = [pos_x, pos_y]  # particle position
        self.velocity = init.vel()  # particle velocities
        self.pos_p_best = []    # best position individual
        self.p_best = inf       # best fitness individual
        self.fitness = inf           # fitness individual
        self.num_dimensions = num_dimensions
        self.bound = bound          # boundary function

        self.w = w
        self.c1 = c1
        self.c2 = c2

    # evaluate current fitness
    def evaluate(self, Func):
        self.fitness = Func(self.position)

        # check to see if the current position is an individual best
        if self.fitness < self.p_best:
            self.pos_p_best = self.position
            self.p_best = self.fitness

    # update new particle velocity
    def update_velocity(self, pos_g_best):

        for i in range(0, self.num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = self.c1 * r1 * (self.pos_p_best[i] - self.position[i])
            vel_social = self.c2 * r2 * (pos_g_best[i] - self.position[i])
            self.velocity[i] = self.w * self.velocity[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self):
        for i in range(0, self.num_dimensions):
            self.position[i] = self.position[i] + self.velocity[i]

            # adjust maximum position if necessary
            if self.position[i] > self.bound[i][1]:
                self.position[i] = self.bound[i][1]

            # adjust minimum position if neseccary
            if self.position[i] < self.bound[i][0]:
                self.position[i] = self.bound[i][0]

def WEPSO(Func, bounds, num_particles, iteration, we, num_dimensions=2, w=0.5, c1=1, c2=2):
    g_best = inf                 # global best
    pos_g_best = []             # global best position
    ls_gbest_iter = np.zeros((iteration+1, )) # fitness of each iteration

    # establish the swarm
    swarm = [] # contains the dimensions of each particle
    for i in range(0, num_particles):
        swarm.append(WE_Particle(num_dimensions, bounds, w, c1, c2, i, we))

    # begin optimization loop
    i = 0
    while i < iteration:

        # cycle through particles in swarm and evaluate fitness
        for j in range(0, num_particles):
            swarm[j].evaluate(Func)
            p_best = swarm[j].p_best
            # determine if current particle is the best (globally)
            if p_best < g_best or g_best == -1:
                pos_g_best = list(swarm[j].position)
                g_best = p_best

        # cycle through swarm and update velocities and position
        for j in range(0, num_particles):
            swarm[j].update_velocity(pos_g_best)
            swarm[j].update_position()
        
        ls_gbest_iter[i] = g_best
        i += 1

    # final results
    return [g_best, ls_gbest_iter]

def call_WEPSO(iteration=1000, num_particles=50):

    # Test function
    func = func_2D()
    all_func = func.all_func()
    name_array = np.empty((func.n_func, 1), dtype=str)
    ls_fit_array = np.empty((func.n_func, iteration+1))
    gbest_array = np.empty((func.n_func, 1))
    #figure, axis = plt.subplots(nrows=func.n_func, ncols=2, figsize=(15,30))
    #ls_iteration = list(range(1, iteration+1))

    for n in range(0, func.n_func):
        fitness_WEPSO = WEPSO(all_func[n,1], all_func[n,2], num_particles, iteration, list_well_initial.ls_init_WELL512)     # return global best & series fitness
        gbest = fitness_WEPSO[0]
        name_array[n, :] = all_func[n,0]
        ls_fit_array[n, :] = fitness_WEPSO[1]
        gbest_array[n, 0] = gbest
        #axis[n, 0].plot(ls_iteration, fitness_PSO[1])
        #axis[n, 0].set_title(all_func[n][0])
    return name_array, ls_fit_array, gbest_array
    #plt.show()

if __name__ == "__main__":
    info = call_WEPSO()
    print(info)