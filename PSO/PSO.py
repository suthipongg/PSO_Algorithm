from cmath import inf
import random
from Test_Function import func_2D
import matplotlib.pyplot as plt
import numpy as np

w = 0.5
c1 = 1
c2 = 2

# set initial position and velocities of each particle
class initial():
    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions

    def pos(self):
        pos = np.zeros((self.num_dimensions, ))
        for i in range(self.num_dimensions):
            pos[i] = random.uniform(bounds[i][0]*0.2, bounds[i][1]*0.2)
        return pos
    
    def vel(self):
        vel = np.zeros((self.num_dimensions, ))
        for i in range(self.num_dimensions):
            vel[i] = random.uniform(bounds[i][0], bounds[i][1])
        return vel

# compute one particle
class Particle:
    def __init__(self, num_dimensions):
        init = initial(num_dimensions) # initial particle
        self.position = init.pos()  # particle position
        self.velocity = init.vel()  # particle velocities
        self.pos_p_best = []    # best position individual
        self.p_best = inf        # best fitness individual
        self.fitness = inf             # fitness individual
        self.num_dimensions = num_dimensions
                  

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

            vel_cognitive = c1 * r1 * (self.pos_p_best[i] - self.position[i])
            vel_social = c2 * r2 * (pos_g_best[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self):
        for i in range(0, self.num_dimensions):
            self.position[i] = self.position[i] + self.velocity[i]

            # adjust maximum position if necessary
            if self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]

            # adjust minimum position if neseccary
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]

def PSO(Func, num_particles, iteration, num_dimensions=2):
    g_best = inf                 # global best
    pos_g_best = []             # global best position
    ls_gbest_iter = np.zeros((iteration+1, )) # fitness of each iteration
    # establish the swarm
    swarm = [] # contains the dimensions of each particle
    for i in range(0, num_particles):
        swarm.append(Particle(num_dimensions))

    # begin optimization loop
    i = 0
    while i < iteration:

        # cycle through particles in swarm and evaluate fitness
        for j in range(0, num_particles):
            swarm[j].evaluate(Func)
            p_best = swarm[j].p_best
            # determine if current particle is the best (globally)
            if p_best < g_best:
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

def call_PSO(iteration=1000, num_particles=50):

    # Test function
    func = func_2D()
    all_func = func.all_func()
    name_array = np.empty((func.n_func, 1), dtype=str)
    ls_fit_array = np.empty((func.n_func, iteration+1))
    gbest_array = np.empty((func.n_func, 1))
    #figure, axis = plt.subplots(nrows=func.n_func, ncols=2, figsize=(15,30))
    #ls_iteration = list(range(1, iteration+1))

    for n in range(0, func.n_func):
        global bounds
        bounds = all_func[n,2]          # boundary function
        fitness_PSO = PSO(all_func[n,1], num_particles, iteration) # return global best & series fitness
        gbest = fitness_PSO[0]
        name_array[n, :] = all_func[n,0]
        ls_fit_array[n, :] = fitness_PSO[1]
        gbest_array[n, 0] = gbest
        #axis[n, 0].plot(ls_iteration, fitness_PSO[1])
        #axis[n, 0].set_title(all_func[n][0])
    return name_array, ls_fit_array, gbest_array
    #plt.show()

if __name__ == "__main__":
    info = call_PSO()
    print(info)