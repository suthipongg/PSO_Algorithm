from cmath import e, inf, pi
from math import gamma, sin, exp
import random

from sklearn.tree import export_text
from Test_Function_file import func_2D
import matplotlib.pyplot as plt
import numpy as np

w = 0.729
c = 0.5
c1 = 1.49445
c2 = 1.49445
f = 0.5     # mutation factor
s = 0.5     # scale factor
m = 6       # time of falling into local optimum
prob = 0.5 
beta = 1.5
n = 1

# set initial position and velocities of each particle
class SHM_initial():
    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions

    def pos(self):
        pos = np.zeros((self.num_dimensions, ))
        for i in range(self.num_dimensions):
            pos[i] = random.uniform(bounds[i][0], bounds[i][1])
        return pos
    
    def vel(self):
        vel = np.zeros((self.num_dimensions, ))
        for i in range(self.num_dimensions):
            vel[i] = random.uniform(bounds[i][0], bounds[i][1])
        return vel

# compute one particle
class SHM_Particle:
    def __init__(self, num_dimensions):
        init = SHM_initial(num_dimensions) # initial particle
        self.position = init.pos()  # particle position
        self.velocity = init.vel()  # particle velocities
        self.pos_p_best = []    # best position individual
        self.p_best = inf       # best fitness individual
        self.fitness = inf           # fitness individual
        self.num_dimensions = num_dimensions
        self.candidate = []
        self.sigma = ((gamma(1+beta) * sin(pi*beta/2))/(gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)

    # evaluate current fitness
    def evaluate(self, Func):
        self.fitness = Func(self.position)

        # check to see if the current position is an individual best
        if self.fitness < self.p_best:
            self.pos_p_best = self.position
            self.p_best = self.fitness

    def check_pos_bound(self, index):
        # adjust maximum position if necessary
        if self.position[index] > bounds[index][1]:
            self.position[index] = bounds[index][1]

        # adjust minimum position if neseccary
        if self.position[index] < bounds[index][0]:
            self.position[index] = bounds[index][0]        

    # update new particle velocity
    def update_velocity_dominant(self, Xi_mean, N_d):
        for i in range(0, self.num_dimensions):
            try: expo = exp(-(self.position[i] - Xi_mean[i]/self.num_dimensions))
            except : expo = inf
            rho = 1 / (1 + expo)
            r = random.random()
            m_best = rho*r*np.sum(self.position)/self.num_dimensions + (1-rho)*(1-r)*Xi_mean[i]/N_d
            self.velocity[i] = w * self.velocity[i] + c*r*(m_best - self.position[i])

    def update_position_dominant_bigger_than_m(self, Pgd, Xa, Xb):
        for i in range(0, self.num_dimensions):
            self.position[i] = Pgd[i] + f*(Xa[i] - Xb[i])
            self.check_pos_bound(i)

    # update the particle position based off new velocity updates
    def update_position_dominant_smaller_than_m(self, fit_mean, N_dom, iteration, sum_all_fitness):
        try: expo = exp(-fit_mean*sum_all_fitness/N_dom)
        except: expo = inf
        phi = 1/(1+expo)**(iteration)
        for i in range(0, self.num_dimensions):
            self.position[i] = phi*self.position[i] + self.velocity[i]
            self.check_pos_bound(i)

    def step_size(self):
        u = np.random.normal(0, self.sigma, self.num_dimensions)
        v = np.random.normal(0, 1, self.num_dimensions)
        s = u / abs(v)**(1/beta) 
        return s
    
    def candidate_big(self, global_dominant_pos, N):
        self.candidate = []
        step = self.step_size()
        for i in range(0, self.num_dimensions):
            self.candidate.append(global_dominant_pos[i] * step[i])
    
    def candidate_small(self, N, Xn):
        self.candidate = []
        step = self.step_size()
        for i in range(0, self.num_dimensions):
            self.candidate.append(step[i] * Xn[i])
    
    def update_poor_velocity(self):
        for i in range(0, self.num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.pos_p_best[i] - self.position[i])
            vel_social = c2 * r2 * (self.candidate[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + vel_cognitive + vel_social

    def update_poor_position(self):
        for i in range(0, self.num_dimensions):
            self.position[i] = self.position[i] + self.velocity[i]
            self.check_pos_bound(i)

def SHMPSO(Func, num_particles, iteration, num_dimensions=2):
    global m

    g_best = inf                # global best
    pos_g_best = []             # global best position
    series_g_best = []          # fitness of each iteration
    N_dom = int(s * num_particles)     # No. of dominant particle
    N_poo = num_particles - N_dom   # No. of poor particle
    sum_fall_time = 0

    ls_fit_all_par = np.zeros((num_particles, ))    # list fitness current iteration all particle
    # establish the swarm
    swarm = [] # contains the dimensions of each particle
    for k in range(0, num_particles):
        swarm.append(SHM_Particle(num_dimensions))
        swarm[k].evaluate(Func)
        p_best = swarm[k].p_best
        ls_fit_all_par[k] = swarm[k].fitness
        if p_best < g_best:
            pos_g_best = swarm[k].position
            g_best = p_best
    ls_fit_all_par_sort = np.sort(ls_fit_all_par) # sort fitness descending 
    sep_par_val = ls_fit_all_par_sort[N_dom] # separate particle value : dominant < , poor >=

    # begin optimization loop
    i = 0
    while i < iteration:
        Xi_dom_mean = np.zeros((num_dimensions, ))
        fit_mean = 0
        dom_index = []
        sum_all_fitness = 0
        global_dominant = inf
        global_dominant_pos = []
        # X_mean dominant particle
        for par in range(0, num_particles):
            sum_all_fitness += swarm[par].fitness
            if ls_fit_all_par[par] <= sep_par_val:
                Xi_dom_mean += swarm[par].position
                fit_mean += swarm[par].fitness
                dom_index.append(par)
                if swarm[par].fitness < global_dominant:
                    global_dominant = swarm[par].fitness
                    global_dominant_pos = swarm[par].position
        Xi_dom_mean /= N_dom
        fit_mean /= N_dom
        
        # cycle through swarm and update velocities and position
        for j in range(0, num_particles):

            # dominant particle
            if ls_fit_all_par[j] < sep_par_val:
                swarm[j].update_velocity_dominant(Xi_dom_mean, N_dom)
                if sum_fall_time > m: 
                    a = random.randint(0, num_dimensions)
                    b = random.randint(0, num_dimensions)
                    while a == b: b = random.choice(dom_index)
                    Xa = swarm[a].position
                    Xb = swarm[b].position
                    swarm[j].update_position_dominant_bigger_than_m(pos_g_best, Xa, Xb)
                else: swarm[j].update_position_dominant_smaller_than_m(fit_mean, N_dom, i+1, sum_all_fitness)

            # poor particle
            else:
                if prob > random.random(): swarm[j].candidate_big(global_dominant_pos, num_particles)
                else: 
                    m = random.choice(dom_index)
                    n = random.choice(dom_index)
                    if swarm[m].fitness < swarm[n].fitness: Xn = swarm[m].position
                    else: Xn = swarm[n].position
                    swarm[j].candidate_small(num_particles, Xn)
                swarm[j].update_poor_velocity()
                swarm[j].update_poor_position()

        i += 1
        series_g_best.append(g_best) # send present global best

        for k in range(0, num_particles):
            swarm[k].evaluate(Func)
            p_best = swarm[k].p_best
            ls_fit_all_par[k] = swarm[k].fitness
            if p_best < g_best:
                pos_g_best = swarm[k].position
                g_best = p_best
                sum_fall_time = 0
            else: sum_fall_time += 1

        ls_fit_all_par_sort = np.sort(ls_fit_all_par) # sort fitness descending 
        sep_par_val = ls_fit_all_par_sort[N_dom] # separate particle value : dominant < , poor >=
    series_g_best.append(g_best) # send present global best

    # final results
    return [g_best, series_g_best]

def call_SHMPSO(iteration=50, num_particles=50):
   
    # Test function
    ls_fitness = []
    func = func_2D()
    all_func = func.all_func()

    #figure, axis = plt.subplots(nrows=func.n_func, ncols=2, figsize=(15,30))
    #ls_iteration = list(range(1, iteration+1))

    for n in range(0, func.n_func):
        global bounds
        bounds = all_func[n,2]          # boundary function
        fitness_PSO = SHMPSO(all_func[n,1], num_particles, iteration)    # global best & series fitness
        print("Fitness",all_func[n][0], ":" , fitness_PSO[0])
        #axis[n, 0].plot(ls_iteration, fitness_PSO[1])
        #axis[n, 0].set_title(all_func[n][0])

    #plt.show()

if __name__ == "__main__":
    call_SHMPSO()