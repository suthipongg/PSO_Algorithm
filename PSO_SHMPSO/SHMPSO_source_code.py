from cmath import inf, pi
from math import gamma, sin
import sys
sys.path.insert(1, '/.../PSO_Algorithm/Test_Function_file')
from Test_Function_numpy import func_2D
import matplotlib.pyplot as plt
import numpy as np

scale_velocity = 0.2

w = 0.729
c = 0.5
c1 = 1.49445
c2 = 1.49445
f = 0.5     # mutation factor
s = 0.5     # scale factor
m = 6       # time of falling into local optimum
prob = 0.5 
n = 1

beta = 1.5
sigma = ((gamma(1+beta) * sin(pi*beta/2))/(gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)

bounds = 0

# set initial position and velocities of each particle
def SHM_initial(num_dimensions, num_particles):
    pos = bounds[0] + (bounds[1] - bounds[0])*np.random.rand(num_particles, num_dimensions)
    vel = (bounds[0] + (bounds[1] - bounds[0])*np.random.rand(num_particles, num_dimensions))*scale_velocity
    return pos, vel

# compute one particle
class SHMPSO:
    def __init__(self, num_dimensions, num_particles, max_iteration):
        self.fall_time = 0
        self.ave1 = 0
        self.iteration = 0
        self.step = 0
        self.candidate_b = self.candidate_s = np.zeros((num_particles, num_dimensions))  # num_dimensions, num_dimensions
        self.num_dimensions = num_dimensions
        self.num_particles = num_particles
        self.max_iteration = max_iteration
        self.N_dominant = int(num_particles*s)
        self.N_poor = num_particles - self.N_dominant
        self.row_index_current_gbest = 0
        self.arange_number_particle = np.arange(num_particles)

        self.particle_dominant = self.particle_poor = np.zeros((num_particles, num_dimensions)) # True, False array
        self.position_dominant = self.position_poor = np.zeros((num_particles, num_dimensions))
        self.velocity_dominant = self.velocity_poor = np.zeros((num_particles, num_dimensions))

        self.position, self.velocity = SHM_initial(num_dimensions, num_particles)  # particle initial
        self.fitness = np.zeros((num_particles, 1))                 # num_particles, 1
        self.p_best = np.inf + np.zeros((num_particles, 1))         # num_particles, 1
        self.pos_pbest = np.zeros((num_particles, num_dimensions))  # num_dimensions, num_dimensions
        self.g_best = inf
        self.pos_gbest = np.zeros((num_dimensions, ))               # num_dimensions, 
        self.candidate = np.zeros((num_particles, num_dimensions))  # num_dimensions, num_dimensions

        self.ls_gbest_iter = np.zeros((max_iteration+1, ))
        
    # evaluate fitness all particle
    def evaluate(self, Func):
        self.fitness = np.reshape(Func(self.position), (self.num_particles, 1))  # num_particles, 
        self.pos_pbest = np.nan_to_num((self.fitness < self.p_best)*self.position) + np.nan_to_num((self.p_best <= self.fitness)*self.position) # position pbest
        self.p_best = np.nan_to_num((self.fitness < self.p_best)*self.fitness) + np.nan_to_num((self.p_best <= self.fitness)*self.p_best) # fitness pbest
        self.row_index_current_gbest = np.argmin(self.p_best)
        if self.p_best[self.row_index_current_gbest, 0] <= self.g_best:
            self.g_best = self.p_best[self.row_index_current_gbest, 0]
            self.fall_time = 0
        else: self.fall_time += 1

    # check position all particle
    def check_bound(self):
        # adjust maximum position if necessary
        self.position = (self.position>bounds[1])*bounds[1] + (self.position<=bounds[1])*self.position
        # adjust minimum position if neseccary
        self.position = (self.position<bounds[0])*bounds[0] + (self.position>=bounds[0])*self.position

        # adjust maximum position if necessary
        self.velocity = (self.velocity>(bounds[1]*scale_velocity))*bounds[1]*scale_velocity + (self.velocity<=(bounds[1]*scale_velocity))*self.velocity
        # adjust minimum position if neseccary
        self.velocity = (self.velocity<(bounds[0]*scale_velocity))*bounds[0]*scale_velocity + (self.velocity>=(bounds[0]*scale_velocity))*self.velocity    

    def split_particle(self):
        split_fitness = np.sort(self.fitness, 0)[self.N_poor, 0]
        self.particle_dominant = self.fitness < split_fitness               # index of dominant is True
        self.particle_poor = self.fitness >= split_fitness                  # index of poor is True
        self.position_dominant = self.particle_dominant * self.position     # position index of poor is 0
        self.position_poor = self.particle_poor * self.position             # position index of fominant is 0
        self.velocity_dominant = self.particle_dominant * self.velocity
        self.velocity_poor = self.particle_poor * self.velocity

    # update new particle dominant velocity all particle
    def update_velocity_dominant(self):
        # num_particles * 1
        mean_dimension = np.reshape(np.sum(self.position_dominant, 1)/self.num_dimensions, (self.num_particles, 1))
        # 1 * num_dimensions
        mean_position_dominant = np.reshape(np.sum(self.position_dominant, 0)/self.N_dominant, (1, self.num_dimensions))
        expo = np.exp(-(self.position_dominant - mean_dimension))
        rho = 1 / (1 + expo)                                                        # num_particles * num_dimensions
        r = np.random.rand(self.num_particles,self.num_dimensions)                  # num_particles * num_dimensions
        m_best = rho*r*mean_dimension + (1-rho)*(1-r)*mean_position_dominant        # num_particles * num_dimensions
        r = np.random.rand(self.num_particles,self.num_dimensions)                  # num_particles * num_dimensions
        self.velocity_dominant = (w * self.velocity_dominant + c1*r*(m_best - self.position_dominant))*self.particle_dominant

    def update_position_dominant_bigger_than_m(self):
        square_metrix_dominant_position = np.tile(self.particle_dominant, (1, self.num_particles))  # create column = row to choose particle
        identity_dominant = np.identity(self.num_particles) == 0                                    # I metrix : dimension = square_metrix_dominant_position
        metrix_dominant = (square_metrix_dominant_position*identity_dominant).T                     # not choose particle in same index of individual
        array_index = np.random.rand(*metrix_dominant.shape)*metrix_dominant                        # random metrix : dimension = square_metrix_dominant_position
        Xa_index = np.argmax(array_index, axis=1)                                                   # choose max index only , it is random process to choosse dominant particle
        metrix_dominant[self.arange_number_particle, Xa_index] = 0                                  # set index that choose last command to 0 for not choose
        array_index = np.random.rand(*metrix_dominant.shape)*metrix_dominant                        # random metrix again
        Xb_index = np.argmax(array_index, axis=1)                                                   # choose max index that don't have Xa_index
        Xa = self.position_dominant[Xa_index, :]                                                    # dominant position a : num_particles * num_dimensions
        Xb = self.position_dominant[Xb_index, :]                                                    # dominant position b : num_particles * num_dimensions
        self.position_dominant = (self.position[self.row_index_current_gbest] + f*(Xa - Xb)) * self.particle_dominant

    # update the particle position based off new velocity updates
    def update_position_dominant_smaller_than_m(self):
        try: expo = np.exp(-self.fitness/self.ave1)
        except: expo = inf + np.zeros((self.num_particles, 1))
        try: phi_1 = (1+expo)**(self.iteration+1)
        except: phi_1 = inf + np.zeros((self.num_particles, 1))
        phi = 1/phi_1
        self.position_dominant = self.position_dominant*phi + self.velocity_dominant

    def step_size(self):
        u = np.random.normal(0, sigma, size=(self.num_particles, self.num_dimensions))
        v = np.random.normal(0, 1, size=(self.num_particles, self.num_dimensions))
        self.step = u / abs(v)**(1/beta)

    def candidate_big(self):
        self.candidate_b = self.step * self.position[self.row_index_current_gbest]
    
    def candidate_small(self):
        metrix_dominant = np.tile(self.particle_dominant, (1, self.num_particles)).T      # create column = row to choose particle
        metrix_dominant[:, self.row_index_current_gbest] = 0                              # set inedx current iteration global optimum is 0
        array_index = np.random.rand(*metrix_dominant.shape)*metrix_dominant              # random metrix : dimension = metrix_dominant
        Xn_index = np.argmax(array_index, axis=1)                                         # choose max index only , it is random process to choosse dominant particle
        metrix_dominant[self.arange_number_particle, Xn_index] = 0                        # set index that choose last command to 0 for not choose
        array_index = np.random.rand(*metrix_dominant.shape)*metrix_dominant              # random metrix again
        Xm_index = np.argmax(array_index, axis=1)                                         # choose max index that don't have Xa_index
        Xn_fit = self.fitness[Xn_index, 0]                                                # Xn fitness
        Xm_fit = self.fitness[Xm_index, 0]                                                # Xm fitness
        Xn = (Xn_fit < Xm_fit) * Xn_index + (Xn_fit >= Xm_fit) * Xm_index                 # index that fitness is minimum
        self.candidate_s = self.step * self.position[Xn]

    def candidate_sum(self):
        self.step_size()
        self.candidate_big()
        self.candidate_small()
        r = np.random.rand(self.num_particles,1)
        self.candidate = self.candidate_b*(prob<r) + self.candidate_s*(prob>=r)
    
    def update_poor_velocity(self):
        r1 = np.random.rand(self.num_particles,self.num_dimensions)
        r2 = np.random.rand(self.num_particles,self.num_dimensions)
        vel_cognitive = c1 * r1 * (self.pos_pbest*self.particle_poor - self.position_poor)
        vel_social = c2 * r2 * (self.candidate - self.position_poor)*self.particle_poor
        self.velocity_poor = w * self.velocity_poor + vel_cognitive + vel_social

    def update_poor_position(self):
        self.position_poor = self.position_poor + self.velocity_poor

    def sum_vel_pos(self):
        self.velocity = self.velocity_poor + self.velocity_dominant
        self.position = self.position_poor + self.position_dominant

    def __call__(self, Func):
        self.evaluate(Func)
        self.split_particle()
        self.ave1 = np.sum(self.fitness*self.particle_dominant)/self.N_dominant
        while self.iteration < self.max_iteration:
            self.update_velocity_dominant()
            if self.fall_time > m: self.update_position_dominant_smaller_than_m()
            else: self.update_position_dominant_bigger_than_m()
            self.candidate_sum()
            self.update_poor_velocity()
            self.update_poor_position()
            self.sum_vel_pos()
            self.check_bound()
            self.ls_gbest_iter[self.iteration] = self.g_best
            self.evaluate(Func)
            self.split_particle()
            
            self.iteration += 1
        self.ls_gbest_iter[self.iteration] = self.g_best
        return self.g_best

def call_SHMPSO(iteration=1000, num_particles=50):
   
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
        bounds = all_func[n, 2]
        fitness_PSO = SHMPSO(2, num_particles, iteration)    # global best & series fitness
        gbest = fitness_PSO(all_func[n,1])
        name_array[n, :] = all_func[n,0]
        ls_fit_array[n, :] = fitness_PSO.ls_gbest_iter
        gbest_array[n, 0] = gbest
        #axis[n, 0].plot(ls_iteration, fitness_PSO[1])
        #axis[n, 0].set_title(all_func[n][0])
    return name_array, ls_fit_array, gbest_array
    #plt.show()
if __name__ == "__main__":
    info = call_SHMPSO()
    print(info[1])