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
def SHM_initial(num_dimensions, num_particles, num_we):
    we44497 = np.array([[0.0352792, 0.0199591],
    [0.972657, 0.13886],
    [0.146975, 0.12694],
    [0.00279465, 0.000736522],
    [0.00377171, 0.482728],
    [0.134356, 0.387609],
    [0.0144807, 0.133271],
    [0.983048, 0.904877],
    [0.00276271, 0.0142066],
    [0.305827, 0.980386],
    [0.146731, 0.146548],
    [0.144717, 0.39138],
    [0.1458, 0.0178036],
    [0.864824, 0.0523703],
    [0.146181, 0.137691],
    [0.146136, 0.145907],
    [0.145968, 0.129398],
    [0.145205, 0.160464],
    [0.145922, 0.145953],
    [0.145769, 0.130206],
    [0.130404, 0.191958],
    [0.407387, 0.819565],
    [0.00260948, 0.0709879],
    [0.302574, 0.676513],
    [0.00259493, 0.238053],
    [0.00370858, 0.487809],
    [0.412087, 0.252487],
    [0.0033577, 0.981942],
    [0.0038307, 0.00375424],
    [0.00302134, 0.75112],
    [0.00299116, 0.063875],
    [0.00354017, 0.753571],
    [0.0191197, 0.00347963],
    [0.0035579, 0.284287],
    [0.00192326, 0.00189245],
    [0.303276, 0.740176],
    [0.00180084, 0.392396],
    [0.00173977, 0.131453],
    [0.017304, 0.118931],
    [0.00161805, 0.0430139],
    [0.00155664, 0.00447005],
    [0.303643, 0.669725],
    [0.00143463, 0.417026],
    [0.00137375, 0.751343],
    [0.0169374, 0.00128227],
    [0.00125145, 0.186952],
    [0.00119041, 0.432895],
    [0.00113004, 0.284186],
    [0.00106857, 0.551651],
    [0.00100774, 0.4668]])
    we512 = np.array([[0.812988, 0.194087],
[0.74689, 0.234253],
[0.607239, 0.316953],
[0.226929, 0.38745],
[0.538458, 0.673871],
[0.160426, 0.997867],
[0.851272, 0.598481],
[0.450097, 0.12832],
[0.768113, 0.853622],
[0.363201, 0.138372],
[0.21819, 0.549466],
[0.348513, 0.872558],
[0.0211213, 0.170215],
[0.760622, 0.507429],
[0.17989, 0.056341],
[0.171056, 0.976813],
[0.0830118, 0.50306],
[0.603178, 0.961601],
[0.360069, 0.575339],
[0.898025, 0.126539],
[0.805037, 0.20386],
[0.269319, 0.0135013],
[0.242239, 0.0944882],
[0.323608, 0.891294],
[0.657497, 0.832687],
[0.546105, 0.0436127],
[0.16577, 0.97504],
[0.182906, 0.816533],
[0.626215, 0.169654],
[0.882774, 0.00581199],
[0.312687, 0.384184],
[0.455356, 0.65979],
[0.501294, 0.184349],
[0.705357, 0.300356],
[0.420769, 0.776489],
[0.436691, 0.000198277],
[0.913217, 0.173794],
[0.335123, 0.961486],
[0.286126, 0.535322],
[0.0190086, 0.117888],
[0.905931, 0.770653],
[0.502669, 0.634161],
[0.995591, 0.377742],
[0.318025, 0.218958],
[0.416671, 0.796536],
[0.33627, 0.542181],
[0.373202, 0.731256],
[0.616567, 0.851952],
[0.863779, 0.988756],
[0.84224, 0.442222]])
    we1024 = np.array([[0.767712, 0.708479],
[0.529068, 0.393935],
[0.162195, 0.309752],
[0.930798, 0.49355],
[0.889497, 0.0838963],
[0.951464, 0.0769261],
[0.916689, 0.772763],
[0.917097, 0.867216],
[0.629471, 0.375865],
[0.240077, 0.357004],
[0.367146, 0.159781],
[0.279899, 0.757958],
[0.727084, 0.182974],
[0.0370337, 0.197886],
[0.656636, 0.163379],
[0.794686, 0.0728167],
[0.743162, 0.617287],
[0.153879, 0.541245],
[0.126998, 0.641736],
[0.149256, 0.0687378],
[0.271647, 0.696038],
[0.595386, 0.98773],
[0.0605284, 0.774843],
[0.915434, 0.848157],
[0.639127, 0.757883],
[0.203744, 0.410127],
[0.14946, 0.945937],
[0.331222, 0.136183],
[0.203402, 0.778795],
[0.266571, 0.594372],
[0.167538, 0.0446446],
[0.140836, 0.721618],
[0.128978, 0.501703],
[0.026001, 0.918282],
[0.309361, 0.812784],
[0.894343, 0.50749],
[0.955845, 0.0849136],
[0.0294987, 0.0497343],
[0.449537, 0.0581108],
[0.117579, 0.944927],
[0.993694, 0.433286],
[0.160816, 0.695717],
[0.460641, 0.861158],
[0.524046, 0.513972],
[0.356882, 0.885937],
[0.84475, 0.928911],
[0.463369, 0.456124],
[0.942843, 0.065224],
[0.114482, 0.38391],
[0.645504, 0.085737]])
    we19973 = np.array([[0.945512, 0.358464],
[0.533985, 0.473291],
[0.321248, 0.478848],
[0.163416, 0.669122],
[0.590305, 0.508565],
[0.378116, 0.853256],
[0.867307, 0.0231629],
[0.859917, 0.968826],
[0.0389623, 0.948691],
[0.702187, 0.523241],
[0.149432, 0.237551],
[0.661452, 0.664152],
[0.322827, 0.951449],
[0.14165, 0.208475],
[0.735349, 0.621844],
[0.384318, 0.770746],
[0.621751, 0.898941],
[0.253551, 0.81789],
[0.763675, 0.743253],
[0.541194, 0.982668],
[0.474096, 0.736901],
[0.293325, 0.850194],
[0.30298, 0.296974],
[0.0506514, 0.284104],
[0.464858, 0.719493],
[0.380786, 0.378163],
[0.623601, 0.338959],
[0.551105, 0.165703],
[0.664077, 0.747416],
[0.676729, 0.0556162],
[0.331031, 0.359817],
[0.226029, 0.582158],
[0.0557666, 0.550544],
[0.0903381, 0.928902],
[0.435583, 0.637465],
[0.57328, 0.941561],
[0.882573, 0.525375],
[0.686879, 0.391321],
[0.280787, 0.338484],
[0.854805, 0.667309],
[0.808386, 0.0537881],
[0.329043, 0.250329],
[0.966797, 0.204677],
[0.760171, 0.780492],
[0.185338, 0.81797],
[0.91959, 0.336981],
[0.132011, 0.0500499],
[0.671405, 0.256288],
[0.110876, 0.51913],
[0.462686, 0.638955]])
    we = [we44497, we512, we1024, we19973]
    pos = bounds[0] + (bounds[1] - bounds[0])*we[num_we]
    vel = (bounds[0] + (bounds[1] - bounds[0])*np.random.rand(num_particles, num_dimensions))*scale_velocity
    return pos, vel

# compute one particle
class SHMPSO:
    def __init__(self, num_dimensions, num_particles, max_iteration, num_we):
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

        self.position, self.velocity = SHM_initial(num_dimensions, num_particles, num_we)  # particle initial
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

def call_WESHMPSO(iteration=1000, num_particles=50, num_we=0):
   
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
        fitness_PSO = SHMPSO(2, num_particles, iteration, num_we)    # global best & series fitness
        gbest = fitness_PSO(all_func[n,1])
        name_array[n, :] = all_func[n,0]
        ls_fit_array[n, :] = fitness_PSO.ls_gbest_iter
        gbest_array[n, 0] = gbest
        #axis[n, 0].plot(ls_iteration, fitness_PSO[1])
        #axis[n, 0].set_title(all_func[n][0])
    return name_array, ls_fit_array, gbest_array
    #plt.show()
if __name__ == "__main__":
    info = call_WESHMPSO()
    print(info[2])