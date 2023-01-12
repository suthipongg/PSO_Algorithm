import numpy as np
from math import pi, exp, sqrt, cos, sin, e

class func_2D:
    def __init__(self):
        self.bound_Ackley = [(-5,5), (-5,5)]
        self.bound_Levi_N13 = [(-10,10), (-10,10)]
        self.bound_Eggholder = [(-512,512), (-512,512)]
        self.bound_Holder_table = [(-10,10), (-10,10)]
        self.bound_Rastrigin = [(-5.12,5.12), (-5.12,5.12)]
        self.bound_Schwefel = [(-500,500), (-500,500)] 
        self.n_func = self.all_func().shape[0]

    def Ackley(self, x):
        return -20.0 * exp(-0.2*sqrt(0.5*(x[0]**2 + x[1]**2))) - exp(0.5*(cos(2*pi*x[0]) + cos(2*pi*x[1]))) + e + 20

    def Levi_N13(self, x):
        return (sin(3*pi*x[0]))**2 + ((x[0] - 1)**2)*(1 + (sin(3*pi*x[1]))**2) + ((x[1] - 1)**2)*(1 + (sin(2*pi*x[1]))**2)

    def Eggholder(self, x):
        return -(x[0] + 47)*(sin(sqrt(abs((x[1]/2) + (x[0] + 47))))) - (x[1])*(sin(sqrt(abs(x[1] - (x[0] + 47))))) + 959.6407

    def Holder_table(self, x):
        return -(abs((sin(x[0])*(cos(x[1]))*(exp(abs(1 - ((sqrt(x[0]**2 + x[1]**2))/pi))))))) + 19.2085

    def Rastrigin(self, x):
        return (x[0]**2 - 10*cos(2*pi*x[0]) + 10) + (x[1]**2 - 10*cos(2*pi*x[1]) + 10)

    def Schwefel(self, x):
        return 418.9829*(2) - ((x[0]*sin(sqrt(abs(x[0])))) + (x[1]*sin(sqrt(abs(x[1])))))

    def all_func(self):
        return np.array([
            ["Ackley", self.Ackley, self.bound_Ackley],
            ["Levi_N13", self.Levi_N13, self.bound_Levi_N13],
            ["Eggholder", self.Eggholder, self.bound_Eggholder],
            ["Holder_table", self.Holder_table, self.bound_Holder_table],
            ["Rastrigin", self.Rastrigin, self.bound_Rastrigin],
            ["Schwefel", self.Schwefel, self.bound_Schwefel]
            ])

if __name__ == "__main__":
    func = func_2D()
    print(func.n_func)