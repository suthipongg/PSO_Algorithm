from cProfile import label
from SHMPSO.SHMPSO_source_code import call_SHMPSO
from SHMPSO.WESHMPSO_source_code import call_WESHMPSO
from WEPSO.WEPSO import call_WEPSO
from PSO.PSO import call_PSO
from Test_Function_numpy import func_2D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

func = func_2D()
ls_func = func.all_func()
name_func = ls_func[:, 0]
iter = 1000
iter_array = np.arange(iter, dtype=int)

shmpso = call_SHMPSO(iter)
ls_fit_shmpso = shmpso[1][:, :-1]
gbest_shmpso = shmpso[2]

we512shmpso = call_WESHMPSO(iter, num_we=1)
ls_fit_we512shmpso = we512shmpso[1][:, :-1]
gbest_we512shmpso = we512shmpso[2]

we512pso = call_WEPSO(iter)
ls_fit_we512pso = we512pso[1][:, :-1]
gbest_we512pso = we512pso[2]

pso = call_PSO(iter)
ls_fit_pso = pso[1][:, :-1]
gbest_pso = pso[2]

d = {'Function' : np.reshape(name_func, (func.n_func, )), 
    'PSO' : np.reshape(gbest_pso, (func.n_func, )), 
    'WE512PSO' : np.reshape(gbest_we512pso, (func.n_func, )), 
    'SHMPSO' : np.reshape(gbest_shmpso, (func.n_func, )),
    'WE512SHMPSO' : np.reshape(gbest_we512shmpso, (func.n_func, ))}
df = pd.DataFrame(data=d)
#df.to_csv("pso_wepso_shmpso_4.csv")
print(df)

figure, axis = plt.subplots(nrows=2, ncols=3, figsize=(15,30))
for i in range(func.n_func):
    axis[i%2, i//2].plot(iter_array, ls_fit_shmpso[i], label="SHMPSO")
    axis[i%2, i//2].plot(iter_array, ls_fit_we512pso[i], label="WE512PSO")
    axis[i%2, i//2].plot(iter_array, ls_fit_pso[i], label="PSO")
    axis[i%2, i//2].plot(iter_array, ls_fit_we512shmpso[i], label="WE512SHMPSO")
    axis[i%2, i//2].set_title(name_func[i])
    axis[i%2, i//2].legend()
plt.show()


