
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'full') / w

import bottleneck as bn
def rollavg_bottlneck(a,n):
    return bn.move_mean(a, window=n,min_count = None)


# %% Notes
# Alaternatively have a number of nodes in the top and bottom to fully mix to
# to simulate the imperfect diffusers. Then have smaller amount of mnodes to
# mix in the storage, e.g., simulate real water conduction

# Ended up using rolling mean to simulate mixing

# To ensure no energy losses, mixing nodes has to be divisble by N?

# To do
# Calculate energy in or out?

# %%

# General
N = 5000
charge_time = 7*24  # Approximate
discharge_time = 7*24  # Approximate

# Time periods
simulation_period = 700#8760  # number of hours
time_step = 1  # time step in hours
no_time_steps = np.int(simulation_period / time_step)
times = np.arange(no_time_steps)*time_step

mixing_nodes = 101  # uneven number! - determines mixing - perfect_stratification=1, perfect mixing=N
mixing_half_nodes = np.int(((mixing_nodes-1)/2))
charge_nodes = np.max([np.int(np.round(N/(charge_time/time_step))), 1])
discharge_nodes = np.max([np.int(np.round(N/(discharge_time/time_step))), 1])

# Temperatures
T_hot = 90
T_cold = 45
T_threshold = 10
# Initialize the storage as empty
T_start = T_cold
temp = pd.Series(np.ones(N)*T_start)
temp_mix = temp

# %%
case = ''

charging = 1  # Initialize storage as charging
for ii, t in enumerate(times):
    # Check if charging or discharging
    if (temp.iloc[-1] >= T_hot - T_threshold) & (charging == 1):
        charging = 0  # discharging
    elif (temp.iloc[0] <= T_cold + T_threshold) & (charging == 0):
        charging = 1  # charging

    # Charge or discharge, i.e., add hot water to the top or cold water to the bottom
    shift = charge_nodes*charging - discharge_nodes*(1-charging)
    temp = temp.shift(shift).fillna(T_hot*charging + T_cold*(1-charging))

    # Mixing - use pandas rolling (min_periods=1 avoids nan at ends)
    # Results in very small difference in avg. temperature
    if case == 'fully_mixed':
        temp_mix[:] = np.mean(temp)
    elif case == 'fully_stratified':
        temp_mix = temp
    else:
        temp_mix = temp.rolling(mixing_nodes, win_type='boxcar', center=True, min_periods=mixing_nodes).mean()
        #temp_mix[temp_mix.isna()] = temp
        if charging:
            temp_mix[:mixing_half_nodes] = np.mean(temp[:mixing_half_nodes])
            temp_mix[-mixing_half_nodes:] = temp[-mixing_half_nodes:]
        else:
            temp_mix[-mixing_half_nodes:] = np.mean(temp[-mixing_half_nodes:])
            temp_mix[:mixing_half_nodes] = temp[:mixing_half_nodes]

    temp = temp_mix
    # Heat losses
    # Plot
    if np.mod(ii, 10) == 0:
        plt.plot(temp[::-1], np.linspace(0,1,len(temp)))
        plt.xlim(40,100)
        plt.ylim(0,1)
        plt.show()

# %%





# %% Garbage
# Previous charging/discharging numpy style
    # if charging:
    #     temp = np.hstack([np.ones(charge_nodes)*T_hot, temp[:-charge_nodes]])
    # else:
    #     temp = np.hstack([temp[discharge_nodes:], np.ones(discharge_nodes)*T_cold])


# Previous attempts at mixing
#mixing_steps = np.int(np.floor(N/mixing_nodes))
#mixing_half_step = np.int((mixing_steps-1)/2)
    #for m in range(0, mixing_steps*mixing_nodes, mixing_nodes):
    #    temp[m:m+mixing_nodes] = np.mean(temp[m:m+mixing_nodes])
    #temp_copy = np.copy(temp2)
    #temp = np.copy(temp2)
    #for m in range(mixing_half_step, mixing_steps*mixing_nodes-mixing_half_step, mixing_half_step):
        #temp[m:m+mixing_nodes] = np.mean(temp[m:m+mixing_nodes])
    #    temp_copy[m-mixing_half_step:m+mixing_half_step] = np.mean(temp_copy[m-mixing_half_step:m+mixing_half_step])
