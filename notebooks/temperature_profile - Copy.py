import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def temp_profile(simulated_case, N, mix_nodes, T_hot, T_cold, T_threshold, simulation_period,
                 charge_time, discharge_time, plot=False):
    '''
    Creates temperature profiles for a 1-D tank.
    Initializes the storage as empty (i.e., uniform temperature equal to T_cold).
    '''
    # General
    #N = Number of nodes in the storage

    # Time periods
    time_step = 1  # time step in hours
    no_time_steps = int(simulation_period / time_step) # number of timesteps
    times = np.arange(no_time_steps)*time_step  # array of time steps in hours

    # Nodes used for mixing, charge and discharge
    mixing_nodes = mix_nodes  # uneven number! - gives the number of nodes that will be mixed (average temperature) at each timestep
    # finds the half of the mixing nodes. These nodes are set to NaN by the rolling window,
    # so is used for replacing the NaNs with actual temperature values
    mixing_half_nodes = int(((mixing_nodes-1)/2))
    charge_nodes = np.max([int(np.round(N/(charge_time/time_step))), 1]) # number of nodes charged each time step (avoid numerical diffusion)
    discharge_nodes = np.max([int(np.round(N/(discharge_time/time_step))), 1]) # number of nodes discharged each time step (to avoid numerical diffusion)

    # Initialize the storage as empty
    T_start = T_cold
    temp = pd.Series(np.ones(N)*T_start)
    temp_mix = temp
    
    case = simulated_case
    dfl = []

    charging = 1  # Initialize storage as charging
    
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 100), ylim=(5100, 0))
    line, = ax.plot([], [], lw=2)
    
    for ii, t in enumerate(times):
        # Check if charging or discharging
        if (temp.iloc[-1] >= T_hot - T_threshold) & (charging == 1):
            charging = 0  # discharging
        elif (temp.iloc[0] <= T_cold + T_threshold) & (charging == 0):
            charging = 1  # charging

        # Charge or discharge, i.e., add hot water to the top or cold water to the bottom
        shift = charge_nodes*charging - discharge_nodes*(1-charging)
        temp = temp.shift(shift).fillna(T_hot*charging + T_cold*(1-charging))

        # Mixing - use pandas rolling (min_periods=1 avoids NaN at ends)
        # Results in very small difference in avg. temperature
        if case == 'fully_mixed':
            temp_mix[:] = np.mean(temp)
        elif case == 'fully_stratified':
            temp_mix = temp
        else:
            temp_mix = temp.rolling(mixing_nodes, win_type='boxcar', center=True, min_periods=mixing_nodes).mean()
            if charging:
                temp_mix[:mixing_half_nodes] = np.mean(temp[:mixing_half_nodes])
                temp_mix[-mixing_half_nodes:] = temp[-mixing_half_nodes:]
            else:
                temp_mix[-mixing_half_nodes:] = np.mean(temp[-mixing_half_nodes:])
                temp_mix[:mixing_half_nodes] = temp[:mixing_half_nodes]

        temp = temp_mix
        dfl.append(temp)

                
    # Create dataframe containing the temperature profiles for each time step.
    # Rows are the simulation time steps and columns are the layers of the storage starting from the top (0) to the bottom (N).
    df = pd.concat(dfl, axis=1).T
    
    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = np.linspace(0,5001,len(temp))
        y = df.iloc[-i]
        line.set_data(y, x)
        return line,
        
    # Plot
    if plot:
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames = 700, interval=60)#, blit=True)
        anim.save('C:/Users/iosif/Desktop/animation.gif')
        plt.show()
    
    return df

#%%

df = temp_profile('', 5000, 501, 90, 45, 10, 700, 7*24, 7*24, plot=True)








