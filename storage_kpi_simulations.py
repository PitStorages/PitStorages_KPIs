
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %% Notes
# Alaternatively have a number of nodes in the top and bottom to fully mix to
# to simulate the imperfect diffusers. Then have smaller amount of mnodes to
# mix in the storage, e.g., simulate real water conduction
# Another note

# Ended up using rolling mean to simulate mixing

# To ensure no energy losses, mixing nodes has to be divisble by N?

# To do
# Calculate energy in or out?
# Heat losses


# Mix number is garbage because you need to set arbitrary T_ref, which actually
# has a big impact. For storages with solar thermal and hp (aka. two cold levels)
# this cannot work.
# Second problem with MIX number is that it is affected by the energy content...
# at high or low energy content, has problems.. does not perform well when close to being full or empty
# Mix number is instantenous - i.e., does not give info of period

# Hypothesis - as water get's cleaned out - stratification should increase?

#%%
def density_water(T):
    '''Calculates density (rho) of water in kg/m^3 based on fluid temperature (T) nearest the flow meter in degrees Celsius'''
    rho = (999.85+5.332*(10**-2)*T-7.564*(10**-3)*(T**2)+4.323*(10**-5)*(T**3)-1.673*(10**-7)*(T**4)+2.447*(10**-10)*(T**5))
    return(rho)

def specific_heat_water(T):
    '''Calculates specific heat (cp) of water in J/(kg K) based on mean fluid temperature (T) in degrees Celsius'''
    cp = (4.2184-2.8218*(10**-3)*T+7.3478*(10**-5)*(T**2)-9.4712*(10**-7)*(T**3)+7.2869*(10**-9)*(T**4)-2.8098*(10**-11)*(T**5)
          +4.4008*(10**-14)*(T**6))*1000
    return(cp)


# %%

# General
N = 5000 # Number of nodes in the storage
charge_time = 7*24  # Approximate
discharge_time = 7*24  # Approximate

# Time periods
simulation_period = 700 #8760  # number of hours
time_step = 1  # time step in hours
no_time_steps = int(simulation_period / time_step) # number of timesteps
times = np.arange(no_time_steps)*time_step  # array of time steps in hours

# Nodes used for mixing, charge and discharge
mixing_nodes = 501  # uneven number! - gives the number of nodes that will be mixed (average temperature) at each timestep
# finds the half of the mixing nodes. These nodes are set to NaN by the rolling window,
# so is used for replacing the NaNs with actual temperature values
mixing_half_nodes = int(((mixing_nodes-1)/2))
charge_nodes = np.max([int(np.round(N/(charge_time/time_step))), 1]) # number of nodes charged each time step (avoid numerical diffusion)
discharge_nodes = np.max([int(np.round(N/(discharge_time/time_step))), 1]) # number of nodes discharged each time step (to avoid numerical diffusion)

# Temperatures
T_hot = 90 # charge temperature
T_cold = 45 # discharge temperature
T_threshold = 10 # sets a threshold for the maximum/minimum temperature that can be during charge/discharge at the top and bottom of the tank

# Initialize the storage as empty
T_start = T_cold
temp = pd.Series(np.ones(N)*T_start)
temp_mix = temp

# %%
# Set the case that will be simulated, e.g. "fully_mixed", "fully_stratified".
# Otherwise, it simulates storage operation with mixing based on mixing_nodes.
case = ''#'fully_stratified'
plot = False
dfl = []

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

    # Plot
    if plot:
        if np.mod(ii, 25) == 0:
            plt.plot(temp[::-1], np.linspace(0,1,len(temp)))
            plt.xlim(40,100)
            plt.ylim(0,1)
            plt.show()


# %% Concat dataframe
# Create dataframe containing the temperature profiles for each time step.
# Rows are the simulation time steps and columns are the layers of the storage starting from the top (0) to the bottom (N).
df = pd.concat(dfl, axis=1).T

#%%

###########################################################################################################################################
################################################################# MIX NUMBER ##############################################################
###########################################################################################################################################

# The MIx number is a dimmensionless KPI that indicates how does the storage perform compared to a fully stratified and a fully mixed storage.

# Calculation of storage energy content
volume_per_layer = 1/N # Assuming that the storage is a cube 1x1x1 m
storage_volume = 1
storage_height = 1

T_ref = 45 # reference temperature for calculating energy

# Energy content of storage
Q_storage = (980 * volume_per_layer * 4200 * (df - T_ref)).sum(axis='columns')


#%%
# Calculation of MIX number for a fully mixed storage
T_avg = df.mean(axis='columns')
# Calculation of the volume of a mixed storage using the energy content of the actual storage
V_mix = Q_storage/(980*4200*(T_avg-T_ref))
# Distance from the bottom of the storage to the middle of the V_mix
dist_mix = storage_height/2
# MIX number for fully mixed storage
M_mix = ((T_avg-T_ref)*980*4200*V_mix)*dist_mix


#%%
# Find the volume of water having 90 degC in order to have the same energy content
# This assumes there is no cold part - i.e., cold part is equal to T_ref
V_90 = Q_storage/(980*4200 * (T_hot - T_ref))
# Distance from the bottom of the storage for a storage that has 90 degrees at the top
dist_90 = storage_height - V_90/storage_volume/2


# Calculate MIX number for stratified tank having 90 degC at the top
M_strat = (V_90*(T_hot-T_ref)*dist_90)*980*4200

#%%
# Make a list with the distances of each layer from the bottom of the storage
dist = np.arange(1/N/2, 1, 1/N)[::-1]
# Calculate the mix number for the actual storage
M_exp = 980 * volume_per_layer* dist * 4200 * (df  - T_ref)
MIX = (M_strat - M_exp.sum(axis=1)) / (M_strat - M_mix)

#%%
fig, axes = plt.subplots(nrows=2, sharex=True)
MIX.plot(ax=axes[0], ylim=[0,1.1])
Q_storage.plot(ax=axes[1])

axes[-1].set_xlim(0,430)

for ax in axes:
    ax.axvline(205)
    ax.axvline(405)

# %%
fig, ax = plt.subplots()
M_strat.plot(ax=ax, label='Stratified')
M_exp.sum(axis=1).plot(ax=ax, label='Actual')
M_mix.plot(ax=ax, label='Mixed')

ax.legend()

#%%
plt.figure()
V_90.plot(ylim=[-0.01,1.01])

# %% Garbage
# Previous charging/discharging numpy style
    # if charging:
    #     temp = np.hstack([np.ones(charge_nodes)*T_hot, temp[:-charge_nodes]])
    # else:
    #     temp = np.hstack([temp[discharge_nodes:], np.ones(discharge_nodes)*T_cold])


# Previous attempts at mixing
#mixing_steps = int(np.floor(N/mixing_nodes))
#mixing_half_step = int((mixing_steps-1)/2)
    #for m in range(0, mixing_steps*mixing_nodes, mixing_nodes):
    #    temp[m:m+mixing_nodes] = np.mean(temp[m:m+mixing_nodes])
    #temp_copy = np.copy(temp2)
    #temp = np.copy(temp2)
    #for m in range(mixing_half_step, mixing_steps*mixing_nodes-mixing_half_step, mixing_half_step):
        #temp[m:m+mixing_nodes] = np.mean(temp[m:m+mixing_nodes])
    #    temp_copy[m-mixing_half_step:m+mixing_half_step] = np.mean(temp_copy[m-mixing_half_step:m+mixing_half_step])
