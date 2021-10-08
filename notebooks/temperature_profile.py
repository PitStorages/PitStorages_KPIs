import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
# simulated_case  = ''
# N = 10
# T_hot = 90
# T_cold = 45
# T_threshold = 10
# simulation_period = 1700
# heat_loss_coeff = 1850
# th_cond_coeff = 2.5
# mix_nodes = int(N/4)
# max_storage_cycles = 2
# plot = True



#%%

def temp_profile(simulated_case, N, mix_nodes, T_hot, T_cold, T_threshold,
                 th_cond_coeff, simulation_period, heat_loss_coeff, T_amb=10,
                 max_storage_cycles=None, plot=False):
    '''
    Creates temperature profiles for a 1-D tank.
    Initializes the storage as empty (i.e., uniform temperature equal to T_cold).
    
    
    Parameters
    ----------
    N: int
        Number of vertical tank nodes.
    heat_loss_coeff: int
        Enable heat losses for the storage.
    T_amb: int, default: 10
        Ambient temperature for the calculation of heat loss.

    Returns
    -------
    df: dataframe
        DataFrame with the temperature profiles for each time-step.
    df_ancil: dataframe
        Dataframe with ancillary calculations, e.g. entropy.
    '''
    # volume flow in the storage (1 node per timestep)
    flow = 1/N

    # Temperature change due to vertical thermal conduction. A cross section area of 1 m2 is assumed.
    node_height = 1/N
        
    # Number of time periods
    no_time_steps = int(simulation_period / flow)
    
    # Series of time steps
    times = np.arange(no_time_steps) * flow

    # Initialize the storage as empty
    T_start = T_cold
    temp = pd.Series(np.ones(N)*T_start)
    temp_mix = temp
    
    # Creation of dataframe containing time-series variables
    df_ancil = pd.DataFrame()
    
    # Create empty lists for the calculations
    dfl = []

    charging = 1  # Initialize storage as charging
    storage_cycles = 1 # Set the storage to do at least one cycle
    
    # Iterate over time-steps
    for ii, t in enumerate(times):
        # Check if charging or discharging
        if (temp.iloc[-1] >= T_hot - T_threshold) & (charging == 1):
            charging = 0  # discharging
        elif (temp.iloc[0] <= T_cold + T_threshold) & (charging == 0):
            charging = 1  # charging
            storage_cycles += 1
            if storage_cycles > max_storage_cycles:
                break
        
        # Number of nodes shifted by charging/discharging
        shift = charging - (1-charging)
        
        energy_rate = (charging*(T_hot - temp[-mix_nodes:].mean())
                       + (1-charging)*(T_cold - temp[:mix_nodes].mean()))*980*4200*flow

        temp = temp.shift(shift).fillna(T_hot*charging + T_cold*(1-charging))

        if simulated_case == 'fully_mixed':
            temp_mix = pd.Series(np.ones(N) * temp.sum()/N)
        elif simulated_case == 'fully_stratified':
            temp_mix = temp
        else:
            if charging == 1:
                temp_mix = temp.copy()
                temp_mix.iloc[:mix_nodes] = temp.iloc[:mix_nodes].sum()/mix_nodes
            else:
                temp_mix = temp.copy()
                temp_mix.iloc[-mix_nodes:] = temp.iloc[-mix_nodes:].sum()/mix_nodes

            # HEAT CONDUCTION BETWEEN LAYERS - ONLY IN REAL CASES
            # Calculation of the upward and downward energy transfer due to thermal conductivity for each node [W]
            Q_cond_down = []
            Q_cond_up = []
            for i in range(N-1):
                Q_down = th_cond_coeff/node_height * (temp_mix[i+1] - temp_mix[i])
                Q_cond_down.append(Q_down)
            # The downward conductivity for the last node is zero
            Q_cond_down.append(0)
            # The upward conductivity for the first node is zero
            Q_cond_up.append(0)
            for i in range(1,N):
                Q_up = th_cond_coeff/node_height * (temp_mix[i-1] - temp_mix[i])
                Q_cond_up.append(Q_up)
            # Total heat conduction (gain or loss) for each node based on the temperatures of the adjacent nodes [J]
            Q_cond = (pd.Series(Q_cond_up) + pd.Series(Q_cond_down)) * 3600 / N
            # Temperature lost or gain due to vertical conduction per layer
            T_cond = Q_cond / (4200 * 980 * (1/N))
            temp_mix = temp_mix.add(T_cond)

        # Internal entropy generation (caused by mixing) formula: S = Q/T
        df_ancil.loc[ii, 'internal_entropy_generation'] = (temp_mix-temp).multiply(1/N).multiply(980*4200).divide(temp+273.15).sum()

        # Heat loss per layer [J]. Tank area 6 m2 and timestep duration in seconds is 3600/N
        Q_heat_loss_layer = heat_loss_coeff * 6 * (1/N) * (temp_mix - T_amb) * 3600/N
        
        # Calculation of the temperatrue lost due to heat loss (if heat loss is not enabled it is zero)
        temp_loss = Q_heat_loss_layer / (4200 * 980 * (1/N))
        
        # Storage temperature subtracting the temperature lost due to heat loss
        temp = temp_mix.subtract(temp_loss)
            
        # Entropy loss due to heat losses
        df_ancil.loc[ii, 'heat_loss_entropy'] = (-temp_loss).multiply(1/N).multiply(980*4200).divide(temp_mix+273.15).sum()

        # Accumulated entropy in the storage
        df_ancil.loc[ii, 'storage_entropy'] = (temp-T_cold).multiply(1/N).multiply(980*4200).divide(temp+273.15).sum()
    
        # Append the temperatrues and charge status to their corresponding list
        dfl.append(temp)
        df_ancil.loc[ii, 'heat_loss'] = Q_heat_loss_layer.sum()
        df_ancil.loc[ii, 'energy_rate'] = energy_rate
        df_ancil.loc[ii, 'energy_content'] = (temp.sum()/N - T_cold)*4200*980
        df_ancil.loc[ii, 'charge_status'] = charging

        # Plot
        if plot:
            if np.mod(ii, 5) == 0:
                plt.plot(temp[::-1], np.linspace(0,1,len(temp)))
                plt.xlim(40,100)
                plt.ylim(0,1)
                plt.show()
    
    # Create dataframe containing the temperature profiles for each time step.
    df_temp = pd.concat(dfl, axis='columns').T
    return df_temp, df_ancil

#%%


