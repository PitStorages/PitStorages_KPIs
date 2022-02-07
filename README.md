# PitStorages_KPIs

In the folder **notebooks** there are scripts for calculating stratification indicators for heat storages. Each script can be used for calculating a particular indicator. A short description of each spript is provided below.

##### [Temperature_profile.py](https://github.com/PitStorages/PitStorages_KPIs/blob/main/notebooks/temperature_profile.py)
It contains a function that creates temperature profiles for a 1-D tank. This function is called in the rest of the scripts for creating a temperature profile on which each indicator is calculated.

##### [Entropy_efficiency.ipynb](https://github.com/PitStorages/PitStorages_KPIs/blob/main/notebooks/Entropy_efficiency.ipynb)
The entropy efficiency of the storage is calculated using the method described in the publication by Haller M. Y. et al. (2010), "A method to determine stratification efficiency of thermal energy storage processes independently from storage heat losses".

##### [Exergy_destruction.ipynb](https://github.com/PitStorages/PitStorages_KPIs/blob/main/notebooks/Exergy_destuction.ipynb)
Exergy destruction is used as an indicator for assessing stratification in a storage. It is calculated through the exergy balance for the storage:
![alt text](https://github.com/PitStorages/PitStorages_KPIs/blob/main/images/Exergy_destruction_equation.PNG)

##### [Exergy_efficiency.ipynb](https://github.com/PitStorages/PitStorages_KPIs/blob/main/notebooks/Exergy_efficiency.ipynb)
The exergy efficiency of the storage is calculated using the method described in the publication by Haller M. Y. et al. (2010) "A method to determine stratification efficiency of thermal energy storage processes independently from storage heat losses".

##### [MIX_number.ipynb](https://github.com/PitStorages/PitStorages_KPIs/blob/main/notebooks/MIX_number.ipynb)
The stratification of a storage is assessed using the MIX number as described in the publication by Andersen E. et al.(2007), "Mutlilayer fabric stratification pipes for solar tanks".

##### [Stratification_coefficienct.ipynb](https://github.com/PitStorages/PitStorages_KPIs/blob/main/notebooks/Stratification_coefficient.ipynb)
The stratification of a storage is assessed using the Stratification coefficient as described in the publication by Wu L. and Bannerot R. B. (1987) "An experimental study of the effect of water extraction on thermal stratification in storage".
