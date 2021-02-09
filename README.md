# files
1. data (this directory contains the configuration files needed to run the sumo simulation)
2. cross_simulation.py (code to launch the sumo simulation, also contains the training of the q-table and the execution of the recommendation algorithm)
3. q_table.txt (File where the q-value table generated by the algorithm execution is stored to have persistence in it).
 
# How to run:
Python code to run the sumo simulation with the recommendation of traffic light times based on the number of vehicles present on the road. 

1. Configure the database to be used.
2. Adjust the configuration variables of cross_simulation.py as required.
3. Run cross_simulation.py

# Notes:
It is necessary to have the sumo tool installed, which can be obtained from the following link https://sumo.dlr.de/docs/Installing.html.

The files contained in this git should be saved in the sumo-doc/tutorial/traci_tls folder.

It is necessary to declare the variable SUMO_HOME as follows:
    SUMO_HOME=/usr/share/sumo
    export SUMO_HOME=/usr/share/sumo
