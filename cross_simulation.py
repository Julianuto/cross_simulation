#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from datetime import date
from datetime import datetime

import os
import sys
import optparse
import random
import numpy       
import json
import psycopg2
import MySQLdb
from functools import reduce

# Setup Variables
simulation_type = 2;   # 0 fix times, 1 basic recommendation of times, 2 RL recommendation of times
route_generation = 1;  # 0 no route generation, 1 route generation 
q_table_training = 1; # 0 no matrix Q training, 1 matrix q training

# Mysql Database Connection Constants 

hostname = 'localhost'
username = 'root'
password = ''
database = 'traffic2020'

# Connection to DB

myConnection = MySQLdb.connect( host=hostname, user=username, passwd=password, db=database )
cursor = myConnection.cursor()

### TRAINING CODE
def dec_a_base3(decimal): # decimal to base 3 number function
    num_base3 = ''
    while decimal // 3 != 0:
        num_base3 = str(decimal % 3) + num_base3
        decimal = decimal // 3
    return str(decimal) + num_base3

def dec_a_base2(decimal): # decimal to base 2 number function
    num_base2 = ''
    while decimal // 2 != 0:
        num_base2 = str(decimal % 2) + num_base2
        decimal = decimal // 2
    return str(decimal) + num_base2

state_m = numpy.array([[0,0,0,0,0,0,0,0]]) 
# Initial states array is 8 bits, because the cross has 4 directions West-East, East-West, North-South, South-North, 
# As it has 4 directions, it has 4 traffic lights and 4 vehicle queues
# 4 MSB (Most Significant Bits) of 8 bits are Times of Green in each traffic light (0 Short Time, 1 Long Time)
# 4 LSB (Least Significant Bits) of 8 bits are vehicle queues in each direction (0 Short queue, 1 Long queue)
  
# The next code adds the 256 values ??(2 to the power 8) that the states array can have
for i in range(1, 256):
   state_l= [int(x) for x in list('{0:0b}'.format(i))]
   state_a= numpy.array(state_l)
   if state_a.size < 8:
     for i in range(state_a.size,8):
       state_a = numpy.append([0], state_a)
   state_m = numpy.append(state_m, [state_a], axis=0)

action_m = numpy.array([[0,0,0,0]])
# Initial action array is 4 digits base 3 number (0, 1 or 2), one action for each direction 
# Each digit in action has value 0,1, or 2. 
# The value 0 in action is when the same time is kept in green, 
# the value 1 in action is when the time in green is decreased,
# the value of 2 in the action is when the time in green is increased

# The next code adds the 81 values ??(3 to the power 4) that the action array can have
for i in range(1, 81):
   action_l = dec_a_base3(i)
   action_l = [int(x) for x in str(action_l)]
   action_a= numpy.array(action_l)
   if action_a.size < 4:
     for i in range(action_a.size,4):
       action_a = numpy.append([0], action_a)
   action_m = numpy.append(action_m, [action_a], axis=0)

# Initial values (array of zeros) of next_state_matrix and reward_matrix
next_state_matrixP = numpy.zeros((256, 81))
reward_matrixP = numpy.zeros((256, 81))

# The next code, 81 cycles per each of 256 states, calculates the posible values of reward_matrix and next state matrix.

for i in range(0, 256):
  for j in range(0, 81):
     reward= numpy.zeros(4)
     next_state_response = numpy.zeros(8)
          
     for k in range(0, 4):
       if ((state_m[i,k] == 0) and (state_m[i,k+4] == 0) and (action_m[j,k] == 0)):
           # If the Green Time is short, the car queue is short and the action is to keep the Green Time is something logical
           reward[k] = 2      
           next_state_response[k] = 0
           next_state_response[k+4] = 0
       if (state_m[i,k] == 0) and (state_m[i,k+4] == 0) and (action_m[j,k] == 1):
           # If the Green Time is short, the car queue is short and the action is to decrease the Green Time is something illogical, it cannot be decreased any more
           reward[k] = -1    
           next_state_response[k] = 0
           next_state_response[k+4] = 0
       if (state_m[i,k] == 0) and (state_m[i,k+4] == 0) and (action_m[j,k] == 2):
           # If the Green Time is short, the queue for cars is short and the action is to increase the Green Time is something illogical, it would not be required to increase
           reward[k] = -1    
           next_state_response[k] = 1
           next_state_response[k+4] = 0
       
       if (state_m[i,k] == 0) and (state_m[i,k+4] == 1) and (action_m[j,k] == 0):
           # If the Green Time is short, the queue from cars is long and the action is to keep the Green Time is not very useful.
           reward[k] = -2    
           next_state_response[k] = 0
           next_state_response[k+4] = 1
       if (state_m[i,k] == 0) and (state_m[i,k+4] == 1) and (action_m[j,k] == 1):
           # If the Time in Green is short, the queue from cars is long and the action is to decrease the Time in Green is something illogical, it cannot be decreased further
           reward[k] = -3    
           next_state_response[k] = 0
           next_state_response[k+4] = 1
       if (state_m[i,k] == 0) and (state_m[i,k+4] == 1) and (action_m[j,k] == 2):
           # If the Green Time is short, the queue from cars is long and the action is to increase the Green Time is something logical and useful.
           reward[k] = 5    
           next_state_response[k] = 1
           next_state_response[k+4] = 1

       if (state_m[i,k] == 1) and (state_m[i,k+4] == 0) and (action_m[j,k] == 0):
           # If the Green Time is long, the queue for cars is short and the action is to keep the Green Time could be useless
           reward[k] = -2    
           next_state_response[k] = 1
           next_state_response[k+4] = 0
       if (state_m[i,k] == 1) and (state_m[i,k+4] == 0) and (action_m[j,k] == 1):
           # If the Green Time is long, the queue for cars is short and the action is to decrease the Green Time is something logical and can be useful.
           reward[k] = 3    
           next_state_response[k] = 0
           next_state_response[k+4] = 0
       if (state_m[i,k] == 1) and (state_m[i,k+4] == 0) and (action_m[j,k] == 2):
           # If the Green Time is long, the queue for cars is short and the action is to increase the Green Time is something illogical.
           reward[k] = -3    
           next_state_response[k] = 1
           next_state_response[k+4] = 0
           
       if (state_m[i,k] == 1) and (state_m[i,k+4] == 1) and (action_m[j,k] == 0):
           # If the Green Time is long, the car queue is long and the action is to keep the Green Time is very logical
           reward[k] = 5    
           next_state_response[k] = 1
           next_state_response[k+4] = 1
       if (state_m[i,k] == 1) and (state_m[i,k+4] == 1) and (action_m[j,k] == 1):
           # If the Time in Green is long, the queue for cars is long and the action is to decrease the Time in Green is something illogical.
           reward[k] = -3    
           next_state_response[k] = 0
           next_state_response[k+4] = 1
       if (state_m[i,k] == 1) and (state_m[i,k+4] == 1) and (action_m[j,k] == 2):
           # If the Time in Green is long, the queue for cars is long and the action is to increase the Time in Green, it is something illogical, it cannot be increased more.
           reward[k] = -3    
           next_state_response[k] = 1
           next_state_response[k+4] = 1
     
     final_reward = reward[0]+reward[1]+reward[2]+reward[3]
     reward_matrixP[i,j] = final_reward
     # The following is to go from a binary array to an integer value
     next_state_value = reduce(lambda a,b: 2*a+b, next_state_response)
     next_state_matrixP[i,j] =next_state_value

# So far the matrix P (reward) has been calculated, now we proceed to fill the matrix Q with training values ("Training")
print("Matrix P calculated.")

print("Training the agent.")
# Hyperparameters
alpha = 0.6
gamma = 0.4
epsilon = 0.9
matrix_Q = numpy.zeros((256, 81))
if(q_table_training==1):
    for i in range(0, 300):
      state = random.randint(0, 255)
      epochs = 0
      while (epochs < 1200):
             # sometimes action will be the position where the maximum value of the matrix Q is in the row with the state number
             # will again be a random value, the decision to assign action will depend on the epsilon value
             if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 80)
             else:
                action = numpy.argmax(matrix_Q[state])

             reward2 = reward_matrixP[state,action]
             next_state = next_state_matrixP[state,action]
             next_state = int(next_state)
             old_value =  matrix_Q[state,action]
             # the maximum of table q must be found in the state next state
             next_max = numpy.amax(matrix_Q[next_state], axis=0)
             new_value = (1 - alpha) * old_value + alpha * (reward2 + gamma * next_max)
             matrix_Q[state, action] = new_value
             state = next_state
             epochs = epochs + 1

    print("...Training finished.\n")
    numpy.savetxt('q_table.txt',matrix_Q)  # this code saves values to training matrix q

### TRAINING CODE END

### RECOMMENDATION FUNCTION
def recommendation_next_state(state):
    print("\nRECOMMENDATION SYSTEM")
    action = numpy.argmax(matrix_Q[state])
    recommendation = dec_a_base3(action)
    print("Actual state: ", state)
    print("Action: ", action)
    return recommendation
### RECOMMENDATION FUNCTION END

### REWARD UPDATE FUNCTION
def update_Q_table(current_state, previous_state):
    # previous_state must be decimal
    state_str=str(previous_state[0])+str(previous_state[1])+str(previous_state[2])+str(previous_state[3])+str(previous_state[4])+str(previous_state[5])+str(previous_state[6])+str(previous_state[7])
    previous_state_dec=int(state_str, base=2)
    print("\nREWARD MATRIX UPDATE")
    action = numpy.argmax(matrix_Q[previous_state_dec])
    action_P = reward_matrixP[previous_state_dec]
    old_value =  matrix_Q[previous_state_dec,action]
    for i in range(4):
	    if(previous_state[i+4]==0 and current_state[i+4]==0):
		    reward[i]=5
	    if(previous_state[i+4]==0 and current_state[i+4]==1):
		    reward[i]=0
	    if(previous_state[i+4]==1 and current_state[i+4]==0):
		    reward[i]=10
	    if(previous_state[i+4]==1 and current_state[i+4]==1):
		    reward[i]=2
    new_reward = reward[0]+reward[1]+reward[2]+reward[3]
    next_state = next_state_matrixP[previous_state_dec,action]
    next_state = int(next_state)
    next_max = numpy.amax(matrix_Q[next_state], axis=0)
    new_value = (1 - alpha) * old_value + alpha * (new_reward + gamma * next_max)
    matrix_Q[previous_state_dec, action] = new_value
    print("Previous state:", previous_state)
    print("Current state:  ", current_state)
    print("New reward:", new_reward)
    print("Next max:", next_max)
    print("Old_value Q:", old_value)
    print("New_value Q:", new_value)
### REWARD UPDATE FUNCTION END

### ACTION TO STL PLAN FUNCTION
def time_converter(matrix_Q,matrix_P):
    next_state_temp=[]
    traffic_light = ""
    matrix_P=[int(i) for i in matrix_P]

    print("State:",matrix_Q)
    print("Action:",matrix_P)
    for i in range (0,4):
        if (matrix_Q[i] ==0 and matrix_P[i]==0): 
            next_state_temp.append(0)
        if (matrix_Q[i] ==1 and matrix_P[i]==0): 
            next_state_temp.append(1)

        if (matrix_Q[i] ==0 and matrix_P[i]==1): 
            next_state_temp.append(0)
        if (matrix_Q[i] ==1 and matrix_P[i]==1): 
            next_state_temp.append(0)

        if (matrix_Q[i] ==0 and matrix_P[i]==2): 
            next_state_temp.append(1)
        if (matrix_Q[i] ==1 and matrix_P[i]==2): 
            next_state_temp.append(1)

    print("Next state: ",next_state_temp)
    for i in next_state_temp:
        if i==0:
            traffic_light=traffic_light+"ST"
        if i==1:
            traffic_light=traffic_light+"LT"
    return traffic_light,next_state_temp
    # LT means Long Time, Long time of traffic_light, ST means Short Time, Short time of traffic_light.
### ACTION TO STL PLAN FUNCTION END


### ROUTE GENERATOR FUNCTION
def generate_routefile():
    random.seed(42)  # make tests reproducible
    N = 86400  # number of time steps in a day (3600 sec x 24 hours). It is estimated, one step per one second 

    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="car" accel="0.9" decel="4.5" sigma="0.5" length="4" minGap="2.5" maxSpeed="19.67" guiShape="passenger"/>
        <vType id="bus" accel="0.6" decel="4.5" sigma="0.5" length="12" minGap="2.8" maxSpeed="16" guiShape="bus"/>
        <vType id="truck" accel="0.4" decel="4.5" sigma="0.5" length="20" minGap="3" maxSpeed="14" guiShape="truck"/>


        <route id="right" edges="1i 2o" />
        <route id="right_up" edges="1i 4o" />
        <route id="right_down" edges="1i 3o" />
        <route id="left" edges="2i 1o" />
        <route id="left_up" edges="2i 4o" />
        <route id="left_down" edges="2i 3o" />
        <route id="down" edges="4i 3o" />
        <route id="down_left" edges="4i 1o" />
        <route id="down_right" edges="4i 2o" />
        <route id="up" edges="3i 4o" />
        <route id="up_left" edges="3i 1o" />
        <route id="up_right" edges="3i 2o" />""", file=routes)
        vehNr = 0
        vehNr = 0
        day=0
        for i in range(N):
            if(0<=i and i<21600): #from 0 am to 6 am
                if (i == 0):
                    dt = datetime.now() # It generates a really RANDOM seed (with actual milliseconds in time)
                    dt = dt.microsecond 
                    random.seed(dt)
                    pWE = 1. / random.randint(48,52) #It is a probability of 1/XX (minimum 0 maximum 1) that a vehicle is created with the YY route, for each step of the execution.
                    pEW = 1. / random.randint(53,57)
                    pWN = 1. / random.randint(43,47)
                    pWS = 1. / random.randint(48,52)
                    pEN = 1. / random.randint(53,57)
                    pES = 1. / random.randint(58,62)
                    pNE = 1. / random.randint(44,48)
                    pNW = 1. / random.randint(42,46)
                    pNS = 1. / random.randint(38,42)
                    pSE = 1. / random.randint(45,49)
                    pSW = 1. / random.randint(48,52)
                    pSN = 1. / random.randint(38,42)

            if(21600<=i and i<32400): #from 6 am to 9am 
                if (i == 21600):
                    dt = datetime.now() # It generates a really RANDOM seed (with actual milliseconds in time)
                    dt = dt.microsecond 
                    random.seed(dt)
                    pWE = 1. / random.randint(14,16) 
                    pEW = 1. / random.randint(17,19)
                    pWN = 1. / random.randint(19,21)
                    pWS = 1. / random.randint(18,20)
                    pEN = 1. / random.randint(22,24)
                    pES = 1. / random.randint(17,19)
                    pNE = 1. / random.randint(10,12)
                    pNW = 1. / random.randint(11,13)
                    pNS = 1. / random.randint(4,4)
                    pSE = 1. / random.randint(10,12)
                    pSW = 1. / random.randint(9,11)
                    pSN = 1. / random.randint(5,5)
            if(32400<=i and i<39600): #from 9am to 11am 
                if (i == 32400):
                    dt = datetime.now() # It generates a really RANDOM seed (with actual milliseconds in time)
                    dt = dt.microsecond 
                    random.seed(dt)
                    pWE = 1. / random.randint(24,28)  
                    pEW = 1. / random.randint(26,30)
                    pWN = 1. / random.randint(32,36)
                    pWS = 1. / random.randint(31,35)
                    pEN = 1. / random.randint(24,28)
                    pES = 1. / random.randint(30,34)
                    pNE = 1. / random.randint(11,13)
                    pNW = 1. / random.randint(14,16)
                    pNS = 1. / random.randint(7,7)
                    pSE = 1. / random.randint(15,17)
                    pSW = 1. / random.randint(17,19)
                    pSN = 1. / random.randint(7,7)
            if(39600<=i and i<50400): #from 11am to 2pm
                if (i == 39600):
                    dt = datetime.now() # It generates a really RANDOM seed (with actual milliseconds in time)
                    dt = dt.microsecond 
                    random.seed(dt)
                    pWE = 1. / random.randint(20,22) 
                    pEW = 1. / random.randint(16,18)
                    pWN = 1. / random.randint(15,17) 
                    pWS = 1. / random.randint(17,19) 
                    pEN = 1. / random.randint(20,22) 
                    pES = 1. / random.randint(15,17) 
                    pNE = 1. / random.randint(13,15) 
                    pNW = 1. / random.randint(15,17)
                    pNS = 1. / random.randint(4,4)
                    pSE = 1. / random.randint(16,18)
                    pSW = 1. / random.randint(13,15)
                    pSN = 1. / random.randint(4,4)
            if(50400<=i and i<61200): #from 2pm to 5pm
                if (i == 50400):
                    dt = datetime.now() # It generates a really RANDOM seed (with actual milliseconds in time)
                    dt = dt.microsecond 
                    random.seed(dt)
                    pWE = 1. / random.randint(19,21)  
                    pEW = 1. / random.randint(17,19)
                    pWN = 1. / random.randint(17,19)
                    pWS = 1. / random.randint(20,22)
                    pEN = 1. / random.randint(23,25)
                    pES = 1. / random.randint(22,24)
                    pNE = 1. / random.randint(17,19)
                    pNW = 1. / random.randint(15,17)
                    pNS = 1. / random.randint(6,6)
                    pSE = 1. / random.randint(16,18)
                    pSW = 1. / random.randint(18,20)
                    pSN = 1. / random.randint(6,6)
            if(61200<=i and i<72000): #from 5pm to 8pm
                if (i == 61200):
                    dt = datetime.now() # It generates a really RANDOM seed (with actual milliseconds in time)
                    dt = dt.microsecond 
                    random.seed(dt)
                    pWE = 1. / random.randint(18,20)  
                    pEW = 1. / random.randint(17,19)
                    pWN = 1. / random.randint(18,21)
                    pWS = 1. / random.randint(16,18)
                    pEN = 1. / random.randint(17,19)
                    pES = 1. / random.randint(16,18)
                    pNE = 1. / random.randint(12,14)
                    pNW = 1. / random.randint(11,13)
                    pNS = 1. / random.randint(6,6)
                    pSE = 1. / random.randint(14,16)
                    pSW = 1. / random.randint(12,14)
                    pSN = 1. / random.randint(4,4)
            if(72000<=i and i<=86400): #from 8pm to 12pm
                if (i == 72000):
                    dt = datetime.now() # It generates a really RANDOM seed (with actual milliseconds in time)
                    dt = dt.microsecond 
                    random.seed(dt)
                    pWE = 1. / random.randint(28,32) 
                    pEW = 1. / random.randint(33,37) 
                    pWN = 1. / random.randint(23,27) 
                    pWS = 1. / random.randint(28,32)  
                    pEN = 1. / random.randint(23,27)  
                    pES = 1. / random.randint(28,32)  
                    pNE = 1. / random.randint(11,13)  
                    pNW = 1. / random.randint(12,14) 
                    pNS = 1. / random.randint(8,8)
                    pSE = 1. / random.randint(13,15)
                    pSW = 1. / random.randint(11,13)
                    pSN = 1. / random.randint(5,5)            

            if random.uniform(0, 1) < pWE:
                if (vehNr % 19)==0:
                    print('    <vehicle id="right_%i" type="bus" route="right" depart="%i" color="1,0,0" />' % (vehNr, i), file=routes)
                elif (vehNr % 99)==0:
                    print('    <vehicle id="right_%i" type="truck" route="right" depart="%i" color="0,1,0"/>' % (vehNr, i), file=routes)
                else:
                    print('    <vehicle id="right_%i" type="car" route="right" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pWN:
                if (vehNr % 19)==0:
                    print('    <vehicle id="right_up_%i" type="bus" route="right_up" depart="%i" color="1,0,0" />' % (vehNr, i), file=routes)
                elif (vehNr % 99)==0:
                    print('    <vehicle id="right_up_%i" type="truck" route="right_up" depart="%i" color="0,1,0" />' % (vehNr, i), file=routes)
                else:
                    print('    <vehicle id="right_up_%i" type="car" route="right_up" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pWS:
                if (vehNr % 19)==0:
                    print('    <vehicle id="right_down_%i" type="bus" route="right_down" depart="%i" color="1,0,0" />' % (vehNr, i), file=routes)
                elif (vehNr % 99)==0:
                    print('    <vehicle id="right_down_%i" type="truck" route="right_down" depart="%i" color="0,1,0" />' % (vehNr, i), file=routes)
                else:
                    print('    <vehicle id="right_down_%i" type="car" route="right_down" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEW:
                if (vehNr % 19)==0:
                    print('    <vehicle id="left_%i" type="bus" route="left" depart="%i" color="1,0,0" />' % (vehNr, i), file=routes)
                elif (vehNr % 99)==0:
                    print('    <vehicle id="left_%i" type="truck" route="left" depart="%i" color="0,1,0" />' % (vehNr, i), file=routes)
                else:
                    print('    <vehicle id="left_%i" type="car" route="left" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEN:
                if (vehNr % 19)==0:
                    print('    <vehicle id="left_up_%i" type="bus" route="left_up" depart="%i" color="1,0,0" />' % (vehNr, i), file=routes)
                elif (vehNr % 99)==0:
                    print('    <vehicle id="left_up_%i" type="truck" route="left_up" depart="%i" color="0,1,0" />' % (vehNr, i), file=routes)
                else:
                    print('    <vehicle id="left_up_%i" type="car" route="left_up" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pES:
                if (vehNr % 19)==0:
                    print('    <vehicle id="left_down_%i" type="bus" route="left_down" depart="%i" color="1,0,0" />' % (vehNr, i), file=routes)
                elif (vehNr % 99)==0:
                    print('    <vehicle id="left_down_%i" type="truck" route="left_down" depart="%i" color="0,1,0" />' % (vehNr, i), file=routes)
                else:
                    print('    <vehicle id="left_down_%i" type="car" route="left_down" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS:
                if (vehNr % 19)==0:
                    print('    <vehicle id="down_%i" type="bus" route="down" depart="%i" color="1,0,0" />' % (vehNr, i), file=routes)
                elif (vehNr % 99)==0:
                    print('    <vehicle id="down_%i" type="truck" route="down" depart="%i" color="0,1,0" />' % (vehNr, i), file=routes)
                else:
                    print('    <vehicle id="down_%i" type="car" route="down" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNW:
                if (vehNr % 19)==0:
                    print('    <vehicle id="down_left_%i" type="bus" route="down_left" depart="%i" color="1,0,0" />' % (vehNr, i), file=routes)
                elif (vehNr % 99)==0:
                    print('    <vehicle id="down_left_%i" type="truck" route="down_left" depart="%i" color="0,1,0" />' % (vehNr, i), file=routes)
                else:
                    print('    <vehicle id="down_left_%i" type="car" route="down_left" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNE:
                if (vehNr % 19)==0:
                    print('    <vehicle id="down_right_%i" type="bus" route="down_right" depart="%i" color="1,0,0" />' % (vehNr, i), file=routes)
                elif (vehNr % 99)==0:
                    print('    <vehicle id="down_right_%i" type="truck" route="down_right" depart="%i" color="0,1,0" />' % (vehNr, i), file=routes)
                else:
                    print('    <vehicle id="down_right_%i" type="car" route="down_right" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pSN:
                if (vehNr % 19)==0:
                    print('    <vehicle id="up_%i" type="bus" route="up" depart="%i" color="1,0,0" />' % (vehNr, i), file=routes)
                elif (vehNr % 99)==0:
                    print('    <vehicle id="up_%i" type="truck" route="up" depart="%i" color="0,1,0" />' % (vehNr, i), file=routes)
                else:
                    print('    <vehicle id="up_%i" type="car" route="up" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pSW:
                if (vehNr % 19)==0:
                    print('    <vehicle id="up_left_%i" type="bus" route="up_left" depart="%i" color="1,0,0" />' % (vehNr, i), file=routes)
                elif (vehNr % 99)==0:
                    print('    <vehicle id="up_left_%i" type="truck" route="up_left" depart="%i" color="0,1,0" />' % (vehNr, i), file=routes)
                else:
                    print('    <vehicle id="up_left_%i" type="car" route="up_left" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pSE:
                if (vehNr % 19)==0:
                    print('    <vehicle id="up_right_%i" type="bus" route="up_right" depart="%i" color="1,0,0" />' % (vehNr, i), file=routes)
                elif (vehNr % 99)==0:
                    print('    <vehicle id="up_right_%i" type="truck" route="up_right" depart="%i" color="0,1,0" />' % (vehNr, i), file=routes)
                else:
                    print('    <vehicle id="up_right_%i" type="car" route="up_right" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1  

        print("</routes>", file=routes)
### ROUTE GENERATOR FUNCTION END

### QUEUES COUNT FUNCTION
def numberv(lane):
            queu = 0
            for k in traci.lane.getLastStepVehicleIDs(lane):
                if (traci.vehicle.getLanePosition(k)) < 486 and (traci.vehicle.getLanePosition(k) > 200):
                   queu += 1
            return queu   
def numberv1(lane):
            queu = 0
            for k in traci.lane.getLastStepVehicleIDs(lane):
                if (traci.vehicle.getLanePosition(k)) < 988 and (traci.vehicle.getLanePosition(k) > 400):
                   queu += 1
            return queu 
### QUEUES COUNT FUNCTION END

### SUMO CODE 
# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
from sumolib import checkBinary  # noqa
import traci  # noqa
def run():
    # """execute the TraCI control loop"""
    ### COUNTERS RESET
    step = 0
    reviewer = 0
    cicle = 0
    flag = 0
    recommendation = 0
    current_state=[0,0,0,0,0,0,0,0]
    previous_state=[0,0,0,0,0,0,0,0]

    ### MAX PARAMETERS SEARCH IN DB
    sql = "SELECT east_west_max, south_north_max, west_east_max, north_south_max FROM crosses where id='1';"
    cursor.execute(sql)
    parameters = cursor.fetchall()
    for parameter in parameters:
        print("Maximum parameters obtained ")

    # It starts with phase 2 where EW has green
    traci.trafficlight.setProgram("0","0")

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        if (reviewer == 150):
            reviewer = 0
            cicle += 1

            ### QUEUE COUNT
            queuWE1 =  numberv("1i_0")      
            queuWE2 =  numberv("1i_1")      
            queuEW1 =  numberv("2i_0")      
            queuEW2 =  numberv("2i_1")      
            queuSN1 =  numberv1("3i_0")      
            queuSN2 =  numberv1("3i_1")
            queuSN3 =  numberv1("3i_2")
            queuNS1 =  numberv1("4i_0")
            queuNS2 =  numberv1("4i_1")
            queuNS3 =  numberv1("4i_2")      
            ### QUEUE COUNT END
            
            queuEW=queuEW1+queuEW2
            queuSN=queuSN1+queuSN2+queuSN3
            queuWE=queuWE1+queuWE2
            queuNS=queuNS1+queuNS2+queuNS3
            sql = "INSERT INTO queues (east_west, south_north, west_east, north_south, simulation_type, creation_date, creation_hour) VALUES(%s, %s, %s, %s,'%s','CURDATE()','CURTIME()');" 
            val = (queuEW, queuSN, queuWE, queuNS, simulation_type)
            cursor.execute(sql,val)
            myConnection.commit()

            # THIS IS THE PRIORITY ORDER OF THE TRAFFIC LIGHT, FIRST GOES TO GREEN EW, THEN SN, THEN WE AND FINALLY NS
            print("\n\nCICLE: ", cicle)
            print("queuEW: ", queuEW)
            print("queuSN: ", queuSN)
            print("queuWE: ", queuWE)
            print("queuNS: ", queuNS)

            ### RL SIMULATION TYPE
            if (simulation_type == 2):
                short_queue = 1
                if (recommendation==1): #2 cycle counter after recommendation
                    flag += 1

                if(flag==2):
                    flag = 0  
                    recommendation = 0
                    short_queue = 0     #It is used to know if it should return to the initial state of the traffic lights

                    if queuEW > parameter[0]:
                        current_state[4]=1
                        short_queue = 1
                    else:    
                        current_state[4]=0
                    if queuSN > parameter[1]:
                        current_state[5]=1
                        short_queue = 1
                    else:
                        current_state[5]=0    
                    if queuWE > parameter[2]:
                        current_state[6]=1
                        short_queue = 1
                    else:    
                        current_state[6]=0
                    if queuNS > parameter[3]:
                        current_state[7]=1
                        short_queue = 1
                    else:
                        current_state[7]=0

                    if ((previous_state == current_state) and (current_state[0] == 0) and (current_state[1] == 0) and (current_state[2] == 0) and (current_state[3] == 0) and (current_state[4] == 0) and (current_state[5] == 0) and (current_state[6] == 0) and (current_state[7] == 0) and (short_queue == 0)):
                        short_queue = 1
                        print("the system should not request recommendation")
                    
                
                    update_Q_table(current_state, previous_state) #It is analyzed if the recommendation improves the status from traffic
                    numpy.savetxt('q_table.txt',matrix_Q)
            
                if ((queuEW > parameter[0] or queuSN > parameter[1] or queuWE > parameter[2] or queuNS > parameter[3]) and recommendation == 0) or (short_queue == 0): #it checks if there is any long queue or if short qeueu = 0
                    recommendation = 1  #The recommendation is executed
                    short_queue = 1   # This is so that you do not enter this if again, if there is no long tail. If there is no long tail, only enter if it finishes from updating the Q array and the queues are all less than the max so that it returns to STSTSTST

                    if queuEW > parameter[0]:
                        current_state[4]=1
                    else:    
                        current_state[4]=0
                    if queuSN > parameter[1]:
                        current_state[5]=1
                    else:
                        current_state[5]=0    
                    if queuWE > parameter[2]:
                        current_state[6]=1
                    else:    
                        current_state[6]=0
                    if queuNS > parameter[3]:
                        current_state[7]=1
                    else:
                        current_state[7]=0
                

                    previous_state = current_state.copy() #The value of the queues is passed to make the comparison

                    state_bin = str(current_state[0])+str(current_state[1])+str(current_state[2])+str(current_state[3])+str(current_state[4])+str(current_state[5])+str(current_state[6])+str(current_state[7])
                    state_dec = int(state_bin, base=2)
                    action_tt = recommendation_next_state(state_dec)  
                    action_tt = list(str(action_tt))
                    
                    for i in range(len(action_tt),4):  
                        action_tt.insert(0,0)
                    new_time, new_state = time_converter(current_state,action_tt)
                
                    for i in range(0,4):
                        current_state[i] = new_state[i]

                    print("Traffic Light Program recommendation: ",new_time)
                    if (new_time == "STSTSTST"):
                        traci.trafficlight.setProgram("0", "0")
                    else:
                        traci.trafficlight.setProgram("0", new_time)
                   
                ### RL SIMULATION TYPE END

            ### BASIC SIMULATION TYPE
            if (simulation_type == 1):
                flag +=1
                if(flag==2):
                    flag = 0
                    if queuEW > parameter[0]:
                        current_state[4]=1
                    else:    
                        current_state[4]=0
                    if queuSN > parameter[1]:
                        current_state[5]=1
                    else:
                        current_state[5]=0    
                    if queuWE > parameter[2]:
                        current_state[6]=1
                    else:    
                        current_state[6]=0
                    if queuNS > parameter[3]:
                        current_state[7]=1
                    else:
                        current_state[7]=0

                    new_time=""
                    for i in range (4,8):
                        if (current_state[i] ==0): 
                            new_time=new_time+"ST"
                        if (current_state[i] ==1): 
                            new_time=new_time+"LT"
                    print("Traffic Light Program recommendation: ",new_time)
                    if (new_time=="STSTSTST"):
                        traci.trafficlight.setProgram("0", "0")
                    else:
                        traci.trafficlight.setProgram("0", new_time)
                    
                ### BASIC SIMULATION TYPE END
    
                ### FIX TIME SIMULATION TYPE 
                if (simulation_type == 0):
                    traci.trafficlight.setProgram("0", "0")
                ### FIX TIME SIMULATION TYPE END


        step += 1
        reviewer += 1
    traci.close()
    cursor.close()
    connection.close()
    sys.stdout.flush()
### SUMO CODE END


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    #server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    if (route_generation == 1):
       generate_routefile() # ROUTE GENERATOR FUNCTION

    traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])
    run()
