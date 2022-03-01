#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 1 14:01:04 2020
@author: tristan
"""
import os
import pandas as pd
import numpy as np

#~~~~~~~~~~~~~~~~~
state_mapping = 1
#~~~~~~~~~~~~~~~~~

if state_mapping == 1:
    S = 21
    A = 2
    gam = 0.5
    alpha = 0.01
    maxIters = 5
    input_file = "random_policy_runs_mapping_1.csv"
    output_file = "Sarsa_Policy_1.policy"
elif state_mapping == 2:
    S = 183
    A = 2
    gam = 0.5
    alpha = 0.01
    maxIters = 5
    input_file = "random_policy_runs_mapping_2.csv"
    output_file = "Sarsa_Policy_2.policy"

df = pd.read_csv(input_file)
s_data = df["s"]
a_data = df["a"]
r_data = df["r"]
sp_data = df["sp"]

# Initialize action value function to all zeros
Q = np.zeros((S, A))
for i in range(maxIters):
    print(i)
    for k in range(len(df)-1):
        s = s_data[k]
        a = a_data[k]
        r = r_data[k]
        sp = sp_data[k]
        ap = a_data[k+1]
        Q[s, a] = Q[s, a] + alpha*(r + gam*Q[sp, ap] - Q[s, a]) #Sarsa
        #Q[s, a] = Q[s, a] + alpha*(r + gam*max(Q[sp, :]) - Q[s, a]) #Q-learn

policy = np.argmax(Q, axis=1)
# Get output file name and path
output_dir = os.getcwd()
output_file = os.path.join(output_dir, output_file)
# Open output file
DF = open(output_file, "w")
# Iterate through each value in policy, writing to output file
for i in range(S):
    DF.write(f"{policy[i]}\n")
# Close output file
DF.close()