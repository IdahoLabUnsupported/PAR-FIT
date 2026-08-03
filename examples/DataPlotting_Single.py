# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:29:17 2026

Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""
import pandas as pd
import matplotlib.pyplot as plt

value = 300
name  = "../datasets/007_Q2_015_0768_T/histories_short_print_" + str(value) + ".csv"
df = pd.read_csv(name)

plt.figure()
plt.plot(df["time"], df["TA21s1"])
