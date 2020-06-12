#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 05:30:46 2020

@author: anusarfarooqui
"""
import os
    
os.chdir('/Users/anusarfarooqui/Docs/Matlab/')
import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_excel('shootings.xlsx',header=0)
print(df)

expr = """Fatal ~ Homicide + Population + Black"""
y_train, X_train = dmatrices(expr,df,return_type='dataframe')
print(y_train)

# train poisson
poisson_results=sm.GLM(y_train,X_train,family=sm.families.Poisson()).fit()
# print(poisson_results.summary())
# print(poisson_results.mu)

import statsmodels.formula.api as smf

df['BB_Lambda']=poisson_results.mu

df['aux_ols_dep']=df.apply(lambda x: ((x['Fatal'] - x['BB_Lambda'])**2 - x['Fatal'])/x['BB_Lambda'],axis=1)

ols_expr="""aux_ols_dep~BB_Lambda - 1"""

aux_olsr_results = smf.ols(ols_expr,df).fit()

print(aux_olsr_results.params)
print(aux_olsr_results.tvalues)

nb2_results = sm.GLM(y_train,X_train,family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
print(nb2_results.summary())

