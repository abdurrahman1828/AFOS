# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 08:19:21 2022

@author: ar2806
"""

import random
import numpy as np
import tensorflow as tf
import pandas as pd

def create_initial_pop(pop_size=30):
    #arg pop is the array of the shape of [no of binary function, no of unary function, no unary function] like [3,4,4]
    #arg pop_size is the number of members in a single generation, like 10,20,30 or 50
    #returns a random population that contains 0 and 1
    
    final_population =[]
    for i in range(pop_size):
        chromosome_len = [3,5,7]
        pop_len = random.choice(chromosome_len)
        if pop_len == 3:
            pop = [7,17,17]
        elif pop_len == 5:
            pop = [7,7,17,17,17] 
        else: 
            pop = [7,7,7,17,17,17,17] 
        single_pop=[]
        for j in range(len(pop)):
            single_pop.append(random.randint(0, pop[j]-1))
        final_population.append(single_pop)
    return final_population



def crossover(a,b): #one point cross-over
    new_a = a
    new_b = b
    prob = random.uniform(0,1)
    if prob<0.8:
        a_len = len(a)
        b_len = len(b)
        if a_len==b_len:
            site = random.randint(0,a_len-1)
        elif a_len>b_len:
            site = random.randint(0,b_len-1)
        else:
            site = random.randint(0,a_len-1)
        new_a = [*a[:site], *b[site:]]
        new_b = [*b[:site], *a[site:]]
    return new_a, new_b



def mutation(a):
    pop_len = len(a)
    prob = random.uniform(0,1)
    unary_replace = random.randint(0,16)
    binary_replace = random.randint(0,6)
    if prob>0.3:
        if pop_len==3:
            loc = random.randint(0,2)
            if loc==0:
                a[0]= binary_replace
            elif loc==1:
                a[1]= unary_replace
            else:
                a[2]= unary_replace
        if pop_len==5:
            loc = random.randint(0,4)
            if loc==0:
                a[0]= binary_replace
            elif loc==1:
                a[1]= binary_replace
            elif loc==2:
                a[2]= unary_replace
            elif loc==3:
                a[3]= unary_replace
            else:
                a[4]= unary_replace
        if pop_len==7:
            loc = random.randint(0,6)
            if loc==0:
                a[0]= binary_replace
            elif loc==1:
                a[1]= binary_replace
            elif loc==2:
                a[2]= binary_replace
            elif loc==3:
                a[3]= unary_replace
            elif loc==4:
                a[4]= unary_replace
            elif loc==5:
                a[5]= unary_replace
            else:
                a[6]= unary_replace
    return a            




def next_generation(population, scores):#input arg is a dataframe containing population ans scores
    #need to take the top 2 best scoring chromosome and save it to another dataframe for passing them to crossover
    #call the crossover function to get the new 2 chromosomes
    #call the create initial population function to generate 6 random chromosome
    # so, best 2 + crossover 2 + random new 6 = 10 will be the new generation
    population = pd.DataFrame(population)
    population['score']= scores
    population.sort_values(by= ['score'], ascending = False, inplace=True)
    population.drop(['score'], inplace=True, axis=1, errors='ignore')
    population.drop_duplicates(subset=None, keep='first', inplace=True)
    new_random_len = 30 - len(population.index)
    best_pop =  population[:6]
    #print(best_pop.head())
    mut_ind = random.randint(0,5)
    mut_candidate = best_pop[mut_ind:mut_ind]
    best_pop = best_pop.values.tolist()
    new_best_pop = []
    for i in best_pop:
        i = [int(x) for x in i if np.isnan(x) == False]
        new_best_pop.append(i)

    for i in [0,2,4]:
        new_a, new_b = crossover(new_best_pop[i], new_best_pop[i+1])
        new_best_pop.append(new_a)
        new_best_pop.append(new_b)
        i = i + 2
    mut_pop = mutation(mut_candidate)
    random_pop = create_initial_pop(18+new_random_len)
    best_pop = [*new_best_pop, *mut_pop.values.tolist(), *random_pop]
    print('Passing to Next Generation')
    return best_pop


