# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 08:28:20 2022

@author: ar2806
"""
from ga_utils import create_initial_pop, next_generation
from fitness import fitness
def run_ga():
    pop_size = 30
    gen = 50
    #chromosome_shape = [7,18,18]
    #term_threshold = 0.80
    population = create_initial_pop(pop_size)
    is_loop = True
    highest_accuracy = []
    for i in range(gen):
        print("Start of Generation No: ", i+1)
        population, scores = fitness(population)
        for pop,sc in zip(population,scores):
            print(pop, sc)
        print("End of a Generation...")
        highest_accuracy.append(max(scores))
        '''for (pop,score) in zip(population,scores):
            if score >= term_threshold:
                print('Expected Result is Achieved')
                print(pop)
                is_loop = False
                break
        if not is_loop:
            break'''
        population = next_generation(population, scores)
        print('completion of a generation')
        #if termination criteria doesn't pass then
        #new_population = next_generation(pop_with_score), it includes cross over and mutation
        #this new population will be used to check fitness function again
        #when termination criteria passes, show the best population
    print("Highest Accuracy over all generations:", highest_accuracy)

if __name__ == "__main__":
    run_ga()
