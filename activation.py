# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 08:19:54 2022

@author: ar2806
"""
import tensorflow as tf 


class Activations(object):
    def b0(self,a,b):
        return a+b
    def b1(self,a,b):
        return a*b
    def b2(self,a,b):
        return tf.math.maximum(tf.math.maximum(a,0),b)
    def b3(self,a,b):
        return a*tf.math.erf(b)
    def b4(self,a,b):
        return tf.math.maximum(a,b)
    def b5(self,a,b):
        return a*tf.math.exp(b)
    def b6(self,a,b):
        return tf.math.sigmoid(a)*b
    
    def f0(self,x):
        return x
    def f1(self,x):
        return tf.math.exp(x)
    def f2(self,x):
        return tf.math.abs(x)
    def f3(self,x):
        return -x
    def f4(self,x):
        return tf.math.exp(-x)
    def f5(self,x):
        return tf.math.minimum(x,0)
    def f6(self,x):
        return tf.math.erf(x)
    def f7(self,x):
        return tf.math.sigmoid(x)
    def f8(self,x):
        return x*tf.math.erf(x)
    def f9(self,x):
        return tf.math.maximum(x,0)
    def f10(self,x):
        return tf.math.sin(x)
    def f11(self,x):
        return tf.math.atan(x)
    def f12(self,x):
        return tf.math.tanh(x)
    def f13(self,x):
        return tf.math.log(1+tf.math.exp(x))
    def f14(self,x):
        return tf.math.sinh(x)
    def f15(self,x):
        return tf.math.log(1+tf.math.exp(x))
    def f16(self,x):
        return tf.math.atanh(x)
    def f17(self,x):
        return tf.math.asinh(x)
    
def get_activation(pop,value):
    pop_len =len(pop)
    functions=Activations()
    if pop_len==3:
        x_1=getattr(functions, 'f'+str(pop[1]))(value)
        x_2=getattr(functions, 'f'+str(pop[2]))(value)
        y=getattr(functions, 'b'+str(pop[0]))(x_1,x_2)
    elif pop_len==5:
        x_1=getattr(functions, 'f'+str(pop[2]))(value)
        x_2=getattr(functions, 'f'+str(pop[3]))(value)
        x_3=getattr(functions, 'f'+str(pop[4]))(value)
        y_1=getattr(functions, 'b'+str(pop[1]))(x_1,x_2)
        y=getattr(functions, 'b'+str(pop[0]))(y_1,x_3)
    else:
        x_1=getattr(functions, 'f'+str(pop[3]))(value)
        x_2=getattr(functions, 'f'+str(pop[4]))(value)
        x_3=getattr(functions, 'f'+str(pop[5]))(value)
        x_4=getattr(functions, 'f'+str(pop[6]))(value)
        y_1=getattr(functions, 'b'+str(pop[1]))(x_1,x_2)
        y_2=getattr(functions, 'b'+str(pop[2]))(x_3,x_4)
        y=getattr(functions, 'b'+str(pop[0]))(y_1,y_2)
    return y