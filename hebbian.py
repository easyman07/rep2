
###NAND GATE HEBBIAN

from ast import main
import numpy as np

class Hebbian :
      def __init__(self) :
        pass
      def HebbAND(self) :
        print ("\n")
        x1 = [1, 1, -1, -1] #input 1 
        x2 = [1, -1, 1, -1] #input 2
        b = [1, 1, 1, 1] # bias 
        y = [-1, 1, 1, 1] #target


        w1=0
        w2=0
        b=0

        #new values of x1 x2 and b

        w1n=0
        w2n=0
        bn=0

       
        for i in range (0,4) :
             w1n = w1+x1[i]*y[i] # formula to calculate w1 new
             w2n = w2+x2[i]*y[i] #formula  to claculate w2 new 
             bn = b+y[i]
             print ("[+] Weights and bias after iteration "+str(i)+" :")
             print ("W1 :" +str(w1n))
             print ("W2 :" +str(w2n))
             print ("b  :" +str(bn))
             w1 = w1n
             w2 = w2n
             b = bn
        print ("\n")
        print ("Final Weights :")
        print (w1, w2, b)

print("")