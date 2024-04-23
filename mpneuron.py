import numpy as np
import pandas as pd
#AND GATE

x1 = [0, 0, 1, 1]  # Bipolar values
x2 = [0, 1, 0, 1]  # Bipolar values
y = [0, 1, 1, 1]  # target

#getting weight frm user 
w1= float(input('Enter Weight(w1):'))
w2= float(input('Enter Weight(w2):'))
th=input("Enter Threshold value:")
yact=[]

def mpneuron(x1,x2,y):
    for i in range(0,4):
       yin=(w1*x1[i])+(w2*x2[i])

       if(yin==th):
           yact.append(1)
       else:
           yact.append(0)

mpneuron(x1,x2,y)
table=pd.DataFrame({"x1":x1,"x2":x2,"y":y,"yact":y})
print (table)




