import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
from numpy.random import randint
from numpy.random import rand

class genetics():

#-----Params initialization
    def __init__(self, a, b, p, N, pc, pm ):
        self.a = a
        self.b = b
        self.p = p
        self.N = N
        self.pc = pc
        self.pm = pm

#-----Function
    def function(self, x):
        return float(((x * m.sin(10 * x * 3.14)) + 1))

#-----Plotting the function
    def show_plot(self):
        xi = np.arange(-1,2,0.1)
        l = []
        for i in xi:
            l.append(self.function(i))

        plt.plot(xi, l)
        plt.xlim(-1, 2)
        plt.show()

#-----Compute the number of bits for croms
    def mi(self):
        for i in range(2,31):
            if (self.b-self.a)*np.power(10,self.p) <= np.power(2,i):
                return i
                break

#-----Converting the croms to base10 number
    def convert2_10(self,list):
        xp = 0
        for i in range(self.mi()):
            xp = xp + list[i]*m.pow(2,i)
        x = self.a + xp * (self.b - self.a)/(m.pow(2,self.mi())-1)
        return x

#-----Init the population
    def init_pop(self):
        p = [randint(0, 2, self.mi()).tolist() for _ in range(self.N)]
        return p

#-----Selection of the croms
#-----Remove "weak" croms
    def selection(self):
        # first iteration
        ev = []
        for i in range(self.N):
            ev.append(self.function(self.convert2_10(self.init_pop()[i])))
        sel = []
        sel.append(max(ev))

        # next iter
        #Sum of total fitness
        F = float(sum(ev))

        #selection probability
        p = []
        for individ in ev:
            p.append(float(individ/F))

        # cumulate selection probability
        q = []
        qi = 0
        for i in range(self.N):
            qi = p[i] + qi
            q.append(qi)
        q.append(max(ev))

        # remove "weaks" crom
        init_pop = self.init_pop()
        weak_crom = []
        for i in range(1,self.N):
            if q[i-1] < np.random.rand() <= q[i]:
                #print("weak crom = ", init_pop[i])
                weak_crom.append(init_pop[i])
        for i in init_pop:
            for j in weak_crom:
                if i==j:
                    init_pop.remove(i)

        return init_pop
#-----Croms cross operation
    def incrutisare(self):
        pop_select = []

        #random select the croms for cross
        for poz in range(round((self.pc*self.N))):
            if np.random.uniform() < self.pc:
                pop_select.append([self.selection()[poz], poz])

        #We need even number for the operation
        if len(pop_select) % 2 != 0:
           pop_select.pop()

        def pairwise(iterable):
            a = iter(iterable)
            return zip(a, a)

        #Each pair was crossed and the population was updated with crossed croms
        incr = []
        for x, y in pairwise(pop_select):
            pos = np.random.choice(range(1, self.mi()-1))

            temp_x = [y for x in [[x[0][:pos], y[0][pos:]]] for y in x]
            o_x = [y for x in [temp_x[0],temp_x[1]] for y in x]
            incr.append(o_x)
            x[0][0] = o_x


            temp_y = [y for x in [[y[0][:pos], x[0][pos:]]] for y in x]
            o_y = [y for x in [temp_y[0], temp_y[1]] for y in x]
            incr.append(o_y)
            y[0][0] = o_y

        return pop_select

#-----Croms mutation operation
#-----!Not ready yet!
    def mutation(self):
        for indiv in range(round(self.pm * len(self.incrutisare()))):
            for bit in range(self.mi()):
                if np.random.uniform() < self.pm:
                    if self.incrutisare()[0][indiv][bit] == 0:
                        self.incrutisare()[0][indiv][bit] = 1
                    else:
                        self.incrutisare()[0][indiv][bit] = 0

        return self.incrutisare()



# -----Define new exceptions
class Error(Exception):
    pass

class InvalidInterval(Error):
    pass

class NotInRange(Error):
    pass

class ProbabilityRange(Error):
    pass


#-----Main function
#-----!to contruct the alg here!
if __name__ == '__main__':

#-----Handling exceptions
#-----!to create a function or a class for this!
    print("--To initialized the genetic algorithm, please insert the values for given parameters: ")
    while True:
        try:
            a = float(input("*Lower range limit*  a = "))
            b = float(input("*Upper range limit*  b = "))
            p = int(input("Precision*  p = "))
            N = int(input("*Number of generations*  N = "))
            pc = float(input("*Probability to choose an crom for cross*  pc = "))
            pm = float(input("*Probability to choose an crom for mutation*  pm = "))
            print("\n")
            if a > b or a == b:
                raise InvalidInterval
            if p < 0 or p > 30:
                raise NotInRange
            if N < 0:
                raise NotInRange
            if pc < 0 or pc > 1:
                raise ProbabilityRange
            if pm < 0 or pm > 1:
                raise ProbabilityRange
            break

        except InvalidInterval:
            print("Please insert an valid interval!")
            print()
        except NotInRange:
            print("Please insert a valid precision or number of generation!")
            print()
        except ProbabilityRange:
            print("Please insert a valid probability!")
            print()

    GA = genetics(a, b, p,
                  N, pc, pm)
    GA.show_plot()


