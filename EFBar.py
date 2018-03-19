import random
import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np

#import numpy as np

parser = argparse.ArgumentParser(description="EFBar")
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-question", type=int)
parser.add_argument("-prob")
parser.add_argument("-repetitions", type=int)
parser.add_argument("-strategy")
parser.add_argument("-state", type=int)
parser.add_argument("-crowded", type=int)
parser.add_argument("-lam","-lambda","-arg_lambda",type=int)
parser.add_argument("-h")
parser.add_argument("-weeks", type = int)
parser.add_argument("-max_t", type = int)
args = parser.parse_args()

class Strategy:
    def __init__(self, h, p, a, b):
        self.h = h
        self.p = p
        self.a = a
        self.b = b
    
    def __str__(self):
        return (str(self.h) + " " + str(self.p) + " " + str(self.a) + " " + str(self.b))
    
    def updatea(self,i,j,new):
        self.a[i][j] = new

    def updateb(self,i,j,new):
        self.b[i][j] = new

def sample(vector_probabilities, repetitions):
    array_vector_probabilities = vector_probabilities.split(" ")
    for i in range(0, repetitions):
        print(str(dist(array_vector_probabilities)))
        

def dist(array_vector_probabilities):
    random_number = random.uniform(0,1)
    n = 0
    accumulation = 0
    for p in array_vector_probabilities:
        accumulation += float(p)
        if random_number < accumulation:
            #ans = n
            #break
            return n
        else:
            n += 1
    #return n

def steprep(strategy, state, crowded, repetitions):
    for i in range(0, repetitions):
        a = step(strategy, state, crowded)
        print( str(a[0]) + "\t" + str(a[1]) )

def step(strategy, state, crowded):
    lstrategy = strategy.split(" ")
    h = int(lstrategy[0])
    lstrategy = lstrategy[1:] #slice first item off
    p = []
    a = []
    b = []
    while lstrategy:
        p.append(lstrategy[0])
        lstrategy = lstrategy[1:]
        a.append(lstrategy[0:h])
        lstrategy = lstrategy[h:]
        b.append(lstrategy[0:h])
        lstrategy = lstrategy[h:]
##        print(str(p))
##        print(str(a))
##        print(str(b))
    
    if crowded:
        nextState = int(dist(a[state]))
    else:
        nextState = int(dist(b[state]))

    if random.uniform(0,1) < float(p[nextState]):
        going = 1
    else:
        going = 0

    #return(str(going) + "\t" + str(nextState))
    return [nextState] + [going]

def lStep(strategy, state, crowded):
    
    if crowded:
        nextState = int(dist(strategy.a[state]))
    else:
        nextState = int(dist(strategy.b[state]))

    if random.uniform(0,1) < float(strategy.p[nextState]):
        going = 1
    else:
        going = 0

    #return(str(going) + " " + str(nextState))
    return [nextState] + [going]
        
def crossover(strat1, strat2):
    h = strat1.h
    newp = []
    newa = []
    newb = []
    for i in range(0,h):
        rowa = []
        rowb = []
        newp.append( (strat1.p[i] + strat2.p[i])/2 )
        for j in range(0,h):
            rowa.append( (strat1.a[i][j] + strat2.a[i][j]) / 2 )
            rowb.append( (strat1.b[i][j] + strat2.b[i][j]) / 2 )
        newa.append(rowa)
        newb.append(rowb)

    return Strategy(h, newp, newa, newb)

##def crossover(strat1, strat2):
##    h = strat1.h
##    newp = []
##    newa = []
##    newb = []
##
##    if random.random() <0.5:
##        newp = strat1.p
##    else:
##        newp = strat2.p
##
##    for i in range(0,h):
##        if random.random() < 0.5:
##            newa.append(strat1.a[i])
##        else:
##            newa.append(strat2.a[i])
##        if random.random() < 0.5:
##            newb.append(strat1.b[i])
##        else:
##            newb.append(strat2.b[i])
##
##    return Strategy(h,newp,newa,newb)
    
        
                
def mutate(strategy, chi, sd):
    if random.random() > chi:
        return strategy

    for i in range(0, strategy.h):
        strategy.p[i] = random.gauss(strategy.p[i], sd)
        for j in range(0, strategy.h):
            val = strategy.a[i][j]
            #strategy.a[i][j] = random.gauss(val, sd)
            newval = random.gauss(val, sd)
            if newval < 0:
                newval = 0
            strategy.updatea(i,j,newval)
            val = strategy.b[i][j]
            #trategy.b[i][j] = random.gauss(val, sd)
            newval = random.gauss(val, sd)
            if newval < 0:
                newval = 0
            strategy.updateb(i,j,newval)

    #normalisation
    for i in range(0, strategy.h):
        totala = sum(strategy.a[i])
        totalb = sum(strategy.b[i])
        for j in range(0, strategy.h):
            #strategy.a[i][j] = strategy.a[i][j]/totala
            if totala > 0:
                strategy.updatea(i,j,strategy.a[i][j]/totala)
            else:
                strategy.updatea(i,j,(1/strategy.h))
            #strategy.b[i][j] = strategy.b[i][j]/totalb
            if totalb > 0:
                strategy.updateb(i,j,strategy.b[i][j]/totalb)
            else:
                strategy.updateb(i,j,(1/strategy.h))

    return strategy
    
        

def generateRandomStrat(h):
    p = []
    a = []
    b = []
    #lStrat = []
    #lStrat.append(h)
    for i in range(0,h):
        #make and add p_i (probability of going)
        p.append(random.uniform(0,1))
        #p.append(0.59)
        #make row for matrix a (crowded)
        r = [random.random() for k in range(0,h)]
        s = sum(r)
        r = [k/s for k in r]
        #lStrat += r
        a += [r]
        ##more efficient method using numpy
        #r = np.random.normal(size=h)
        #r /= r.sum()
        #lStrat +=r    
        #make row for matrix b (not crowded)
        r = [random.random() for k in range(0,h)]
        s = sum(r)
        r = [k/s for k in r]
        b += [r]
    strat = Strategy(h, p, a, b)
    #possible optimisation here of not translating it back to string
    ##strat = " ".join(str(x) for x in lStrat)
    return strat

def tournSelection(poplist, k):
    #k = 2
    maximum = 0
    for i in range(0,k):
        member = poplist[random.randint(0,len(poplist)-1)]
        if int(member[3]) >= int(maximum):
            winner = member[0]
            maximum = member[3]
    return winner
    
def elfarol(arg_lambda, h, weeks, max_t):
    chi = 0.1
    sd = 0.1
    k = 2
    pop = []
    for i in range(0, arg_lambda):
        pop.append( [generateRandomStrat(int(h)), 0, 0, 0] ) #tuple of strat, currentstate, currentgoing, fitness

    crowded = 0

    for t in range(0, max_t):
        total_gen = 0
        
        #evaluate
        for i in range(0, weeks):
            #count how many go
            going = 0
            for member in pop:
                sim = lStep(member[0], member[1], crowded)
                if sim[1] == 1:
                    member[1] = sim[0]
                    member[2] = 1
                    going += 1
                else:
                    member[1] = sim[0]
                    member[2] = 0
                    
            if going/arg_lambda >= 0.6:
                crowded = 1
            else:
                crowded = 0
            
            for member in pop:
                #if member said they went and it was not crowded +1 to fitness
                if member[2] == 1 and not crowded:
                    member[3] += 1
                #similarly for not go and crowded
                elif member[2] == 0 and crowded:
                    member[3] += 1
                    
            total_gen += going
                
        avg_gen = (total_gen / (arg_lambda * weeks)) * 100


            #print outputs
##            print(str(i) + "\t" + str(t) + "\t" + str(going) + "\t" + str(crowded) + "\t", end='')
##            for individual in pop:
##                if individual[2]:
##                    print(str(1) + "\t",end='')
##                else:
##                    print(str(0) + "\t",end='')
##            print("\n",end="")

        print("Gen: " + str(t) + " Average: " + str(avg_gen) +"%")

            
        #GA
        newpop = []
        for i in range(0, arg_lambda):
            x = tournSelection(pop,k)
            y = tournSelection(pop,k)
            mx = mutate(x, chi, sd)
            my = mutate(y, chi, sd)
            child = crossover(mx, my)
            newpop.append([child,0,0,0])
        pop = newpop
        
    print(str(pop[0][0].p) + " " + str(pop[0][0].a) + " " + str(pop[0][0].b) )


def elfaroltests(test_variable, value):
    arg_lambda = 500
    h = 3
    weeks = 10
    max_t = 10
    chi = 0.1
    sd = 0.1
    k = 2

    if test_variable == "arg_lambda":
        arg_lambda = value
    elif test_variable == "h":
        h = value
    elif test_variable == "chi":
        chi = value
    elif test_variable == "sd":
        sd = value
    elif test_variable == "h":
        h = value
    elif test_variable == "k":
        k = value

    resultsArray = []
    for r in range(0, 30):
        pop = []
        for i in range(0, arg_lambda):
            pop.append( [generateRandomStrat(h), 0, 0, 0] ) #tuple of strat, currentstate, currentgoing, fitness

        crowded = 0

        
        for t in range(0, max_t):
            total_gen = 0
            #evaluate
            for i in range(0, weeks):
                #count how many go
                going = 0
                for member in pop:
                    sim = lStep(member[0], member[1], crowded)
                    if sim[1] == 1:
                        member[1] = sim[0]
                        member[2] = 1
                        going += 1
                    else:
                        member[1] = sim[0]
                        member[2] = 0
                        
                if going/arg_lambda >= 0.6:
                    crowded = 1
                else:
                    crowded = 0
                
                for member in pop:
                    #if member said they went and it was not crowded +1 to fitness
                    if member[2] == 1 and not crowded:
                        member[3] += 1
                    #similarly for not go and crowded
                    elif member[2] == 0 and crowded:
                        member[3] += 1

                total_gen += going
            avg_gen = (total_gen / (arg_lambda * weeks)) * 100
            
            #print("Gen: " + str(t) + " Average: " + str(avg_gen) +"%")
              
            #GA
            newpop = []
            for i in range(0, arg_lambda):
                x = tournSelection(pop,k)
                y = tournSelection(pop,k)
                mx = mutate(x, chi, sd)
                my = mutate(y, chi, sd)
                child = crossover(mx, my)
                newpop.append([child,0,0,0])
            pop = newpop
        #append the most recent result
        print(str(avg_gen))
        resultsArray.append(avg_gen) 

    return resultsArray
    
    #print(str(pop[0][0].p) + " " + str(pop[0][0].a) + " " + str(pop[0][0].b) )

def elfarolTestsAutomater():
    populationSizesArray = [10,50,100,200,300,400,500,600,700,800,900,1000]
    chiArray = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1]
    sdArray = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    hArray = [1,2,3,4,5,10,20,30]
    kArray = [1,2,3,4,5,6,7,8,9,10]

    data = []
    for value in kArray:
        data.append(elfaroltests("k", value))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(data, labels=kArray)
    ax.set_xlabel('Tournament Size')
    ax.set_ylabel('Average Attendance %')
    #plt.show()
    fig.savefig('kTests.png', bbox_inches='tight')
    
##    data = []
##    for value in populationSizesArray:
##        data.append(elfaroltests("arg_lambda", value))
##    fig = plt.figure()
##    ax = fig.add_subplot(111)
##    ax.boxplot(data, labels=populationSizesArray)
##    ax.set_xlabel('Population Size')
##    ax.set_ylabel('Average Attendance %')
##    #plt.show()
##    fig.savefig('popTests.png', bbox_inches='tight')
##
##
##    data = []
##    for value in chiArray:
##        data.append(elfaroltests("chi", value))
##    fig = plt.figure()
##    ax = fig.add_subplot(111)
##    ax.boxplot(data, labels=chiArray)
##    ax.set_xlabel('Chi')
##    ax.set_ylabel('Average Attendance %')
##    #plt.show()
##    fig.savefig('chiTests.png', bbox_inches='tight')
##
##    data = []
##    for value in sdArray:
##        data.append(elfaroltests("sd", value))
##    fig = plt.figure()
##    ax = fig.add_subplot(111)
##    ax.boxplot(data, labels=sdArray)
##    ax.set_xlabel('Standard Deviation')
##    ax.set_ylabel('Average Attendance %')
##    #plt.show()
##    fig.savefig('sdTests.png', bbox_inches='tight')
##
##    data = []
##    for value in hArray:
##        data.append(elfaroltests("h", value))
##    fig = plt.figure()
##    ax = fig.add_subplot(111)
##    ax.boxplot(data, labels=hArray)
##    ax.set_xlabel('Number of States')
##    ax.set_ylabel('Average Attendance %')
##    #plt.show()
##    fig.savefig('hTests.png', bbox_inches='tight')


##    with open("Niso3Pop.csv", 'w', newline='') as csvfile:
##        csvwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
##        for value in populationSizesArray:
##            print("Pop size: " + str(value))
##            csvwriter.writerow(elfaroltests("arg_lambda",value))
##
##    with open("Niso3chi.csv", 'w', newline='') as csvfile:
##        csvwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
##        for value in chiArray:
##            print("Chi: " + str(value))
##            csvwriter.writerow(elfaroltests("chi",value))
##
##    with open("Niso3sd.csv", 'w', newline='') as csvfile:
##        csvwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
##        for value in sdArray:
##            print("SD: " + str(value))
##            csvwriter.writerow(elfaroltests("sd",value))

##    with open("Niso3h.csv", 'w', newline='') as csvfile:
##        csvwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
##        for value in hArray:
##            print("H: " + str(value))
##            csvwriter.writerow(elfaroltests("h",value))



if args.question == 1:
    sample(args.prob, args.repetitions)
if args.question == 2:
    steprep(args.strategy, args.state, args.crowded, args.repetitions)
if args.question == 3:
    elfarol(args.lam, args.h, args.weeks, args.max_t)
    
    
                
        
    






