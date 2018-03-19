import random
import argparse
import csv
import math
import signal
#import matplotlib.pyplot as plt
#import numpy as np
from string import whitespace

#import numpy as np



parser = argparse.ArgumentParser(description="Lab4")
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-question", type=int)
parser.add_argument("-n", type=int)
parser.add_argument("-x")
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

global b
b = 10

class Node:
    def __init__(self, data, val1=None, val2=None, val3=None, val4=None):
        self.data = data
        self.val1 = val1
        self.val2 = val2
        self.val3 = val3
        self.val4 = val4

    def __str__(self):
        if str(self.val3) != "None":
            return '(ifleq ' + str(self.val1) + ' ' + str(self.val2) + ' '+ str(self.val3) + ' '+ str(self.val4) + ')'
        if str(self.val2) != "None":
            return '(' + str(self.data) + ' ' + str(self.val1) + ' ' + str(self.val2) + ')'
        if str(self.val1) != "None":
            return '(' + str(self.data) + ' ' + str(self.val1) + ')'
        if str(self.data) != "None":
            return str(self.data)
        else:
            return None

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
        

### From https://gist.github.com/pib/240957
### if the type is a tuple, then it is an operataion
atom_end = set('()"\'') | set(whitespace)
def parse(sexp):
    stack, i, length = [[]], 0, len(sexp)
    while i < length:
        c = sexp[i]

        #print(c, stack)
        reading = type(stack[-1])
        if reading == list:
            if   c == '(': stack.append([])
            elif c == ')': 
                stack[-2].append(stack.pop())
                if stack[-1][0] == ('quote',): stack[-2].append(stack.pop())
            elif c == '"': stack.append('')
            elif c == "'": stack.append([('quote',)])
            elif c in whitespace: pass
            else: stack.append((c,))
        elif reading == str:
            if   c == '"': 
                stack[-2].append(stack.pop())
                if stack[-1][0] == ('quote',): stack[-2].append(stack.pop())
            elif c == '\\': 
                i += 1
                stack[-1] += sexp[i]
            else: stack[-1] += c
        elif reading == tuple:
            if c in atom_end:
                atom = stack.pop()
                if atom[0][0].isdigit(): stack[-1].append(eval(atom[0]))
                else: stack[-1].append(atom)
                if stack[-1][0] == ('quote',): stack[-2].append(stack.pop())
                continue
            else: stack[-1] = ((stack[-1][0] + c),)
        i += 1
    return stack.pop()

def evaluate(expr, n, x):
    #used for q1 (expr given in string)
    if type(expr) == str:
        lexpr = parse(expr)[0]
        tree = lexprToTree(lexpr)
    else:
        tree = expr
        
    if type(x) != list:
        lx = x.split(" ")
    else:
        lx = x
        
    #return receval(lexpr, n, lx)
    return nodeEval(tree, n, lx)



##def receval(lexpr, n, lx):
##    if type(lexpr) != tuple and type(lexpr) != list:
##        return float(lexpr)
##
##    if type(lexpr[0]) == tuple: #then it is an operation
##        if lexpr[0][0] == "add":
##            return float(receval(lexpr[1], n, lx)) + float(receval(lexpr[2], n, lx))
##        elif lexpr[0][0] == "sub":
##            return float(receval(lexpr[1], n, lx)) - float(receval(lexpr[2], n, lx))
##        elif lexpr[0][0] == "mul":
##            return float(receval(lexpr[1], n, lx)) * float(receval(lexpr[2], n, lx))
##        elif lexpr[0][0] == "div":
##            value2 = float(receval(lexpr[2], n, lx))
##            if value2 > 0:
##                return float( receval(lexpr[1], n, lx) / value2)
##            else:
##                return 0
##        elif lexpr[0][0] == "pow":
##            value = float(receval(lexpr[1], n, lx))
##            power = float(receval(lexpr[2], n, lx))
##            if value <= 0: #negative values can be raised by real numbers but this can sometimes lead to imaginary numbers so we'll ignore them here
##                return 0
##            else:
##                try:
##                    return  value ** power
##                except OverflowError:
##                    return 0
##        elif lexpr[0][0] == "sqrt":
##            value = float( receval(lexpr[1], n, lx) )
##            if value > 0:
##                return math.sqrt(value)
##            else:
##                return 0
##        elif lexpr[0][0] == "log":
##            value = float( receval(lexpr[1], n, lx) )
##            if value > 0:
##                return math.log(value, 2)
##            else:
##                return 0
##        elif lexpr[0][0] == "exp":
##            try:
##                value =  math.exp( float(receval(lexpr[1], n, lx)) )
##            except OverflowError:
##                value = 0
##            return value
##        elif lexpr[0][0] == "max":
##            return max(float(receval(lexpr[1], n, lx)) , float(receval(lexpr[2], n, lx)))
##        elif lexpr[0][0] == "ifleq":
##            if float(receval(lexpr[1], n, lx)) <= float(receval(lexpr[2], n, lx)):
##                return float(receval(lexpr[3], n, lx))
##            else:
##                return float(receval(lexpr[4], n, lx))
##        elif lexpr[0][0] == "data":
##            j = math.floor(float(receval(lexpr[1], n, lx))) % n
##            return float(lx[j])
##        elif lexpr[0][0] == "diff":
##            k = math.floor(float(receval(lexpr[1], n, lx))) % n
##            l = math.floor(float(receval(lexpr[2], n, lx))) % n
##            return float(lx[k]) - float(lx[l])
##        elif lexpr[0][0] == "avg":
##            k = math.floor(float(receval(lexpr[1], n, lx))) % n
##            l = math.floor(float(receval(lexpr[2], n, lx))) % n
##            higher = max(k,l) - 1
##            lower = min(k,l)
##            total = 0
##            for i in range(lower,higher):
##                total += float(lx[i])
##            if (higher-lower) > 0:
##                return total/(higher-lower)
##            else:
##                return 0

def nodeEval(nexpr, n, lx):
    

    if nexpr.data == "add":
            return float(nodeEval(nexpr.val1, n, lx)) + float(nodeEval(nexpr.val2, n, lx))
    elif nexpr.data == "sub":
            return float(nodeEval(nexpr.val1, n, lx)) - float(nodeEval(nexpr.val2, n, lx))
    elif nexpr.data == "mul":
            return float(nodeEval(nexpr.val1, n, lx)) * float(nodeEval(nexpr.val2, n, lx))
    elif nexpr.data == "div":
            value2 = float(nodeEval(nexpr.val2, n, lx))
            if value2 > 0:
                    return float( nodeEval(nexpr.val1, n, lx) / value2)
            else:
                    return 0
    elif nexpr.data == "pow":
            value = float(nodeEval(nexpr.val1, n, lx))
            power = float(nodeEval(nexpr.val2, n, lx))
            if value <= 0: #negative values can be raised by real numbers but this can sometimes lead to imaginary numbers so we'll ignore them here
                    return 0
            else:
                    try:
                            return  value ** power
                    except OverflowError:
                            return 0
    elif nexpr.data == "sqrt":
            value = float( nodeEval(nexpr.val1, n, lx) )
            if value > 0:
                    return math.sqrt(value)
            else:
                    return 0
    elif nexpr.data == "log":
            value = float( nodeEval(nexpr.val1, n, lx) )
            if value > 0:
                    return math.log(value, 2)
            else:
                    return 0
    elif nexpr.data == "exp":
            try:
                    value =  math.exp( float(nodeEval(nexpr.val1, n, lx)) )
            except OverflowError:
                    value = 0
            return value
    elif nexpr.data == "max":
            return max(float(nodeEval(nexpr.val1, n, lx)) , float(nodeEval(nexpr.val2, n, lx)))
    elif nexpr.data == "ifleq":
            if float(nodeEval(nexpr.val1, n, lx)) <= float(nodeEval(nexpr.val2, n, lx)):
                    return float(nodeEval(nexpr.val3, n, lx))
            else:
                    return float(nodeEval(nexpr.val4, n, lx))
    elif nexpr.data == "data":
            j = math.floor(float(nodeEval(nexpr.val1, n, lx))) % n
            return float(lx[j])
    elif nexpr.data == "diff":
            k = math.floor(float(nodeEval(nexpr.val1, n, lx))) % n
            l = math.floor(float(nodeEval(nexpr.val2, n, lx))) % n
            return float(lx[k]) - float(lx[l])
    elif nexpr.data == "avg":
            k = math.floor(float(nodeEval(nexpr.val1, n, lx))) % n
            l = math.floor(float(nodeEval(nexpr.val2, n, lx))) % n
            higher = max(k,l) - 1
            lower = min(k,l)
            total = 0
            for i in range(lower,higher):
                    total += float(lx[i])
            if (higher-lower) > 0:
                    return total/(higher-lower)
            else:
                    return 0
    else:
        return float(nexpr.data)


#transform a parsed expression to a tree
def lexprToTree(lexpr):
    if type(lexpr) != tuple and type(lexpr) != list:
        return Node(float(lexpr))

    if type(lexpr[0]) == tuple: #then it is an operation
        if lexpr[0][0] == "add":
            return Node("add", lexprToTree(lexpr[1]), lexprToTree(lexpr[2]))
        elif lexpr[0][0] == "sub":
            return Node("sub", lexprToTree(lexpr[1]), lexprToTree(lexpr[2]))
        elif lexpr[0][0] == "mul":
            return Node("mul", lexprToTree(lexpr[1]), lexprToTree(lexpr[2]))
        elif lexpr[0][0] == "div":
            return Node("div", lexprToTree(lexpr[1]), lexprToTree(lexpr[2]))
        elif lexpr[0][0] == "pow":
            return Node("pow", lexprToTree(lexpr[1]), lexprToTree(lexpr[2]))
        elif lexpr[0][0] == "sqrt":
            return Node("sqrt", lexprToTree(lexpr[1]))
        elif lexpr[0][0] == "log":
            return Node("log", lexprToTree(lexpr[1]))
        elif lexpr[0][0] == "exp":
            return Node("exp", lexprToTree(lexpr[1]))
        elif lexpr[0][0] == "max":
            return Node("max", lexprToTree(lexpr[1]), lexprToTree(lexpr[2]))
        elif lexpr[0][0] == "ifleq":
            return Node("ifleq", lexprToTree(lexpr[1]), lexprToTree(lexpr[2]), lexprToTree(lexpr[3]), lexprToTree(lexpr[4]) )
        elif lexpr[0][0] == "data":
            return Node("data", lexprToTree(lexpr[1]))
        elif lexpr[0][0] == "diff":
            return Node("diff", lexprToTree(lexpr[1]), lexprToTree(lexpr[2]))
        elif lexpr[0][0] == "avg":
            return Node("avg", lexprToTree(lexpr[1]), lexprToTree(lexpr[2]))


def fitness(expr, n, m, data):
    rawfile = open(data, "r")
    file = rawfile.read()
    filelines = file.splitlines()

    totalDiff = 0
    for line in filelines:
        lline = line.split("\t")
        val = evaluate(expr, n, lline[0:-1])
        #overflow error when evaluation value is too high                
        try:
            diff = ( float(lline[-1]) - val )**2
        except OverflowError:
            diff = 999
            
        totalDiff += diff

    return totalDiff/m


        

def randomBalancedTree(treeDepth):
    if treeDepth != 0:
        random_case = random.randint(0,12)
        if random_case == 0:
            return Node("add", randomBalancedTree(treeDepth-1), randomBalancedTree(treeDepth-1))
        if random_case == 1:
            return Node("sub", randomBalancedTree(treeDepth-1), randomBalancedTree(treeDepth-1))
        if random_case == 2:
            return Node("mul", randomBalancedTree(treeDepth-1), randomBalancedTree(treeDepth-1))
        if random_case == 3:
            return Node("div", randomBalancedTree(treeDepth-1), randomBalancedTree(treeDepth-1))
        if random_case == 4:
            return Node("pow", randomBalancedTree(treeDepth-1), randomBalancedTree(treeDepth-1))
        if random_case == 5:
            return Node("sqrt", randomBalancedTree(treeDepth-1))
        if random_case == 6:
            return Node("log", randomBalancedTree(treeDepth-1))
        if random_case == 7:
            return Node("exp", randomBalancedTree(treeDepth-1))
        if random_case == 8:
            return Node("max", randomBalancedTree(treeDepth-1), randomBalancedTree(treeDepth-1))
        if random_case == 9:
            return Node("ifleq", randomBalancedTree(treeDepth-1), randomBalancedTree(treeDepth-1), randomBalancedTree(treeDepth-1), randomBalancedTree(treeDepth-1))
        if random_case == 10:
            return Node("data", randomBalancedTree(treeDepth-1))
        if random_case == 11:
            return Node("diff", randomBalancedTree(treeDepth-1), randomBalancedTree(treeDepth-1))
        if random_case == 12:
            return Node("avg", randomBalancedTree(treeDepth-1), randomBalancedTree(treeDepth-1))
    else:
        return Node(random.randint(0,b))    
    
def randomTree(maxDepth):
    if maxDepth != 0:
        random_case = random.randint(0,13)
        if random_case == 0:
            return Node("add", randomTree(maxDepth-1), randomTree(maxDepth-1))
        if random_case == 1:
            return Node("sub", randomTree(maxDepth-1), randomTree(maxDepth-1))
        if random_case == 2:
            return Node("mul", randomTree(maxDepth-1), randomTree(maxDepth-1))
        if random_case == 3:
            return Node("div", randomTree(maxDepth-1), randomTree(maxDepth-1))
        if random_case == 4:
            return Node("pow", randomTree(maxDepth-1), randomTree(maxDepth-1))
        if random_case == 5:
            return Node("sqrt", randomTree(maxDepth-1))
        if random_case == 6:
            return Node("log", randomTree(maxDepth-1))
        if random_case == 7:
            return Node("exp", randomTree(maxDepth-1))
        if random_case == 8:
            return Node("max", randomTree(maxDepth-1), randomTree(maxDepth-1))
        if random_case == 9:
            return Node("ifleq", randomTree(maxDepth-1), randomTree(maxDepth-1), randomTree(maxDepth-1), randomTree(maxDepth-1))
        if random_case == 10:
            return Node("data", randomTree(maxDepth-1))
        if random_case == 11:
            return Node("diff", randomTree(maxDepth-1), randomTree(maxDepth-1))
        if random_case == 12:
            return Node("avg", randomTree(maxDepth-1), randomTree(maxDepth-1))
        if random_case == 13:
            return Node(random.randint(0,b))
    else:
        return Node(random.randint(0,b))    

def recTraveseTree(tree, branches):
    branches.append(tree)
    if (str(tree.val3) != "None"):
        recTraveseTree(tree.val1, branches)+recTraveseTree(tree.val2, branches)+recTraveseTree(tree.val3, branches)+recTraveseTree(tree.val4, branches)
    if str(tree.val1) != "None" and str(tree.val2) != "None":
        recTraveseTree(tree.val1, branches) + recTraveseTree(tree.val2, branches)
    if str(tree.val1) != "None" and str(tree.val2) == "None":
        recTraveseTree(tree.val1, branches)
    
    if str(tree.data) != "None":
        return branches    

def replace(tree, oldBranch, newBranch):
    sTree = str(tree)
    sOld = str(oldBranch)
    sNew = str(newBranch)
    #convert to string to replace
    sReplace = sTree.replace(sOld, sNew,1)
    #then turn it back into a tree
    lTree = parse(sReplace)
    newTree = lexprToTree(lTree[0])
    return newTree
        

def crossover(t1, t2):
    lt1 = recTraveseTree(t1, [])
    lt2 = recTraveseTree(t2, [])

    if len(lt1) > 1:
        randomT1branch = lt1[random.randint(1,(len(lt1)-1))]
    else:
        return t2 #can't crossover 
    if len(lt2) > 1:
        randomT2branch = lt2[random.randint(1,(len(lt2)-1))]
    else:
        return t1 #can't crossover

    newT1 = replace(t1, randomT1branch, randomT2branch)
    newT2 = replace(t2, randomT2branch, randomT1branch)

    if random.random() < 0.5:
        return newT1
    else:
        return newT2

def tournSelection(poplist, k):
    winner = None
    lowest = float("inf")
    for i in range(0,k):
        member = poplist[random.randint(0,len(poplist)-1)]
        if float(member[1]) <= float(lowest):
            winner = member[0]
            lowest = member[1]

    #if all random members didn't get below default lowest value, return a random member as winner
    if type(winner) == None:
        return poplist[random.randint(0,len(poplist)-1)][0]
    else:
        return winner


def mutate(tree, chi, newTreeMaxDepth):
    if random.random() < chi:
        lTree = recTraveseTree(tree, [])
        randomBranch = lTree[random.randint(0,(len(lTree)-1))]
        newTree = replace(tree, randomBranch, randomTree(newTreeMaxDepth))
        return newTree
    else:
        return tree
    
def ga(arg_lambda, n, m, data, time_budget):
    #n is dimension of input vector
    #m is how many lines of training data there is
    #need global variable here for generate randomBalancedTree function
    global b
    b = n
    
    treeDepth = 6
    k = 2
    chi = 0
    mutateTreeMaxDepth = 5
    
    #start timer
    timeup = False
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(time_budget)
    try:
        pop = []
        t = 0
        #ramped half and half initial population
        partitions = treeDepth - 1
        perSection = math.floor(arg_lambda / partitions)
        for i in range(2,treeDepth):
            for j in range(0,perSection):
                pop.append([randomBalancedTree(i),0])
                pop.append([randomTree(i),1])
        if len(pop) < arg_lambda: #add more to make up to lambda if any are missing
            for i in range(len(pop), arg_lambda):
                pop.append([randomTree(treeDepth),0])

    ##    for i in pop:
    ##        print(i[0])
        
        lowest = 99999999
        xbest = pop[0][0]
        #evaluate fitness
        for member in pop:
            score = fitness(member[0], n, m, data)
            member[1] = score
            if score <= lowest:
                xbest = member
                lowest = score

        while not timeup:
            #GA
            newpop = []
            for i in range(0, arg_lambda):
                x = tournSelection(pop,k)
                y = tournSelection(pop,k)
                mx = mutate(x, chi, mutateTreeMaxDepth)
                my = mutate(y, chi, mutateTreeMaxDepth)
                child = crossover(mx, my)
                childScore = fitness(child, n, m, data)
                if childScore <= lowest:
                    xbest = child
                    lowest = childScore
                newpop.append([child,childScore])
            pop = newpop
            t += 1
            
    except StopIteration:
        timeup = True
        print(xbest)
    
def signal_handler(signum, frame):
    raise StopIteration("Time up")

                                                   
def test():
    a = str(randomTree(5))
    print(a)
    print(str(evaluate(a, 10, "1 2 3 4 5 6 7 8 9 10")))

def test2():
    a = randomTree(2)
    b = randomTree(2)
    print(a)
    print(b)
    print(crossover(a,b)[0])
    print(crossover(a,b)[1])

if args.question == 1:
    evaluate(args.expr, args.n, args.x)
if args.question == 2:
    steprep(args.strategy, args.state, args.crowded, args.repetitions)
if args.question == 3:
    elfarol(args.lam, args.h, args.weeks, args.max_t)
