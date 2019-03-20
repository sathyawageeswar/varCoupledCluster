
# coding: utf-8

# In[ ]:


# Routines implementing the variational couple cluster
# method using the LCU method and FPOAA

import numpy as np                          # for rank_1_projector and other custom matrices
import math
import matplotlib.pyplot as plt
from collections import Counter

from projectq import MainEngine
from projectq.ops import H, Ry, Rx, X, Y, Z, R, Ph, All, Measure, ControlledGate
                                            # Ph for global phase
                                            # R is for phase gates
from projectq.ops._basics import SelfInverseGate
                                            # because they don't have a named identity
        
from projectq.meta import Dagger, Compute, Uncompute, Control

from projectq.ops import QubitOperator
# from fermilib.ops import FermionOperator
# from fermilib.transforms import jordan_wigner 
                                            # for Jordan-Wigner transform 
                                            # and fermionic Hamiltonians
class IdGate(SelfInverseGate):
    """ Identity gate class """
    def __str__(self):
        return "I"
    
    @property
    def matrix(self):
        return np.matrix([1, 0], [0,1])
    
#: Shortcut (instance of) :class:`IdGate`
I = Id = IdGate()

# the LCU 'V' map that prepares the control state
# take the list of 'm' coefts as input
# return the ceil(\log m) qubit register as output

def lcu_control_state_prep(eng, coefts, reg, dim):
    m = len(coefts)
    # dim = math.ceil(math.log(m,2))     
    size = pow(2, dim)
    
    probs = np.zeros(size)
    probs[0:m] = [abs(x) for x in coefts]
    weight = np.sum(probs)
    probs = (1.0/weight) * probs 
    # print("prob vec: {}".format(probs))
    
    # compute the rotation angles required for preparing
    # the LCU control state, apply conditional rotations
    # following the method of Grover & Rudolph (2002)
    
    # we need to perform bifurcations over dim rounds
    
    for i in range(1,dim+1):
        block_size = 2**(dim-i+1)
        # print("block_size: {}".format(block_size))
               
        # in each round, need to bifurcate 2^(round - 1) blocks
        num_blocks = 2**(i-1)
#        print("num_blocks: {}".format(num_blocks))
        target = np.zeros((2**i, 2**i))
        for j in range(0, num_blocks):
            # break loop if we've already crossed the last non-zero coefficient
            if (j*block_size > m-1):
#                 print("Broke from j={} because we crossed the last element\n".format(int(j)))
                break
            
            start = j*block_size
            end = start + block_size
#            print("start: {}, end: {}".format(int(start),int(end)))
            
            vec_j = probs[start : end]
#            print("block to be bifurcated: {}".format(vec_j))
            # break loop if singleton
            if (len(vec_j)<=1):
#                 print("Broke from j={} because we encountered a singleton\n".format(int(j)))
                break
            
            left_cond_prob_j = bifurcate(vec_j)
            # print("left cond prob of vec_j: {}".format(left_cond_prob_j))
            
            # perform a rotation with angle corresponding to sqrt of this prob
            f_j = math.sqrt(left_cond_prob_j)
            ang_j = math.acos(f_j)
            
            temp = np.binary_repr(j)
            temp = temp.zfill(i-1)        # padding with zeros to get fixed length bin
            with Compute(eng):
                for j in range(0,i-1):
                    if (int(temp[j])==0):
                        X | reg[j]
            
            with Control(eng, reg[0:i-1]):
                Ry(2*ang_j) | reg[i-1]
            Uncompute(eng)   
    return reg

# bifurcate vector into left and right 
# halves, and return the conditional
# probability of the left half
# will always expect len(vec) a power of 2
def bifurcate(vec):
    m = len(vec)
    tot = np.sum(vec)
    if (tot==0):
#         print("some segment has probability 0")
        return 0
    
    left = 0.0
    for i in range (0, m//2):          # // = int division op
        left += vec[i]
        
    return left/tot


# lu_controlled_unitary takes a 
# ctrl + sys register and returns
# it after applying the lcu unitary list
# to sys, controlled on ctrl

# include check to make sure that the 
# matrices supplied are unitary

def lcu_controlled_unitary(eng, list_of_U, coefts, ctrl, sys, sys_dim):
    size = len(list_of_U)
    ctrl_dim = math.ceil(math.log(size, 2))
    
    for i in range(0,size):
        temp = np.binary_repr(i)
        temp = temp.zfill(ctrl_dim)        # pad with zeros for fixed length bin
        
        with Compute(eng):
            for j in range(0,ctrl_dim):
                if (int(temp[j])==0):
                    X | ctrl[j]
        
        # if unitaries passed are qubitoperator, directly apply them, no unpacking required
        if isinstance(list_of_U[1], QubitOperator):
            with Control(eng, ctrl):
                list_of_U[i] | sys
            Uncompute(eng)       
        
        else:
            with Control(eng, ctrl):
                if (isinstance(coefts[i], complex)):   # can be i, or -1
                    Ph(0.5*math.pi) | sys[j]       # apply global phase i
                    if (np.sign(-1j*coefts[i])<0):
                        Ph(math.pi) | sys[j]       # global phase is actually -i
                elif (np.sign(coefts[i])<0):
                        Ph(math.pi) | sys[j]       # apply global phase -1
                for j in range(0, sys_dim):
                    if (list_of_U[i][j]==I):
                        continue
                    list_of_U[i][j] | sys[j]
            Uncompute(eng)
        
def lcu(eng, list_of_unitaries, coefts, ctrl, sys, ctrl_dim, sys_dim):
        with Compute(eng):
            lcu_control_state_prep(eng, coefts, ctrl, ctrl_dim)
        lcu_controlled_unitary(eng, list_of_unitaries, coefts, ctrl, sys, sys_dim)
        Uncompute(eng)

def postselect(ctrl, ctrl_dim):
    for idx in range(0, ctrl_dim):
        Measure | ctrl[idx]
        if (bool(ctrl[idx])):
            return 0
    return 1

# calculate lcu success probability
def lcu_success_prob(eng, reg, sys_dim, dim):
    eng.flush()
    prob = 0
    sysmax = pow(2, sys_dim)
    for i in range(0, sysmax):
        bin_i = np.binary_repr(i)
        bin_i = bin_i.zfill(dim)        # padding with zeros to get fixed length bin
        prob += eng.backend.get_probability(bin_i, reg)
    return prob

# conditional phase for amplitude amplification
def cond_phase(eng, ctrl, sys, phase):
    with Compute(eng):
        All(X) | ctrl
    with Control(eng, ctrl):
        Ph(phase) | sys[0]               # give global phase to any one sys qubit
    Uncompute(eng)

# oblivious amplitude amplification for lcu
# iterative algorithm
# __TO_DO__ : write more general oaa method
def lcu_oaa(eng, list_of_unitaries, coefts, ctrl, sys, ctrl_dim, sys_dim, rounds=1):
    phi = -1*math.pi
    for i in range(0, rounds):
        cond_phase(eng, ctrl, sys, phi)
        with Dagger(eng):
            lcu(eng, list_of_unitaries, coefts, ctrl, sys, ctrl_dim, sys_dim)
        size = pow(2, ctrl_dim)
        for l in range(1,size):                # -R flips sign of everything except 00..0
            temp = np.binary_repr(i)
            temp = temp.zfill(ctrl_dim)        # pad with zeros for fixed length bin
        with Compute(eng):
            for j in range(0,ctrl_dim):
                if (int(temp[j])==0):
                    X | ctrl[j]
        with Control(eng, ctrl):
            Ph(phi) | sys[0]                   # flip sign using any one sys qubit
        Uncompute(eng)
        lcu(eng, list_of_unitaries, coefts, ctrl, sys, ctrl_dim, sys_dim)
        print("Amplitudes of ctrl+sys state after {} rounds of OAA:\n".format(int(i)+1))
        print_amplitudes(eng, ctrl+sys, ctrl_dim+sys_dim)

# Fixed Point Oblivious Amplitude Amplification (FPOAA) for lcu
# unitary input_map prepares some target state with success prob 1-p
# Recursive algorithm, takes recursion depth as input
# __TO_DO__ : write more general FPOAA method
def lcu_fpoaa(eng, list_of_unitaries, coefts, ctrl, sys, ctrl_dim, sys_dim, depth):
    phi = math.pi/3.0
    gate_seq = fpoaa_string(depth)
    t = len(gate_seq)
    # the rightmost operator is always W
    # which has already been applied above in the lcu step
    for i in range(1,t):
        if (gate_seq[t-1-i]=='W'):
            lcu(eng, list_of_unitaries, coefts, ctrl, sys, ctrl_dim, sys_dim)
        elif (gate_seq[t-1-i]=='M'):
            with Dagger(eng):
                lcu(eng, list_of_unitaries, coefts, ctrl, sys, ctrl_dim, sys_dim)
        elif (gate_seq[t-1-i]=='R'):
            cond_phase(eng, ctrl, sys, phi)
        elif (gate_seq[t-1-i]=='S'):
            cond_phase(eng, ctrl, sys, -1*phi)

# classical hacks
# some string helper functions for FPOAA
# fpoaa_string computes as a classical 
# string the sequence of operators to be
# applied for depth n FPOAA 
# string_dagger takes a string representing
# an fpoaa block and returns its dagger
def fpoaa_string(depth):
    if(depth==1):
        return "WRMRW"
    else:
        V = fpoaa_string(depth-1)
        U = dagger(V)
        return V + 'R' + ''.join(U) + 'R' + V
    
def dagger(source):
    string = list(source)
    t = len(string)
    temp = [None] * t
    
    for i in range(0, t):
        if (string[t-1-i]=='W'):
             temp[i] = 'M'
        elif (string[t-1-i]=='M'):
             temp[i] = 'W'
        elif (string[t-1-i]=='R'):
             temp[i] = 'S'
        elif (string[t-1-i]=='S'):
            temp[i] = 'R'
    return temp


# code snippet for backend.cheat
def print_wavefunction(eng):
    eng.flush()
    print("The net backend state is \n")
    print(eng.backend.cheat())
    
# print bitstring probability amplitudes
def print_amplitudes(eng, reg, dim):
    eng.flush()
    maxout = pow(2, dim)
    print("amplitude of the basis state:")
    for i in range(0, maxout):
        bin_i = np.binary_repr(i)
        bin_i = bin_i.zfill(dim)        # padding with zeros to get fixed length bin
        temp = eng.backend.get_amplitude(bin_i, reg)
        if (temp!=0):
            print("\t|{}> (=|{}>) is {}".format(bin_i, int(i), temp))

# print bitstring probabilities
def print_probabilities(eng, reg, dim):
    eng.flush()
    maxout = pow(2, dim)
    print("probability of the string:")
    for i in range(0, maxout):
        bin_i = np.binary_repr(i)
        bin_i = bin_i.zfill(dim)        # padding with zeros to get fixed length bin
        temp = eng.backend.get_probability(bin_i, reg)
        print("\t{} (={}) is {}".format(bin_i, int(i), temp))


# print output string after measurement
def print_mmt_output(reg, dim):
    eng.flush()
    # make sure mmt has been made: ideally use some flag
    # Just in case, mmt now - if already mmed, this causes no difference
    All(measure) | reg
    for i in range(0,dim):
        print("{}".format(int(reg[dim-1-i])), end=' ')

