
# coding: utf-8

# In[ ]:


import _vcc_lcu

# testing root2 * H = X + Z
def test_hadamard1(eng, rounds=1):
    root2 = 1.0/math.sqrt(2)
    
    # H is a good test because X + Z = root2 * H
    
    sys_dim = 1
    A = [[X], [Z]]      
    coefts = [1/root2, 1/root2]
    
    m = len(coefts)
    ctrl_dim = math.ceil(math.log(m,2))
    dim = sys_dim + ctrl_dim
   
    num_1=0
    for i in range(0,rounds):
        ctrl = eng.allocate_qureg(ctrl_dim)
        sys = eng.allocate_qureg(sys_dim)
        lcu(eng, A, coefts, ctrl, sys, ctrl_dim, sys_dim)
                
        success = postselect(ctrl, ctrl_dim)
        if (success):
            H | sys[0]     # try to uncompute the H applied to qubit 1
            Measure | sys[0]
            num_1 += int(sys[0])
        else:
            All(Measure) | ctrl + sys

        eng.flush()
#        print("val={}".format(int(anc[0])))

    All(Measure) | ctrl + sys
    eng.flush()
    print("num of 1 = {}".format(num_1))


# testing 2-qbit hadamard 2HH=(X+Z)(X+Z)
def test_hadamard2(eng, rounds=1, oaa_rounds=0, fpoaa_depth=0):
    sys_dim = 2
       
    # newer versions of ProjectQ overload 
    # '|' for Pauli strings 
    A = [[X,X], [X,Z], [Z,X], [Z,Z]]
    coefts = [0.5, 0.5, 0.5, 0.5]
    
    m = len(coefts)
    ctrl_dim = math.ceil(math.log(m,2))
    dim = sys_dim + ctrl_dim
   
    num_1=0
    num = np.zeros(pow(2,sys_dim))
    for i in range(0,rounds):
        ctrl = eng.allocate_qureg(ctrl_dim)
        sys = eng.allocate_qureg(sys_dim)
        lcu(eng, A, coefts, ctrl, sys, ctrl_dim, sys_dim)
        
        print("Amplitudes of ctrl+sys state after lcu:\n")
        print_amplitudes(eng, ctrl+sys, dim)
        
        # testing amplitude amplification
        # case 1) FPOAA
        if (fpoaa_depth > 0):
            # theoretical success probability after fpoaa_depth
            t = lcu_success_prob(eng, ctrl+sys, sys_dim, dim)
            theory_prob = 1 - pow(1-t, pow(3, fpoaa_depth))
        
            lcu_fpoaa(eng, A, coefts, ctrl, sys, ctrl_dim, sys_dim, fpoaa_depth)
    
            prob = lcu_success_prob(eng, ctrl+sys, sys_dim, dim)
            print("lcu success prob after FPOAA({}) = {}\n".format(int(fpoaa_depth), float(prob)))
            print("discrepancy: prediction - practice = {}".format(float(theory_prob - prob)))

        # case 2) OAA
        # repeat S = -(W R M R) oaa-1 times
        elif (oaa_rounds > 0):
            lcu_oaa(eng, A, coefts, ctrl, sys, ctrl_dim, sys_dim, oaa_rounds)
            print("Amplitudes of ctrl+sys state after lcu & OAA({}):\n".format(int(oaa_rounds)))
            print_amplitudes(eng, ctrl+sys, dim)

        success = postselect(ctrl, ctrl_dim)
        if (success):
#             All(H) | sys     # try to uncompute the HH applied to sys
            print("Amplitudes of ctrl+sys state after postselection:\n")
            print_amplitudes(eng, ctrl+sys, dim)
            All(Measure) | sys
           
            # repeating and measuring frequencies
            if (int(sys[0])==0):
                if (int(sys[1])==0):
                    num[0] += 1
                else:
                    num[1] = num[1] + 1
            else:
                if (int(sys[1])==0):
                    num[2] += 1
                else:
                    num[3] += 1
            num_1 += int(sys[0]) + int(sys[1])
        else:
            All(Measure) | ctrl + sys
        eng.flush()

    print("frequencies: ")
    for i in range(len(num)):
        print("{} ".format(int(num[i])))
    All(Measure) | ctrl + sys
    eng.flush()
    print("num of 1 = {}".format(num_1))


# one possibility: a family of 2-qubit unitaries
# controlled phase
# CP(1,2) = 0.5 * (\id + Z1 + Z2 - Z1 Z2 )
def test_controlled_phase_12(eng, oaa_rounds = 0, fpoaa_depth = 0):
    sys_dim = 2
       
    # newer versions of ProjectQ overload 
    # '|' for Pauli strings 
    list_of_unitaries = [[I,I], [Z,I], [I,Z], [Z,Z]]
    coefts = [0.5, 0.5, 0.5, -0.5]
    
    m = len(coefts)
    ctrl_dim = math.ceil(math.log(m,2))
    dim = sys_dim + ctrl_dim
    
    ctrl = eng.allocate_qureg(ctrl_dim)
    sys = eng.allocate_qureg(sys_dim)
    
    # initialise sys state to the unif superposition
    All(H) | sys
    
    lcu(eng, list_of_unitaries, coefts, ctrl, sys, ctrl_dim, sys_dim)
    
    print("Amplitudes of ctrl+sys state after lcu:\n")
    print_amplitudes(eng, ctrl+sys, dim)
    
    prob = lcu_success_prob(eng, ctrl+sys, sys_dim, dim)
    print("lcu success prob = {}\n".format(float(prob)))
    
    # testing amplitude amplification
    # case 1) FPOAA
    if (fpoaa_depth > 0):
        # theoretical success probability after fpoaa_depth
        t = lcu_success_prob(eng, ctrl+sys, sys_dim, dim)
        theory_prob = 1 - pow(1-t, pow(3, fpoaa_depth))
        
        lcu_fpoaa(eng, list_of_unitaries, coefts, ctrl, sys, ctrl_dim, sys_dim, fpoaa_depth)
        
        prob = lcu_success_prob(eng, ctrl+sys, sys_dim, dim)
        print("lcu success prob after FPOAA({}) = {}\n".format(int(fpoaa_depth), float(prob)))
        print("theoretical lcu success prob after FPOAA({}) = {}\n".format(int(fpoaa_depth), float(theory_prob)))
        print("discrepancy: prediction - practice = {}".format(float(theory_prob - prob)))
    
    
    # case 2) OAA
    # repeat S = -(W R M R) oaa-1 times
    elif (oaa_rounds > 0):
        lcu_oaa(eng, list_of_unitaries, coefts, ctrl, sys, ctrl_dim, sys_dim, oaa_rounds)
        print("Amplitudes of ctrl+sys state after lcu & OAA({}):\n".format(int(oaa_rounds)))
        print_amplitudes(eng, ctrl+sys, dim)
    
    success = postselect(ctrl, ctrl_dim)
    if (success):
#       ControlledGate(R(math.pi), 1) | (sys[0], sys[1])     # try to uncompute the CPhase applied to sys
#       All(H) | sys                                         # uncompute the initial sys state
        print("Amplitudes of ctrl+sys state after postselection:\n")
        print_amplitudes(eng, ctrl+sys, dim)
        All(Measure) | sys
    else:
        All(Measure) | ctrl + sys
    eng.flush()

    All(Measure) | ctrl + sys
    eng.flush()
    
# Testing a single 1+a^{\dagger}a term
# m=4 orbitals, n=2 electrons, pick a_3^{\dagger}a_1
# CCS = Coupled Cluster Singles
def test_CCS(eng, t, oaa_rounds = 0, fpoaa_depth = 0):
    sys_dim = 4
       
    list_of_unitaries = [[X,Z,X,I], [X,Z,Y,I], [Y,Z,X,I], [Y,Z,Y,I],[I,I,I,I]]
    # The coefficients are complex - need some Y rotations to deal with this
    # Please write up newer state preparation routines
    # coefficients depend on the cluster amplitude t  
    coefts = [0.25, -0.25j, 0.25j, 0.25]
    coefts = list(map(lambda x: x*t, coefts))
    coefts.append(1.0)
    
    m = len(coefts)
    ctrl_dim = math.ceil(math.log(m,2))
    dim = sys_dim + ctrl_dim
    
    ctrl = eng.allocate_qureg(ctrl_dim)
    sys = eng.allocate_qureg(sys_dim)
    
    # sys state initialised to the reference (Slater) determinant
    # Since we are in second quantisation, JW means this is simply an all zeroes state
    # Hence the actual target state is then |0000> + 0.5*t|1010>, properly normalised
    # So we can compare the amplitudes
    lcu(eng, list_of_unitaries, coefts, ctrl, sys, ctrl_dim, sys_dim)
    
#    print("Amplitudes of ctrl+sys state after lcu:\n")
#    print_amplitudes(eng, ctrl+sys, dim)
    
    prob = lcu_success_prob(eng, ctrl+sys, sys_dim, dim)
#    print("lcu success prob = {}\n".format(float(prob)))
    
    # testing amplitude amplification
    # case 1) FPOAA
    theory_prob=practice_prob=prob
    if (fpoaa_depth > 0):
        # theoretical success probability after fpoaa_depth
        temp = lcu_success_prob(eng, ctrl+sys, sys_dim, dim)
        theory_prob = float(1 - pow(1-temp, pow(3, fpoaa_depth)))
        
        lcu_fpoaa(eng, list_of_unitaries, coefts, ctrl, sys, ctrl_dim, sys_dim, fpoaa_depth)
        
        practice_prob = lcu_success_prob(eng, ctrl+sys, sys_dim, dim)
#         print("lcu success prob after FPOAA({}) = {}\n".format(int(fpoaa_depth), float(practice_prob)))
#         print("theoretical lcu success prob after FPOAA({}) = {}\n".format(int(fpoaa_depth), float(theory_prob)))
#         print("discrepancy: prediction - practice = {}".format(float(theory_prob - practice_prob)))
     
    success = postselect(ctrl, ctrl_dim)
    if (success):
#         print("\n Amplitudes of ctrl+sys state after postselection (FPOAA depth = {}):".format(int(fpoaa_depth)))
#         print_amplitudes(eng, ctrl+sys, dim)
        # Save amplitudes to a file: col 1 for |0000>, col 2 for |1010>
        amp1 = eng.backend.get_amplitude('0000000', ctrl+sys)
        amp2 = eng.backend.get_amplitude('0001010', ctrl+sys)
        f = open("CCS_amp_%s_data.txt" % t, "a")
        f.write("{}\t{}\n".format(complex(amp1),complex(amp2)))
        f.close()
        All(Measure) | sys
    else:
        All(Measure) | ctrl + sys
    eng.flush()

    All(Measure) | ctrl + sys
    eng.flush()
    
    return theory_prob, practice_prob

