# Written by Micah Nichols (mwn45@msstate.edu)
# Python code using SciPy minimize Nelder-Mead (simplex) method to optimize parameters for RANN potential fitting
#
# Files needed - RANN calibration input script, run.sh (script to run calibration on temp.input), RANN calibration input script to write best fit
#
# ---------- Instructions ----------
# Change input_file to RANN input script filename
# Note ---- This code will run much faster by only using radial fingerprints for each element and setting all layer sizes to 1
#           (change o and n to 1)
# Ensure max epochs is set to 1
# Ensure validation is set to 0

# Change write_file to RANN input script you wish to write final values
# Change parameters to be fit to True, Change parameters not to be fit to False
# Change bounds for parameters to be fit (0 can cause an error - use something like 1e-10)
# -----------------------------------

from scipy.optimize import minimize
import numpy as np
import os

#--------- USER INPUT ----------
# FULL INPUT FILE NAME
input_file = 'y.input'
# FULL INPUT FILE TO WRITE FOR USE
write_file = 'y2.input'

# PARAMETERS TO FIT - True for yes, False for no
# Single Element Parameters
eshift_1 = True
ec_1 = True
re_1 = False
rc_1 = False
alpha_1 = False
delta_1 = False
dr_1 = False
cweight_1 = False
beta_1 = False
Asub_1 = False
Cmax_1 = False
Cmin_1 = False

# Two Element Parameters
eshift_2 = False
ec_2 = False
ec_12 = False
re_2 = False
re_12 = False
rc_2 = False
rc_12 = False
alpha_2 = False
alpha_12 = False
delta_2 = False
delta_12 = False
dr_2 = False
dr_12 = False
cweight_2 = False
beta_2 = False
Asub_2 = False
Cmax_2 = False
Cmin_2 = False
Cmax_12 = False
Cmin_12 = False

# Single Element Bounds
eshift_1_bnds = [(None, 0)]
ec_1_bnds = [(0.000001, None)]
re_1_bnds = [(0.0000001, None)]
rc_1_bnds = [(5, 9)]
alpha_1_bnds = [(0.00000001, None)]
delta_1_bnds = [(None, None)]
dr_1_bnds = [(0, None)]
cweight_1_bnds = [(None, None)]
beta_1_bnds = [(0, None)]
Asub_1_bnds = [(0, None)]
Cmax_1_bnds = [(None, 3)]
Cmin_1_bnds = [(0.5, None)]

# Binary Element Bounds
eshift_2_bnds = [(None, 0)]
ec_12_bnds = [(0.0000001, None)]
ec_2_bnds = [(0.0000001, None)]
re_12_bnds = [(0.0000001, None)]
re_2_bnds = [(0.0000001, None)]
rc_12_bnds = [(5, 9)]
rc_2_bnds = [(4, 9)]
alpha_12_bnds = [(0.00000001, None)]
alpha_2_bnds = [(0.00000001, None)]
delta_12_bnds = [(0.00000001, None)]
delta_2_bnds = [(0.00000001, None)]
dr_12_bnds = [(0, None)]
dr_2_bnds = [(0, None)]
cweight_2_bnds = [(None, None)]
beta_2_bnds = [(0, None)]
Asub_2_bnds = [(0, None)]
Cmax_2_bnds = [(None, None)]
Cmin_2_bnds = [(None, None)]
Cmax_12_bnds = [(2, 2.9)]
Cmin_12_bnds = [(0.541, 2)]

#------- END USER INPUT --------

ec_bnds = (ec_1_bnds, ec_12_bnds, ec_2_bnds)
re_bnds = (re_1_bnds, re_12_bnds, re_2_bnds)
rc_bnds = (rc_1_bnds, rc_12_bnds, rc_2_bnds)
alpha_bnds = (alpha_1_bnds, alpha_12_bnds, alpha_2_bnds)
delta_bnds = (delta_1_bnds, delta_12_bnds, delta_2_bnds)
dr_bnds = (dr_1_bnds, dr_12_bnds, dr_2_bnds)
cweight_bnds = (cweight_1_bnds, cweight_2_bnds)
beta_bnds = (beta_1_bnds, beta_2_bnds)
Asub_bnds = (Asub_1_bnds, Asub_2_bnds)
# GET ELEMENT TYPES
elements = np.genfromtxt(input_file, skip_header=1, max_rows=1, dtype=str)
if len(np.atleast_1d(elements)) != 1:
    elem1 = elements[0]
    elem2 = elements[1]
    if len(np.atleast_1d(elements)) == 3:
        elem3 = elements[2]
else:
    elem1 = str(elements)

# GET TRAINING SIZE
training_size = 0
print('Using simulations from the following dump files:')
for file in os.listdir(r'.'):
    if(file.endswith('.dump')):
        print(file)
        nsims = np.genfromtxt(file, skip_header=1, max_rows=1, usecols=4)
        training_size += nsims

training_size = int(training_size)
print('\nTraining size = '+str(training_size)+'\n')

# GET DEBUG FILENAME
string_1 = np.genfromtxt(input_file, skip_header=0, usecols=0, dtype=str)
debug_pos = np.where(string_1 == 'calibrationparameters:potentialoutputfile:')
debug_pos = int(*debug_pos)+1
debug = np.genfromtxt(input_file, skip_header=debug_pos, max_rows=1, usecols=0, dtype=str)
debug = 'DEBUG/'+str(debug)+'.debug1'


# GET PARAMETERS TO FIT
if ec_1 == True or ec_12 == True or ec_2 == True:
    ec = True
else:
    ec = False
if re_1 == True or re_12 == True or re_2 == True:
    re = True
else:
    re = False
if rc_1 == True or rc_12 == True or rc_2 == True:
    rc = True
else:
    rc = False
if alpha_1 == True or alpha_12 == True or alpha_2 == True:
    alpha = True
else:
    alpha = False
if delta_1 == True or delta_12 == True or delta_2 == True:
    delta = True
else:
    delta = False
if dr_1 == True or dr_12 == True or dr_2 == True:
    dr = True
else:
    dr = False
if beta_1 == True or beta_2 == True:
    beta = True
else:
    beta = False
if cweight_1 == True or cweight_2 == True:
    cweight = True
else:
    cweight = False
if Asub_1 == True or Asub_2 == True:
    Asub = True
else:
    Asub = False
#---------------------------------
#--------------------------------
params = np.array([eshift_1, eshift_2, ec, re, rc, alpha, delta, dr, cweight, beta, Asub, Cmax_1, Cmin_1, Cmax_2, Cmin_2, Cmax_12, Cmin_12])
print('Array of parameters to adjust. True will be changed, False will not.\n[eshift1, eshift2, ec, re, rc, alpha, dr, cweight, beta, Asub, Cmax1, Cmin1, Cmax2, Cmin2, Cmax12, Cmin12]\n')
print(params)
params_to_fit = np.zeros(17)
for i in range(17):
    if params[i] == True:
        params_to_fit[i] = params[i]
    else:
        continue
print(params_to_fit)
num_elem = int(len(np.atleast_1d(elements)))
ind_values = (num_elem * (num_elem + 1)) / 2
ec_to_fit = np.array([ec_1, ec_12, ec_12, ec_2])
re_to_fit = np.array([re_1, re_12, re_12, re_2])
rc_to_fit = np.array([rc_1, rc_12, rc_12, rc_2])
alpha_to_fit = np.array([alpha_1, alpha_12, alpha_12, alpha_2])
delta_to_fit = np.array([delta_1, delta_12, delta_12, delta_2])
dr_to_fit = np.array([dr_1, dr_12, dr_12, dr_2])
cweight_to_fit = np.array([cweight_1, cweight_2])
beta_to_fit = np.array([beta_1, beta_2])
Asub_to_fit = np.array([Asub_1, Asub_2])
# test_params = [ec_to_fit, re_to_fit, rc_to_fit, alpha_to_fit, delta_to_fit, dr_to_fit, beta_to_fit, Asub_to_fit]

ec_to_bnds = np.array([ec_1, ec_12, ec_2])
re_to_bnds = np.array([re_1, re_12, re_2])
rc_to_bnds = np.array([rc_1, rc_12, rc_2])
alpha_to_bnds = np.array([alpha_1, alpha_12, alpha_2])
delta_to_bnds = np.array([delta_1, delta_12, delta_2])
dr_to_bnds = np.array([dr_1, dr_12, dr_2])
cweight_to_bnds = np.array([cweight_1, cweight_2])
beta_to_bnds = np.array([beta_1, beta_2])
Asub_to_bnds = np.array([Asub_1, Asub_2])
bnds_params = [ec_to_bnds, re_to_bnds, rc_to_bnds, alpha_to_bnds, delta_to_bnds, dr_to_bnds, cweight_to_bnds, beta_to_bnds, Asub_to_bnds]


# print('-----------------------------')
# print(test_params)


def get_pos(file):
    string = np.genfromtxt(file, skip_header=0, usecols=0, dtype=str)
    eshift1_pos = np.where(string == 'stateequationconstants:'+elem1+':eshift_0:eshift:')
    ec_pos = np.where(string == 'stateequationconstants:all_all:eamscreenedr_0:ec:')
    re_pos = np.where(string == 'stateequationconstants:all_all:eamscreenedr_0:re:')
    rc_pos = np.where(string == 'stateequationconstants:all_all:eamscreenedr_0:rc:')
    alpha_pos = np.where(string == 'stateequationconstants:all_all:eamscreenedr_0:alpha:')
    delta_pos = np.where(string == 'stateequationconstants:all_all:eamscreenedr_0:delta:')
    dr_pos = np.where(string == 'stateequationconstants:all_all:eamscreenedr_0:dr:')
    cweight_pos = np.where(string == 'stateequationconstants:all_all:eamscreenedr_0:cweight:')
    beta_pos = np.where(string == 'stateequationconstants:all_all:eamscreenedr_0:beta:')
    Asub_pos = np.where(string == 'stateequationconstants:all_all:eamscreenedr_0:Asub:')
    Cmax_111_pos = np.where(string == 'screening:'+elem1+'_'+elem1+'_'+elem1+':Cmax:')
    Cmin_111_pos = np.where(string == 'screening:'+elem1+'_'+elem1+'_'+elem1+':Cmin:')

    num_elem = int(len(np.atleast_1d(elements)))

    if num_elem != 1:
        Cmax_222_pos = np.where(string == 'screening:'+elem2+'_'+elem2+'_'+elem2+':Cmax:')
        Cmin_222_pos = np.where(string == 'screening:'+elem2+'_'+elem2+'_'+elem2+':Cmin:')

        Cmax_112_pos = np.where(string == 'screening:'+elem1+'_'+elem1+'_'+elem2+':Cmax:')
        Cmin_112_pos = np.where(string == 'screening:'+elem1+'_'+elem1+'_'+elem2+':Cmin:')

        Cmax_122_pos = np.where(string == 'screening:'+elem1+'_'+elem2+'_'+elem2+':Cmax:')
        Cmin_122_pos = np.where(string == 'screening:'+elem1+'_'+elem2+'_'+elem2+':Cmin:')

        Cmax_121_pos = np.where(string == 'screening:'+elem1+'_'+elem2+'_'+elem1+':Cmax:')
        Cmin_121_pos = np.where(string == 'screening:'+elem1+'_'+elem2+'_'+elem1+':Cmin:')

        Cmax_211_pos = np.where(string == 'screening:'+elem2+'_'+elem1+'_'+elem1+':Cmax:')
        Cmin_211_pos = np.where(string == 'screening:'+elem2+'_'+elem1+'_'+elem1+':Cmin:')

        Cmax_221_pos = np.where(string == 'screening:'+elem2+'_'+elem2+'_'+elem1+':Cmax:')
        Cmin_221_pos = np.where(string == 'screening:'+elem2+'_'+elem2+'_'+elem1+':Cmin:')

        Cmax_212_pos = np.where(string == 'screening:'+elem2+'_'+elem1+'_'+elem2+':Cmax:')
        Cmin_212_pos = np.where(string == 'screening:'+elem2+'_'+elem1+'_'+elem2+':Cmin:')

# COME BACK AND FIX FOR 3 ELEMENTS
    # if num_elem == 3:
    #     Cmax_333 = np.where(string == 'screening:'+elem3+'_'+elem3+'_'+elem3+':Cmax:')
    #     Cmin_333 = np.where(string == 'screening:'+elem3+'_'+elem3+'_'+elem3+':Cmin:')

    #     Cmax_331 = np.where(string == 'screening:'+elem3+'_'+elem3+'_'+elem1+':Cmax:')
    #     Cmin_331 = np.where(string == 'screening:'+elem3+'_'+elem3+'_'+elem1+':Cmin:')

    eshift1_pos = int(*eshift1_pos)+2
    ec_pos = int(*ec_pos)+2
    re_pos = int(*re_pos)+2
    rc_pos = int(*rc_pos)+2
    alpha_pos = int(*alpha_pos)+2
    delta_pos = int(*delta_pos)+2
    dr_pos = int(*dr_pos)+2
    cweight_pos = int(*cweight_pos)+2
    beta_pos = int(*beta_pos)+2
    Asub_pos = int(*Asub_pos)+2
    # print(*Cmax_111_pos)
    Cmax_111_pos = int(*Cmax_111_pos)+2
    Cmin_111_pos = int(*Cmin_111_pos)+2
    if num_elem != 1:
        Cmax_222_pos = int(*Cmax_222_pos)+2
        Cmin_222_pos = int(*Cmin_222_pos)+2

        Cmax_112_pos = int(*Cmax_112_pos)+2
        Cmin_112_pos = int(*Cmin_112_pos)+2

        Cmax_122_pos = int(*Cmax_122_pos)+2
        Cmin_122_pos = int(*Cmin_122_pos)+2

        Cmax_121_pos = int(*Cmax_121_pos)+2
        Cmin_121_pos = int(*Cmin_121_pos)+2

        Cmax_221_pos = int(*Cmax_221_pos)+2
        Cmin_221_pos = int(*Cmin_221_pos)+2

        Cmax_211_pos = int(*Cmax_211_pos)+2
        Cmin_211_pos = int(*Cmin_211_pos)+2

        Cmax_212_pos = int(*Cmax_212_pos)+2
        Cmin_212_pos = int(*Cmin_212_pos)+2

        Cmax_12s_pos = [Cmax_112_pos, Cmax_122_pos, Cmax_121_pos, Cmax_221_pos, Cmax_211_pos, Cmax_212_pos]
        Cmin_12s_pos = [Cmin_112_pos, Cmin_122_pos, Cmin_121_pos, Cmin_221_pos, Cmin_211_pos, Cmin_212_pos]

        # Cmax_2s_pos = np.array([Cmax_222_pos, Cmax_221_pos, Cmax_211_pos, Cmax_212_pos])
        # Cmin_2s_pos = np.array([Cmin_222_pos, Cmin_221_pos, Cmin_211_pos, Cmin_212_pos])

    if len(np.atleast_1d(elements)) == 2:
        eshift2_pos = np.where(string == 'stateequationconstants:'+elem2+':eshift_0:eshift:')
        eshift2_pos = int(*eshift2_pos)+2
        positions = [eshift1_pos, eshift2_pos, ec_pos, re_pos, rc_pos, alpha_pos, delta_pos, dr_pos, cweight_pos, beta_pos, Asub_pos, Cmax_111_pos, Cmin_111_pos, Cmax_222_pos, Cmin_222_pos, Cmax_12s_pos, Cmin_12s_pos]


    else:
        positions = np.array([eshift1_pos, 0, ec_pos, re_pos, rc_pos, alpha_pos, delta_pos, dr_pos, cweight_pos, beta_pos, Asub_pos, Cmax_111_pos, Cmin_111_pos, 0, 0, 0, 0])

    return positions

def initial_guess():
    num_elem = int(len(np.atleast_1d(elements)))
    cols_mat = num_elem ** 2
    ind_values = int((num_elem * (num_elem+1)) / 2)


    args1 = get_pos(input_file)

    guess = []
    for i in range(len(params_to_fit)-2):
        guess.append(int(args1[i]-1))

    eshift_1 = np.genfromtxt(input_file, skip_header=guess[0], max_rows=1, unpack=True)
    ec = np.genfromtxt(input_file, skip_header=guess[2], max_rows=1, unpack=True)
    re = np.genfromtxt(input_file, skip_header=guess[3], max_rows = 1, unpack=True)
    rc = np.genfromtxt(input_file, skip_header=guess[4], max_rows = 1, unpack=True)
    alpha = np.genfromtxt(input_file, skip_header=guess[5], max_rows = 1, unpack=True)
    delta = np.genfromtxt(input_file, skip_header=guess[6], max_rows = 1, unpack=True)
    # print(delta)
    # print('-------------------')
    dr = np.genfromtxt(input_file, skip_header=guess[7], max_rows = 1, unpack=True)
    cweight = np.genfromtxt(input_file, skip_header=guess[8], max_rows=1, unpack=True)
    beta = np.genfromtxt(input_file, skip_header=guess[9], max_rows = 1, unpack=True)
    Asub = np.genfromtxt(input_file, skip_header=guess[10], max_rows = 1, unpack=True)

    if num_elem != 1:
        eshift_2 = np.genfromtxt(input_file, skip_header=guess[1], max_rows=1, unpack=True)
    else:
        eshift_2 = np.array([0])

    Cmax_111 = np.genfromtxt(input_file, skip_header=guess[11], max_rows=1, unpack=True)
    Cmin_111 = np.genfromtxt(input_file, skip_header=guess[12], max_rows = 1, unpack=True)

    Cmax_222 = np.genfromtxt(input_file, skip_header=guess[13], max_rows=1, unpack=True)
    Cmin_222 = np.genfromtxt(input_file, skip_header=guess[14], max_rows = 1, unpack=True)

    Cmax_12s = []
    Cmin_12s = []
    for i in range((num_elem ** 3) - num_elem):
        num_max = args1[15][i]-1
        num_min = args1[16][i]-1
        app = np.genfromtxt(input_file, skip_header=num_max, max_rows=1, unpack=True)
        Cmax_12s.append(app)
        Cmin_12s.append(np.genfromtxt(input_file, skip_header=num_min, max_rows=1, unpack=True))
    # print(Cmax_12s)
    # print('Cmax')
    # print(Cmax_12s)
    if params_to_fit[15] == True:
        del Cmax_12s[5]
        del Cmax_12s[2]
    if params_to_fit[16] == True:
        del Cmin_12s[5]
        del Cmin_12s[2]

    if num_elem != 1:
        total = (eshift_1, eshift_2, ec, re, rc, alpha, delta, dr, cweight, beta, Asub, Cmax_111, Cmin_111, Cmax_222, Cmin_222, Cmax_12s, Cmin_12s)
    else:
        total = (eshift_1, ec, re, rc, alpha, delta, dr, cweight, beta, Asub, Cmax_111, Cmin_111)

    if num_elem != 1:
        new_ec = np.zeros((num_elem, num_elem))
        new_re = np.zeros((num_elem, num_elem))
        new_rc = np.zeros((num_elem, num_elem))
        new_alpha = np.zeros((num_elem, num_elem))
        new_delta = np.zeros((num_elem, num_elem))
        new_dr = np.zeros((num_elem, num_elem))
        new_cweight = np.zeros(num_elem)
        new_beta = np.zeros(num_elem)
        new_Asub = np.zeros(num_elem)
        if len(np.atleast_1d(elements)) != 1:
            count = 0
            for i in range(num_elem):
                for j in range(num_elem):
                    new_ec[i][j] = ec[count]
                    new_re[i][j] = re[count]
                    new_rc[i][j] = rc[count]
                    new_alpha[i][j] = alpha[count]
                    new_delta[i][j] = delta[count]
                    new_dr[i][j] = dr[count]
                    count += 1
                new_cweight[i] = cweight[i]
                new_beta[i] = beta[i]
                new_Asub[i] = Asub[i]
            ec = np.delete(ec,1)
            re = np.delete(re,1)
            rc = np.delete(rc,1)
            dr = np.delete(dr,1)
            alpha = np.delete(alpha,1)
            delta = np.delete(delta,1)

    eshift_1= eshift_1.tolist()
    eshift_2 = eshift_2.tolist()
    ec = ec.tolist()
    re = re.tolist()
    rc = rc.tolist()
    alpha = alpha.tolist()
    delta = delta.tolist()
    dr = dr.tolist()
    cweight = cweight.tolist()
    beta = beta.tolist()
    Asub = Asub.tolist()
    Cmax_111 = Cmax_111.tolist()
    Cmin_111 = Cmin_111.tolist()
    Cmax_222 = Cmax_222.tolist()
    Cmin_222 = Cmin_222.tolist()
    values = eshift_1, eshift_2, ec, re, rc, alpha, delta, dr, cweight, beta, Asub, Cmax_111, Cmin_111, Cmax_222, Cmin_222, Cmax_12s, Cmin_12s
    values = list(values)
    return_values = []

    if params_to_fit[0] == True:
        return_values.append(eshift_1)
    if params_to_fit[1] == True:
        return_values.append(eshift_2)
    if params_to_fit[2] == True:
        if num_elem != 1:
            for i in range(int(len(bnds_params[0]))):
                if num_elem != 1 and bnds_params[0][i] == True:
                    return_values.append(ec[i])
        else:
            return_values.append(ec)
        # if num_elem == 1 and bnds_params[0][1] == True:
        #     return_values.append(ec)
    if params_to_fit[3] == True:
        if num_elem != 1:
            for i in range(int(len(bnds_params[1]))):
                if num_elem != 1 and bnds_params[1][i] == True:
                    return_values.append(re[i])
        else:
            return_values.append(re)

#         for i in range(ind_values):
#             if num_elem != 1:
#                 return_values.append(re[i])
#             else:
#                 return_values.append(re)
    if params_to_fit[4] == True:
        if num_elem != 1:
            for i in range(int(len(bnds_params[2]))):
                if num_elem != 1 and bnds_params[2][i] == True:
                    return_values.append(rc[i])
        else:
            return_values.append(rc)

        # for i in range(ind_values):
        #     if num_elem != 1:
        #         return_values.append(rc[i])
        #     else:
        #         return_values.append(rc)
    if params_to_fit[5] == True:
        if num_elem != 1:
            for i in range(int(len(bnds_params[3]))):
                if num_elem != 1 and bnds_params[3][i] == True:
                    return_values.append(alpha[i])
        else:
            return_values.append(alpha)

    if params_to_fit[6] == True:
        if num_elem != 1:
            for i in range(int(len(bnds_params[4]))):
                if num_elem != 1 and bnds_params[4][i] == True:
                    return_values.append(delta[i])
        else:
            return_values.append(delta)

        # for i in range(ind_values):
        #     if num_elem != 1:
        #         return_values.append(alpha[i])
        #     else:
        #         return_values.append(alpha)
    if params_to_fit[7] == True:
        if num_elem != 1:
            for i in range(int(len(bnds_params[5]))):
                if num_elem != 1 and bnds_params[5][i] == True:
                    return_values.append(dr[i])
        else:
            return_values.append(dr)

        # for i in range(ind_values):
        #     if num_elem != 1:
        #         return_values.append(dr[i])
        #     else:
        #         return_values.append(dr)
    if params_to_fit[8] == True:
        if num_elem != 1:
            for i in range(int(len(bnds_params[6]))):
                if bnds_params[6][i] == True:
                    return_values.append(cweight[i])
        else:
            return_values.append(cweight)

    if params_to_fit[9] == True:
        if num_elem != 1:
            for i in range(int(len(bnds_params[7]))):
                if num_elem != 1 and bnds_params[7][i] == True:
                    return_values.append(beta[i])
        else:
            return_values.append(beta)

        # for i in range(num_elem):
        #     if num_elem != 1:
        #         return_values.append(beta[i])
        #     else:
        #         return_values.append(beta)
    if params_to_fit[10] == True:
        if num_elem != 1:
            for i in range(int(len(bnds_params[8]))):
                if num_elem != 1 and bnds_params[8][i] == True:
                    return_values.append(Asub[i])
        else:
            return_values.append(Asub)

        # for i in range(num_elem):
        #     if num_elem != 1:
        #         return_values.append(Asub[i])
        #     else:
        #         return_values.append(Asub)
    if params_to_fit[11] == True:
        return_values.append(Cmax_111)
    if params_to_fit[12] == True:
        return_values.append(Cmin_111)
    if params_to_fit[13] == True:
        return_values.append(Cmax_222)
    if params_to_fit[14] == True:
        return_values.append(Cmin_222)
    if params_to_fit[15] == True:
        for i in range((num_elem **3) - (2 * num_elem)):
            return_values.append(Cmax_12s[i].tolist())
    if params_to_fit[16] == True:
        for i in range((num_elem **3) - (2 * num_elem)):
            return_values.append(Cmin_12s[i].tolist())

    # print(return_values)
    # print('---------------------------')
    # new_return_values = []
    # counter = 0
    # for i in range(int(len(params_to_fit))):
    #     if params_to_fit[i] != 0:
    #         if i <= num_elem:
    #             new_return_values.append(return_values[i])
    #             counter += 1
    #         if i > num_elem and i <= 6:
    #             for j in range(int(ind_values)):
    #                 if test_params[i-2][j] != 0:
    #                     new_return_values.append(return_values[counter])
    #                     counter += 1
    #         if i > 6 and i <= 8:
    #             for j in range(num_elem):
    #                 if test_params[i-2][j] != 0:
    #                     new_return_values.append(return_values[counter])
    #                     counter += 1
    #         if i > 8:
    #             new_return_values.append(return_values[counter])
    #             counter += 1

    # print('delta = '+str(delta))
    # print(return_values)
    # print('total')
    # print(total)
    return return_values, total

def write_input(args, value_args):
    positions = get_pos(input_file)
    cmd_copy = 'cp '+input_file+' temp.input'
    # print(cmd_copy)
    os.system(cmd_copy)

    num_elem = int(len(np.atleast_1d(elements)))
    ind_values = (num_elem * (num_elem+1)) / 2
    ec_w = []
    re_w = []
    rc_w = []
    alpha_w = []
    delta_w = []
    dr_w = []
    cweight_w = []
    beta_w = []
    Asub_w = []
    Cmax_1_w = []
    Cmin_1_w = []
    Cmax_2_w = []
    Cmin_2_w = []
    Cmax_12_w = []
    Cmin_12_w = []
    
    new_values = []
    initial, all_values = initial_guess()
    # print('###########################')
    # print(all_values)
    mwn = 0
    # if num_elem != 1:
    for i in range(len(args)):
        if i < num_elem:
            if args[i] == 0:
                new_values.append(0)
            else:
                new_values.append(value_args[mwn])
                mwn += 1
        if i >= num_elem and i < (int(len(params_to_fit))-9):
            for j in range(int(ind_values)):
                if bnds_params[i-num_elem][j] == 0 and num_elem != 1:
                    new_values.append(all_values[i][j])
                    if i == num_elem:
                        ec_w.append(all_values[i][j])
                        if j == 1:
                            ec_w.append(all_values[i][j])
                    if i == num_elem+1:
                        re_w.append(all_values[i][j])
                        if j == 1:
                            re_w.append(all_values[i][j])
                    if i == num_elem+2:
                        rc_w.append(all_values[i][j])
                        if j == 1:
                            rc_w.append(all_values[i][j])
                    if i == num_elem+3:
                        alpha_w.append(all_values[i][j])
                        if j == 1:
                            alpha_w.append(all_values[i][j])
                    if i == num_elem+4:
                        delta_w.append(all_values[i][j])
                        if j == 1:
                            delta_w.append(all_values[i][j])
                    if i == num_elem+5:
                        dr_w.append(all_values[i][j])
                        if j == 1:
                            dr_w.append(all_values[i][j])
                if bnds_params[i-num_elem][j] == 0 and num_elem == 1:
                    new_values.append(all_values[i])
                    if i == num_elem:
                        ec_w.append(all_values[i])
                    if i == num_elem+1:
                        re_w.append(all_values[i])
                    if i == num_elem+2:
                        rc_w.append(all_values[i])
                    if i == num_elem+3:
                        alpha_w.append(all_values[i])
                    if i == num_elem+4:
                        delta_w.append(all_values[i])
                    if i == num_elem+5:
                        dr_w.append(all_values[i])
                if bnds_params[i-num_elem][j] != 0 and num_elem != 1:
                    new_values.append(value_args[mwn])
                    if i == num_elem:
                        ec_w.append(value_args[mwn])
                        if j == 1:
                            ec_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+1:
                        re_w.append(value_args[mwn])
                        if j == 1:
                            re_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+2:
                        rc_w.append(value_args[mwn])
                        if j == 1:
                            rc_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+3:
                        alpha_w.append(value_args[mwn])
                        if j == 1:
                            alpha_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+4:
                        delta_w.append(value_args[mwn])
                        if j == 1:
                            delta_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+5:
                        dr_w.append(value_args[mwn])
                        if j == 1:
                            dr_w.append(value_args[mwn])
                        mwn += 1
                if bnds_params[i-num_elem][j] != 0 and num_elem == 1:
                    new_values.append(value_args[mwn])
                    if i == num_elem:
                        ec_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+1:
                        re_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+2:
                        rc_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+3:
                        alpha_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+4:
                        delta_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+5:
                        dr_w.append(value_args[mwn])
                        mwn += 1

        if i >= (int(len(params_to_fit))-9) and i < (int(len(params_to_fit))-6):
            if num_elem != 1:
                for j in range(num_elem):
                    if bnds_params[i-num_elem][j] == 0:
                        new_values.append(all_values[i+j])
                        if i == 8:
                            cweight_w.append(all_values[i+j])
                        if i == 9:
                            beta_w.append(all_values[i+j])
                        else:
                            Asub_w.append(all_values[i+j])
                    if bnds_params[i-num_elem][j] != 0:
                        new_values.append(value_args[mwn])
                        if i == 8:
                            cweight_w.append(value_args[mwn])
                            mwn += 1
                        if i == 9:
                            beta_w.append(value_args[mwn])
                            mwn += 1
                        if i == 10:
                            Asub_w.append(value_args[mwn])
                            mwn += 1
            else:
                if params_to_fit[i] == 0:
                    new_values.append(all_values[i])
                    if i == 8:
                        cweight_w.append(all_values[i])
                    if i == 9:
                        beta_w.append(all_values[i])
                    else:
                        Asub_w.append(all_values[i])
                if params_to_fit[i] != 0:
                    new_values.append(value_args[mwn])
                    if i == 8:
                        cweight_w.append(value_args[mwn])
                        mwn += 1
                    if i == 9:
                        beta_w.append(value_args[mwn])
                        mwn += 1
                    if i == 10:
                        Asub_w.append(value_args[mwn])
                        mwn += 1

        if i >= (int(len(params_to_fit))-6) and i < (int(len(params_to_fit))-2):
            if args[i] == 0:
                new_values.append(0)
                if i == 11:
                    Cmax_1_w.append(0)
                if i == 12:
                    Cmin_1_w.append(0)
                if i == 13:
                    Cmax_2_w.append(0)
                if i == 14:
                    Cmin_2_w.append(0)

            else:
                new_values.append(value_args[mwn])
                if i == 11:
                    Cmax_1_w.append(value_args[mwn])
                    mwn += 1
                if i == 12:
                    Cmin_1_w.append(value_args[mwn])
                    mwn += 1
                if i == 13:
                    Cmax_2_w.append(value_args[mwn])
                    mwn += 1
                if i == 14:
                    Cmin_2_w.append(value_args[mwn])
                    mwn += 1

        if i >= (int(len(params_to_fit))-2):
            if args[i] == 0:
                new_values.append(0)
                for j in range((num_elem **3) - (2 * num_elem)):
                    if i == 15:
                        Cmax_12_w.append(0)
                    if i == 16:
                        Cmin_12_w.append(0)
            else:
                for j in range((num_elem **3) - (2 * num_elem)):
                    new_values.append(value_args[mwn])
                    if i == 15:
                        Cmax_12_w.append(value_args[mwn])
                        mwn += 1
                    if i == 16:
                        Cmin_12_w.append(value_args[mwn])
                        mwn += 1

    # print('new')
    # print(new_values)
    if num_elem != 1:
        Cmax_12_w.insert(2, Cmax_12_w[0])
        Cmin_12_w.insert(2, Cmin_12_w[0])
        Cmax_12_w.append(Cmax_12_w[3])
        Cmin_12_w.append(Cmin_12_w[3])
    # print(Cmax_12_w)
    # print(Cmin_12_w)
    # if num_elem != 1:
    mult_value = (ec_w, re_w, rc_w, alpha_w, delta_w, dr_w, cweight_w, beta_w, Asub_w, Cmax_1_w, Cmin_1_w, Cmax_2_w, Cmin_2_w, Cmax_12_w, Cmin_12_w)

    # else:
    #     mult_value = value_args

    # print('!!!!!!!!!!!!!!!!!!!!!!!!')
    # print(mult_value)

    for i in range(len(args)):
        ii = 0
        if args[i] != 0:
            if i < num_elem:
                cmd_write = 'sed -i -e "'+str(positions[i])+'s/.*/'+str(value_args[i])+'/g" temp.input'
                os.system(cmd_write)
            elif i >= int(len(args))-2 and num_elem != 1:
                for j in range(int(len(positions[i]))):
                    # if j == 2:
                    #     cmd_write = 'sed -i -e "'+str(positions[i][j])+'s/.*/'+str(mult_value[i-num_elem][0])+'/g" temp.input'
                    # elif j == 5:
                    #     cmd_write = 'sed -i -e "'+str(positions[i][j])+'s/.*/'+str(mult_value[i-num_elem][2])+'/g" temp.input'
                    # else:
                    cmd_write = 'sed -i -e "'+str(positions[i][j])+'s/.*/'+str(mult_value[i-num_elem][j])+'/g" temp.input'
                    os.system(cmd_write)
            else:
                if num_elem != 1:
                    replace = ' '.join(map(str,mult_value[i-num_elem]))
                else:
                    # print(mult_value)
                    replace = str(*mult_value[i-2])
                cmd_write = 'sed -i -e "'+str(positions[i])+'s/.*/'+replace+'/g" temp.input'
                # print(cmd_write)
                os.system(cmd_write)

    return value_args

def evals():
    cmd1 = "bash run.sh"
    os.system(cmd1)
    state_errors = np.genfromtxt(debug, skip_header=4, unpack=False, max_rows=training_size, usecols=6)
    return state_errors

def function(v):
    write_input(params_to_fit,v)
    a = evals()
    # c = np.sqrt(np.mean(a ** 2))
    c = 0
    for i in range(training_size):
        b = a[i] **2
        c += b
    c *= (1/training_size)
    c = np.sqrt(c)
    print('----------------------------------------------')
    print_list = ('eshift', 'eshift', 'ec', 're', 'rc', 'alpha', 'delta', 'dr', 'cweight', 'beta', 'Asub', 'Cmax', 'Cmin', 'Cmax', 'Cmin', 'Cmax', 'Cmin')

    num_elem = int(len(np.atleast_1d(elements)))
    element_list = []
    x = 0
    if num_elem != 1:
        while (num_elem-x) > 0:
            for i in range(num_elem):
                for j in range(num_elem-x):
                    element_list.append(elements[i]+elements[j+x])
                x += 1

        xx = 0
        # for i in range(len(params_to_fit)):
        #     elem_count = 0
        #     if params_to_fit[i] != 0 and i < len(np.atleast_1d(elements)):
        #         print('New '+print_list[i]+'_'+elements[i]+' = '+str(v[i]))
        #         xx += 1
        #     if params_to_fit[i] != 0 and i >= len(np.atleast_1d(elements)) and i < int(len(params_to_fit))-9:
        #         for j in range(int(ind_values)):
        #             if bnds_params[i-num_elem][j] == False:
        #                 continue
        #             else:
        #                 print('New '+print_list[i]+'_'+element_list[j]+' = '+str(v[xx]))
        #                 xx += 1
        #     if params_to_fit[i] != 0 and i >= int(len(params_to_fit))-9 and i < int(len(params_to_fit))-6:
        #         for j in range(num_elem):
        #             if bnds_params[i-num_elem][j] == False:
        #                 continue
        #             else:
        #                 print('New '+print_list[i]+'_'+element_list[j+j]+' = '+str(v[xx]))
        #                 xx += 1
        #     if params_to_fit[i] != 0 and i >= int(len(params_to_fit))-6 and i < int(len(params_to_fit))-5:
        #         print('New '+print_list[i]+'_'+elem1+'_'+elem1+'_'+elem1+' = '+str(v[xx]))
        #         xx += 1
        #     if params_to_fit[i] != 0 and i >= int(len(params_to_fit))-5 and i < int(len(params_to_fit))-4:
        #         print('New '+print_list[i]+'_'+elem1+'_'+elem1+'_'+elem1+' = '+str(v[xx]))
        #         xx += 1
        #     if params_to_fit[i] != 0 and i >= int(len(params_to_fit))-4 and i < int(len(params_to_fit))-3:
        #         print('New '+print_list[i]+'_'+elem2+'_'+elem2+'_'+elem2+' = '+str(v[xx]))
        #         xx += 1
        #     if params_to_fit[i] != 0 and i >= int(len(params_to_fit))-3 and i < int(len(params_to_fit))-2:
        #         print('New '+print_list[i]+'_'+elem2+'_'+elem2+'_'+elem2+' = '+str(v[xx]))
        #         xx += 1
        #     if params_to_fit[i] != 0 and i >= int(len(params_to_fit))-2 and i < int(len(params_to_fit))-1:
        #         # for j in range((num_elem ** 3)-num_elem):
        #         print('New '+print_list[i]+'_'+elem1+'_'+elem1+'_'+elem2+' = '+str(v[xx]))
        #         print('New '+print_list[i]+'_'+elem1+'_'+elem2+'_'+elem1+' = '+str(v[xx]))

        #         xx += 1
        #         print('New '+print_list[i]+'_'+elem1+'_'+elem2+'_'+elem2+' = '+str(v[xx]))
        #         xx += 1
        #         print('New '+print_list[i]+'_'+elem2+'_'+elem2+'_'+elem1+' = '+str(v[xx]))
        #         print('New '+print_list[i]+'_'+elem2+'_'+elem1+'_'+elem2+' = '+str(v[xx]))
        #         xx += 1
        #         print('New '+print_list[i]+'_'+elem2+'_'+elem1+'_'+elem1+' = '+str(v[xx]))
        #         xx += 1
        #     if params_to_fit[i] != 0 and i >= int(len(params_to_fit))-1 and i < int(len(params_to_fit)):
        #         # for j in range((num_elem ** 3)-num_elem):
        #         print('New '+print_list[i]+'_'+elem1+'_'+elem1+'_'+elem2+' = '+str(v[xx]))
        #         print('New '+print_list[i]+'_'+elem1+'_'+elem2+'_'+elem1+' = '+str(v[xx]))
        #         xx += 1
        #         print('New '+print_list[i]+'_'+elem1+'_'+elem2+'_'+elem2+' = '+str(v[xx]))
        #         xx += 1
        #         print('New '+print_list[i]+'_'+elem2+'_'+elem2+'_'+elem1+' = '+str(v[xx]))
        #         print('New '+print_list[i]+'_'+elem2+'_'+elem1+'_'+elem2+' = '+str(v[xx]))
        #         xx += 1
        #         print('New '+print_list[i]+'_'+elem2+'_'+elem1+'_'+elem1+' = '+str(v[xx]))
        #         xx += 1

    # else:
        # element_list.append(elem1)
        # xx = 0
        # for i in range(len(params_to_fit)):
        #     if params_to_fit[i] != 0 and i < len(np.atleast_1d(elements)):
        #         print('New '+print_list[i]+'_'+str(elements)+' = '+str(v[i]))
        #         xx += 1
        #     if params_to_fit[i] != 0 and i >= len(np.atleast_1d(elements)) and i < int(len(params_to_fit))-9:
        #         for j in range(len(element_list)):
        #             print('New '+print_list[i]+'_'+str(elements)+' = '+str(v[xx]))
        #             xx += 1
        #     if params_to_fit[i] != 0 and i >= int(len(params_to_fit))-9 and i < int(len(params_to_fit))-6:
        #         for j in range(num_elem):
        #             print('New '+print_list[i]+'_'+str(elements)+' = '+str(v[xx]))
        #             xx += 1
        #     if params_to_fit[i] != 0 and i >= int(len(params_to_fit))-6 and  i < int(len(params_to_fit))-5:
        #         print('New '+print_list[i]+'_'+elem1+'_'+elem1+'_'+elem1+' = '+str(v[xx]))
        #         xx += 1
        #     if params_to_fit[i] != 0 and i >= int(len(params_to_fit))-5 and i < int(len(params_to_fit))-4:
        #         print('New '+print_list[i]+'_'+elem1+'_'+elem1+'_'+elem1+' = '+str(v[xx]))
        #         xx += 1




    print('rms = '+str(c))

    return c

def write_best_input(args, value_args, file):
    positions = get_pos(write_file)
    cmd_copy = 'cp '+file+' best_fit.input'
    # print(cmd_copy)
    os.system(cmd_copy)

    num_elem = int(len(np.atleast_1d(elements)))
    ind_values = (num_elem * (num_elem+1)) / 2
    ec_w = []
    re_w = []
    rc_w = []
    alpha_w = []
    delta_w = []
    dr_w = []
    cweight_w = []
    beta_w = []
    Asub_w = []
    Cmax_1_w = []
    Cmin_1_w = []
    Cmax_2_w = []
    Cmin_2_w = []
    Cmax_12_w = []
    Cmin_12_w = []
    
    new_values = []
    initial, all_values = initial_guess()
    # print('###########################')
    # print(all_values)
    mwn = 0
    # if num_elem != 1:
    for i in range(len(args)):
        if i < num_elem:
            if args[i] == 0:
                new_values.append(0)
            else:
                new_values.append(value_args[mwn])
                mwn += 1
        if i >= num_elem and i < (int(len(params_to_fit))-9):
            for j in range(int(ind_values)):
                if bnds_params[i-num_elem][j] == 0 and num_elem != 1:
                    new_values.append(all_values[i][j])
                    if i == num_elem:
                        ec_w.append(all_values[i][j])
                        if j == 1:
                            ec_w.append(all_values[i][j])
                    if i == num_elem+1:
                        re_w.append(all_values[i][j])
                        if j == 1:
                            re_w.append(all_values[i][j])
                    if i == num_elem+2:
                        rc_w.append(all_values[i][j])
                        if j == 1:
                            rc_w.append(all_values[i][j])
                    if i == num_elem+3:
                        alpha_w.append(all_values[i][j])
                        if j == 1:
                            alpha_w.append(all_values[i][j])
                    if i == num_elem+4:
                        delta_w.append(all_values[i][j])
                        if j == 1:
                            delta_w.append(all_values[i][j])
                    if i == num_elem+5:
                        dr_w.append(all_values[i][j])
                        if j == 1:
                            dr_w.append(all_values[i][j])
                if bnds_params[i-num_elem][j] == 0 and num_elem == 1:
                    new_values.append(all_values[i])
                    if i == num_elem:
                        ec_w.append(all_values[i])
                    if i == num_elem+1:
                        re_w.append(all_values[i])
                    if i == num_elem+2:
                        rc_w.append(all_values[i])
                    if i == num_elem+3:
                        alpha_w.append(all_values[i])
                    if i == num_elem+4:
                        delta_w.append(all_values[i])
                    if i == num_elem+5:
                        dr_w.append(all_values[i])
                if bnds_params[i-num_elem][j] != 0 and num_elem != 1:
                    new_values.append(value_args[mwn])
                    if i == num_elem:
                        ec_w.append(value_args[mwn])
                        if j == 1:
                            ec_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+1:
                        re_w.append(value_args[mwn])
                        if j == 1:
                            re_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+2:
                        rc_w.append(value_args[mwn])
                        if j == 1:
                            rc_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+3:
                        alpha_w.append(value_args[mwn])
                        if j == 1:
                            alpha_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+4:
                        delta_w.append(value_args[mwn])
                        if j == 1:
                            delta_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+5:
                        dr_w.append(value_args[mwn])
                        if j == 1:
                            dr_w.append(value_args[mwn])
                        mwn += 1
                if bnds_params[i-num_elem][j] != 0 and num_elem == 1:
                    new_values.append(value_args[mwn])
                    if i == num_elem:
                        ec_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+1:
                        re_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+2:
                        rc_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+3:
                        alpha_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+4:
                        delta_w.append(value_args[mwn])
                        mwn += 1
                    if i == num_elem+5:
                        dr_w.append(value_args[mwn])
                        mwn += 1

        if i >= (int(len(params_to_fit))-9) and i < (int(len(params_to_fit))-6):
            if num_elem != 1:
                for j in range(num_elem):
                    if bnds_params[i-num_elem][j] == 0:
                        new_values.append(all_values[i+j])
                        if i == 8:
                            cweight_w.append(all_values[i+j])
                        if i == 9:
                            beta_w.append(all_values[i+j])
                        else:
                            Asub_w.append(all_values[i+j])
                    if bnds_params[i-num_elem][j] != 0:
                        new_values.append(value_args[mwn])
                        if i == 8:
                            cweight_w.append(value_args[mwn])
                            mwn += 1
                        if i == 9:
                            beta_w.append(value_args[mwn])
                            mwn += 1
                        else:
                            Asub_w.append(value_args[mwn])
                            mwn += 1
            else:
                if params_to_fit[i] == 0:
                    new_values.append(all_values[i])
                    if i == 8:
                        cweight_w.append(all_values[i])
                    if i == 9:
                        beta_w.append(all_values[i])
                    else:
                        Asub_w.append(all_values[i])
                if params_to_fit[i] != 0:
                    new_values.append(value_args[mwn])
                    if i == 8:
                        cweight_w.append(value_args[mwn])
                        mwn += 1
                    if i == 9:
                        beta_w.append(value_args[mwn])
                        mwn += 1
                    else:
                        Asub_w.append(value_args[mwn])
                        mwn += 1

        if i >= (int(len(params_to_fit))-6) and i < (int(len(params_to_fit))-2):
            if args[i] == 0:
                new_values.append(0)
                if i == 11:
                    Cmax_1_w.append(0)
                if i == 12:
                    Cmin_1_w.append(0)
                if i == 13:
                    Cmax_2_w.append(0)
                if i == 14:
                    Cmin_2_w.append(0)

            else:
                new_values.append(value_args[mwn])
                if i == 11:
                    Cmax_1_w.append(value_args[mwn])
                    mwn += 1
                if i == 12:
                    Cmin_1_w.append(value_args[mwn])
                    mwn += 1
                if i == 13:
                    Cmax_2_w.append(value_args[mwn])
                    mwn += 1
                if i == 14:
                    Cmin_2_w.append(value_args[mwn])
                    mwn += 1

        if i >= (int(len(params_to_fit))-2):
            if args[i] == 0:
                new_values.append(0)
                for j in range((num_elem **3) - (2 * num_elem)):
                    if i == 15:
                        Cmax_12_w.append(0)
                    if i == 16:
                        Cmin_12_w.append(0)
            else:
                for j in range((num_elem **3) - (2 * num_elem)):
                    new_values.append(value_args[mwn])
                    if i == 15:
                        Cmax_12_w.append(value_args[mwn])
                        mwn += 1
                    if i == 16:
                        Cmin_12_w.append(value_args[mwn])
                        mwn += 1

    # print('new')
    # print(new_values)
    if num_elem != 1:
        Cmax_12_w.insert(2, Cmax_12_w[0])
        Cmin_12_w.insert(2, Cmin_12_w[0])
        Cmax_12_w.append(Cmax_12_w[3])
        Cmin_12_w.append(Cmin_12_w[3])
    # print(Cmax_12_w)
    # print(Cmin_12_w)
    # if num_elem != 1:
    mult_value = (ec_w, re_w, rc_w, alpha_w, delta_w, dr_w, cweight_w, beta_w, Asub_w, Cmax_1_w, Cmin_1_w, Cmax_2_w, Cmin_2_w, Cmax_12_w, Cmin_12_w)

    # else:
    #     mult_value = value_args

    # print('!!!!!!!!!!!!!!!!!!!!!!!!')
    # print(mult_value)

    for i in range(len(args)):
        ii = 0
        if args[i] != 0:
            if i < num_elem:
                cmd_write = 'sed -i -e "'+str(positions[i])+'s/.*/'+str(value_args[i])+'/g" best_fit.input'
                os.system(cmd_write)
            elif i >= int(len(args))-2 and num_elem != 1:
                for j in range(int(len(positions[i]))):
                    # if j == 2:
                    #     cmd_write = 'sed -i -e "'+str(positions[i][j])+'s/.*/'+str(mult_value[i-num_elem][0])+'/g" temp.input'
                    # elif j == 5:
                    #     cmd_write = 'sed -i -e "'+str(positions[i][j])+'s/.*/'+str(mult_value[i-num_elem][2])+'/g" temp.input'
                    # else:
                    cmd_write = 'sed -i -e "'+str(positions[i][j])+'s/.*/'+str(mult_value[i-num_elem][j])+'/g" best_fit.input'
                    os.system(cmd_write)
            else:
                if num_elem != 1:
                    replace = ' '.join(map(str,mult_value[i-num_elem]))
                else:
                    # print(mult_value)
                    replace = str(*mult_value[i-2])
                cmd_write = 'sed -i -e "'+str(positions[i])+'s/.*/'+replace+'/g" best_fit.input'
                # print(cmd_write)
                os.system(cmd_write)

def main():
    num_elem = int(len(np.atleast_1d(elements)))
    if len(np.atleast_1d(elements)) == 2:

        guess1, temp_guess = initial_guess()
        bnds = []
        if params_to_fit[0] == True:
            bnds += eshift_1_bnds
        if params_to_fit[1] == True:
            bnds += eshift_2_bnds
        if params_to_fit[2] == True:
            for i in range(int(ind_values)):
                if bnds_params[0][i] == True:
                    bnds += ec_bnds[i]
        if params_to_fit[3] == True:
            for i in range(int(ind_values)):
                if bnds_params[1][i] == True:
                    bnds += re_bnds[i]
        if params_to_fit[4] == True:
            for i in range(int(ind_values)):
                if bnds_params[2][i] == True:
                    bnds += rc_bnds[i]
        if params_to_fit[5] == True:
            for i in range(int(ind_values)):
                if bnds_params[3][i] == True:
                    bnds += alpha_bnds[i]
        if params_to_fit[6] == True:
            for i in range(int(ind_values)):
                if bnds_params[4][i] == True:
                    bnds += delta_bnds[i]
        if params_to_fit[7] == True:
            for i in range(int(ind_values)):
                if bnds_params[5][i] == True:
                    bnds += dr_bnds[i]
        if params_to_fit[8] == True:
            for i in range(num_elem):
                if bnds_params[6][i] == True:
                    bnds += cweight_bnds[i]
        if params_to_fit[9] == True:
            for i in range(num_elem):
                if bnds_params[6][i] == True:
                    bnds += beta_bnds[i]
        if params_to_fit[10] == True:
            for i in range(num_elem):
                if bnds_params[7][i] == True:
                    bnds += Asub_bnds[i]
        if params_to_fit[11] == True:
            bnds += Cmax_1_bnds
        if params_to_fit[12] == True:
            bnds += Cmin_1_bnds
        if params_to_fit[13] == True:
            bnds += Cmax_2_bnds
        if params_to_fit[14] == True:
            bnds += Cmin_2_bnds
        if params_to_fit[15] == True:
            for j in range((num_elem ** 3)-(2 * num_elem)):
                bnds += Cmax_12_bnds
        if params_to_fit[16] == True:
            for j in range((num_elem ** 3)-(2 * num_elem)):
                bnds += Cmin_12_bnds
        # print(bnds)
        # print(guess1)
        result = minimize(function, guess1, method='nelder-mead', bounds=bnds)

    else:
        bnds = []
        if params_to_fit[0] == True:
            bnds += eshift_1_bnds
        if params_to_fit[1] == True:
            bnds += eshift_2_bnds
        if params_to_fit[2] == True:
            bnds += ec_1_bnds
        if params_to_fit[3] == True:
            bnds += re_1_bnds
        if params_to_fit[4] == True:
            bnds += rc_1_bnds
        if params_to_fit[5] == True:
            bnds += alpha_1_bnds
        if params_to_fit[6] == True:
            bnds += delta_1_bnds
        if params_to_fit[7] == True:
            bnds += dr_1_bnds
        if params_to_fit[8] == True:
            bnds += cweight_1_bnds
        if params_to_fit[9] == True:
            bnds += beta_1_bnds
        if params_to_fit[10] == True:
            bnds += Asub_1_bnds
        if params_to_fit[11] == True:
            bnds += Cmax_1_bnds
        if params_to_fit[12] == True:
            bnds += Cmin_1_bnds

        guess1, temp_guess = initial_guess()
        # print('guess1 = '+str(guess1))

        result = minimize(function, guess1, method='nelder-mead', bounds = bnds)

    # print('!!!!!!!!!!!!!!!')
    print(result)
    write_best_input(params_to_fit, result.x, write_file)

main()
# write_input(params_to_fit, initial_guess()[0])
# a1, a2 = initial_guess()
# print(a1)
# print('---------------------')
# print(a2)
