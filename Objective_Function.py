from numpy.random import rand

from Global_Vars import Global_Vars
import numpy as np

def LOA(initsol, fname, xmin, xmax, Max_iter):
    pass

def Obj_fun(Soln):
    Feat = Global_Vars.Feat
    Target = Global_Vars.Target
    learnperc = Global_Vars.learnperc
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 1:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i]).astype('uint8')
            # Eval = Model_Ensemble(Feat, Target, sol)
            Fitn[i] = 1 / rand() #(Eval[4] + Eval[7])
        return Fitn
    else:
        sol = np.round(Soln).astype('uint8')
        #Eval = Model_Ensemble(Feat, Target, learnperc, sol)
        Fitn = 1 / rand() #(Eval[4] + Eval[7])
        return Fitn


def Obj_fun_CLS(Soln):
    Feat = Global_Vars.Feat
    Target = Global_Vars.Target
    learnperc = Global_Vars.learnperc
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 1:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i]).astype('uint8')
            Eval = Model_Ensemble(Feat, Target, sol)
            Fitn[i] = 1 / (Eval[4] + Eval[7])
        return Fitn
    else:
        sol = np.round(Soln).astype('uint8')
        Eval = Model_Ensemble(Feat, Target, learnperc, sol)
        Fitn = 1 / (Eval[4] + Eval[7])
        return Fitn