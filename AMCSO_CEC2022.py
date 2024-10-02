# import packages
import os
from opfunu.cec_based.cec2022 import *
import numpy as np
from copy import deepcopy


PopSize = 200
DimSize = 100
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 30
MaxFEs = 1000 * DimSize

Pop = np.zeros((PopSize, DimSize))
Velocity = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)
curFEs = 0
FuncNum = 1
curIter = 0
MaxIter = int(MaxFEs / PopSize * 2)
phi = 0.1


# initialize the M randomly
def Initialization(func):
    global Pop, Velocity, FitPop
    Velocity = np.zeros((PopSize, DimSize))
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func(Pop[i])


def Check(indi):
    global LB, UB
    for i in range(DimSize):
        range_width = UB[i] - LB[i]
        if indi[i] > UB[i]:
            n = int((indi[i] - UB[i]) / range_width)
            mirrorRange = (indi[i] - UB[i]) - (n * range_width)
            indi[i] = UB[i] - mirrorRange
        elif indi[i] < LB[i]:
            n = int((LB[i] - indi[i]) / range_width)
            mirrorRange = (LB[i] - indi[i]) - (n * range_width)
            indi[i] = LB[i] + mirrorRange
        else:
            pass
    return indi


def calc_distance_matrix(Pop):
    PopSize = Pop.shape[0]
    DistMatrix = np.zeros((PopSize, PopSize))
    for i in range(PopSize):
        for j in range(i + 1, PopSize):
            DistMatrix[i][j] = np.linalg.norm(Pop[i] - Pop[j])
            DistMatrix[j][i] = DistMatrix[i][j]
    return DistMatrix


def CSO_with_nearest(func):
    global Pop, Velocity, FitPop, phi

    PopSize = Pop.shape[0]
    DistMatrix = calc_distance_matrix(Pop)
    compared = np.zeros(PopSize, dtype=bool)
    
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    Xmean = np.mean(Pop, axis=0)

    for i in range(PopSize):
        if compared[i]:
            continue
        distances = DistMatrix[i]
        Kp=51
        nearest_indices = np.argsort(distances)[1:Kp]
        idx2 = np.random.choice(nearest_indices)
        compared[i] = True
        compared[idx2] = True

        if FitPop[i] < FitPop[idx2]:
            Off[i] = deepcopy(Pop[i])
            FitOff[i] = FitPop[i]
            Velocity[idx2] = np.random.rand(DimSize) * Velocity[idx2] + np.random.rand(DimSize) * (
                Pop[i] - Pop[idx2]) + phi * (Xmean - Pop[idx2])
            Off[idx2] = Pop[idx2] + Velocity[idx2]
            Off[idx2] = Check(Off[idx2])
            FitOff[idx2] = func(Off[idx2])
        else:
            Off[idx2] = deepcopy(Pop[idx2])
            FitOff[idx2] = FitPop[idx2]
            Velocity[i] = np.random.rand(DimSize) * Velocity[i] + np.random.rand(DimSize) * (
                Pop[idx2] - Pop[i]) + phi * (Xmean - Pop[i])
            Off[i] = Pop[i] + Velocity[i]
            Off[i] = Check(Off[i])
            FitOff[i] = func(Off[i])

    Pop[:] = deepcopy(Off)
    FitPop[:] = deepcopy(FitOff)


def RunCSO(func):
    global FitPop, curIter, TrialRuns, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        BestList = []
        curIter = 0
        np.random.seed(2023 + 23 * i)
        Initialization(func)
        BestList.append(min(FitPop))
        while curIter < MaxIter:
            CSO_with_nearest(func)
            curIter += 1
            BestList.append(min(FitPop))
        All_Trial_Best.append(BestList)
    np.savetxt("./AMCSO/AMCSO_Data/CEC2022/F" + str(FuncNum+1) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, MaxFEs, MaxIter, Pop, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize * 2)
    LB = [-100] * dim
    UB = [100] * dim

    CEC2022 = [F12022(Dim), F22022(Dim), F32022(Dim), F42022(Dim), F52022(Dim), F62022(Dim),
               F72022(Dim), F82022(Dim), F92022(Dim), F102022(Dim), F112022(Dim), F122022(Dim)]

    for i in range(len(CEC2022)):
        FuncNum = i
        RunCSO(CEC2022[i].evaluate)


if __name__ == "__main__":
    if not os.path.exists('./AMCSO/AMCSO_Data/CEC2022'):
        os.makedirs('./AMCSO/AMCSO_Data/CEC2022')
    Dims = [10, 20]
    for Dim in Dims:
        main(Dim)
