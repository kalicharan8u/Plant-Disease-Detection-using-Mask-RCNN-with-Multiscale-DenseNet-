import numpy as np
import os
import cv2 as cv
import random as rn
from AVOA import AVOA
from DOX import DOX
from EOO import EOO
from Global_Vars import Global_Vars
from Model_DAFCNN import Model_DAFCNN
from Model_DenseNet import Model_DenseNet
from Model_Resnet import Model_Resnet
from Model_VGG16 import Model_VGG16
from Objective_Function import Obj_fun, Obj_fun_CLS, LOA
from Proposed import Proposed


def Read_Image(Filename):
    image = cv.imread(Filename)
    image = np.uint8(image)
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (512, 512))
    return image


def Read_Images(directory_name):
    Fold_Array = os.listdir(directory_name)
    Images = []
    Target = []
    iter = 1
    flag = 0
    for i in range(len(Fold_Array)):
        Img_Array = os.listdir(directory_name + Fold_Array[i])
        for j in range(len(Img_Array)):
            print(i, j)
            image = Read_Image(directory_name + Fold_Array[i] + '/' + Img_Array[j])
            Images.append(image)
            if Fold_Array[i][len(Fold_Array[i]) - 7:] == 'healthy':
                Target.append(0)
            else:
                flag = 1
                Target.append(iter)
        if flag == 1:
            iter = iter + 1
    return Images, Target


no_of_dataset = 10


# Read Dataset
an = 0
if an == 1:
    Directory = './Dataset/'
    Dataset_List = os.listdir(Directory)
    for n in range(len(Dataset_List)):
        Images, Target = Read_Images(Directory + Dataset_List[n] + '/')
        np.save('Images_' + str(n + 1) + '.npy', Images)
        np.save('Target_' + str(n + 1) + '.npy', Target)


# Assemble Target for Classification
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        uniq = np.unique(Target)

        Tar = np.zeros((len(Target), len(uniq)))
        for j in range(len(uniq)):
            Index = np.where(Target == uniq[j])
            Tar[Index[0], j] = 1
        np.save('Targets_' + str(n + 1) + '.npy', Tar)

# Optimization for DAFCNN
an = 0
if an == 1:
    for n in range(no_of_dataset):
        BestSol = []
        Images = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        Tar = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Images = Images
        Global_Vars.Target = Tar
        Npop = 10
        Chlen = 3  # Here we optimized Hidden Neuron Count, No of epoches, Steps per epoch in Transunet3+
        xmin = np.matlib.repmat([5, 5, 300], Npop, 1)
        xmax = np.matlib.repmat([255, 50, 1000], Npop, 1)
        initsol = np.zeros(xmax.shape)
        for p1 in range(Npop):
            for p2 in range(Chlen):
                initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
        fname = Obj_fun
        Max_iter = 25

        print("DOX...")
        [bestfit1, fitness1, bestsol1, time1] = DOX(initsol, fname, xmin, xmax, Max_iter)  # DOX

        print("EOO...")
        [bestfit2, fitness2, bestsol2, time2] = EOO(initsol, fname, xmin, xmax, Max_iter)  # EOO

        print("AVOA...")
        [bestfit3, fitness3, bestsol3, time3] = AVOA(initsol, fname, xmin, xmax, Max_iter)  # AVOA

        print("LOA...")
        [bestfit4, fitness4, bestsol4, time4] = LOA(initsol, fname, xmin, xmax, Max_iter)  # LOA

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

        BestSol.append([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])  #

        np.save('BestSol_seg.npy', BestSol)

# Segmentation by DAFCNN
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Images = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        GT = np.load('GT_' + str(n + 1) + '.npy', allow_pickle=True)
        Bestsol = np.load('BestSol_seg' + str(n + 1) + '.npy', allow_pickle=True)
        Seg = []
        for i in range(len(Images)):
            sol = np.round(Bestsol[i, :]).astype(np.int16)
            Image = Model_DAFCNN(Images[i], sol)
            Seg.append(Image)
        np.save('Segmented_' + str(n + 1) + '.npy', Seg)


# Optimization for DAFCNN
an = 0
if an == 1:
    for n in range(no_of_dataset):
        BestSol = []
        Images = np.load('Segmented_' + str(n + 1) + '.npy', allow_pickle=True)
        Tar = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Images = Images
        Global_Vars.Target = Tar
        Npop = 10
        Chlen = 3  # Here we optimized Hidden Neuron Count, No of epoches, Steps per epoch in Transunet3+
        xmin = np.matlib.repmat([5, 5, 300], Npop, 1)
        xmax = np.matlib.repmat([255, 50, 1000], Npop, 1)
        initsol = np.zeros(xmax.shape)
        for p1 in range(Npop):
            for p2 in range(Chlen):
                initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
        fname = Obj_fun_CLS
        Max_iter = 25

        print("DOX...")
        [bestfit1, fitness1, bestsol1, time1] = DOX(initsol, fname, xmin, xmax, Max_iter)  # DOX

        print("EOO...")
        [bestfit2, fitness2, bestsol2, time2] = EOO(initsol, fname, xmin, xmax, Max_iter)  # EOO

        print("AVOA...")
        [bestfit3, fitness3, bestsol3, time3] = AVOA(initsol, fname, xmin, xmax, Max_iter)  # AVOA

        print("LOA...")
        [bestfit4, fitness4, bestsol4, time4] = LOA(initsol, fname, xmin, xmax, Max_iter)  # LOA

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

        BestSol.append([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])  #

        np.save('BestSol_seg.npy', BestSol)


# Classification
an = 0
if an == 1:
    Feat = np.load('Seg_Img.npy', allow_pickle=True)
    Tar = np.load('Target.npy', allow_pickle=True)
    bests = np.load('bestsol.npy', allow_pickle=True)
    Eval_all = []
    pern = [0.35, 0.55, 0.65, 0.75, 0.85]
    for m in range(len(pern)):  # for all learning percentage
        EVAL = np.zeros((10, 14))
        per = round(len(Feat) * (pern[m]))
        Train_Data = Feat[:per, :, :]
        Train_Target = Tar[:per, :]
        Test_Data = Feat[per:, :, :]
        Test_Target = Tar[per:, :]
        for j in range(bests.shape[0]):  # for all algorithms
            soln = bests[j]
            EVAL[j, :], pred = Model_DenseNet(Train_Data, Train_Target, Test_Data, Test_Target,
                                               soln)  # with Optimization ensemble
        EVAL[5, :], pred = Model_Resnet(Train_Data, Train_Target, Test_Data, Test_Target)  # Resnet Model
        EVAL[6, :], pred = Model_VGG16(Train_Data, Train_Target, Test_Data, Test_Target)  # Inception model
        EVAL[7, :], pred = Model_DenseNet(Train_Data, Train_Target, Test_Data, Test_Target)  # Mobilenet model
        EVAL[8, :] = Model_DenseNet(Train_Data, Train_Target, Test_Data,
                                     Test_Target)  # without Optimization ensemble
        EVAL[9, :] = EVAL[4, :]  # with Optimization ensemble
        Eval_all.append(EVAL)
    np.save('Eval_all.npy', Eval_all)
