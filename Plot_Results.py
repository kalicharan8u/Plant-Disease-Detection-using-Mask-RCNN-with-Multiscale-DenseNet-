from itertools import cycle
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn import metrics


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plot_results():
    # matplotlib.use('TkAgg')
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 4, 5, 7, 8, 9]
    Algorithm = ['TERMS', 'DO', 'EOO', 'AVOA', 'LO', 'AVOA+LO']
    Classifier = ['TERMS', 'Resnet', 'VGG16', 'Densenet', 'Multiscale densenet', 'PROPOSED']
    value = eval[0, 4, :, 4:]

    Batch_Size = [4, 8, 16, 32, 48, 64]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]

            plt.plot(Batch_Size, Graph[:, 0], color='r', linestyle='dashed', linewidth=3, marker='v',
                     markerfacecolor='b', markersize=16,
                     label="DO-DAA-MDeNet")
            plt.plot(Batch_Size, Graph[:, 1], color='g', linestyle='dashed', linewidth=3, marker='s',
                     markerfacecolor='red', markersize=12,
                     label="EOO-DAA-MDeNet")
            plt.plot(Batch_Size, Graph[:, 2], color='b', linestyle='dashed', linewidth=3, marker='>',
                     markerfacecolor='green', markersize=16,
                     label="AVOA-DAA-MDeNet")
            plt.plot(Batch_Size, Graph[:, 3], color='c', linestyle='dashed', linewidth=3, marker='D',
                     markerfacecolor='cyan', markersize=12,
                     label="LO-DAA-MDeNet")
            plt.plot(Batch_Size, Graph[:, 4], color='k', linestyle='dashed', linewidth=3, marker='p',
                     markerfacecolor='black', markersize=16,
                     label="AVLO-DAA-MDeNet")
            plt.xticks(Batch_Size, ('4', '8', '16', '32', '48', '64'))
            plt.xlabel('Batch Size')
            plt.ylabel(Terms[Graph_Term[j]])
            # plt.ylim([80, 100])

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            # plt.ylim([0.8, 1])
            path1 = "./Results/Dataset_%s_%s_line.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            # ax = plt.axes(projection="3d")
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(6)
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="ResNet")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="VGG16")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="DenseNet")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="MDeNet")
            ax.bar(X + 0.40, Graph[:, 9], color='k', width=0.10, label="AVLO-DAA-MDeNet")
            plt.xticks(X + 0.10, ('4', '8', '16', '32', '48', '64'))
            plt.xlabel('Batch Size')
            plt.ylabel(Terms[Graph_Term[j]])

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            # plt.ylim([0.8, 1])
            path1 = "./Results/Dataset_%s_%s_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


def plot_results_1():
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 4, 5, 7, 8, 9]
    Algorithm = ['TERMS', 'DO', 'EOO', 'AVOA', 'LO', 'AVOA+LO']
    Classifier = ['TERMS', 'Resnet', 'VGG16', 'Densenet', 'Multiscale densenet', 'PROPOSED']
    for i in range(eval.shape[0]):
        value = eval[0, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('-------------------------------------------------- Dataset - ', i + 1, ' - 75%-Algorithm Comparison ',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Dataset - ', i + 1, ' - 75%-Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
            learnperc = [35, 45, 55, 65, 75, 85]
            plt.plot(learnperc, Graph[:, 0], color='r', linewidth=3, marker='x', markerfacecolor='b', markersize=16,
                     label="DO-DAA-MDeNet")
            plt.plot(learnperc, Graph[:, 1], color='g', linewidth=3, marker='D', markerfacecolor='red', markersize=12,
                     label="EOO-DAA-MDeNet")
            plt.plot(learnperc, Graph[:, 2], color='b', linewidth=3, marker='x', markerfacecolor='green',
                     markersize=16,
                     label="AVOA-DAA-MDeNet")
            plt.plot(learnperc, Graph[:, 3], color='c', linewidth=3, marker='D', markerfacecolor='cyan', markersize=12,
                     label="LO-DAA-MDeNet")
            plt.plot(learnperc, Graph[:, 4], color='k', linewidth=3, marker='x', markerfacecolor='black',
                     markersize=16,
                     label="AVLO-DAA-MDeNet")
            plt.xticks(learnperc, ('35', '45', '55', '65', '755', '88'))
            plt.xlabel('Learning Percentage')
            plt.ylabel(Terms[Graph_Term[j]])
            # plt.ylim([80, 100])

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_line-learnperc.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            # ax = plt.axes(projection="3d")
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(6)
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="ResNet")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="VGG16")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="Densenet")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="MDeNett")
            ax.bar(X + 0.40, Graph[:, 9], color='k', width=0.10, label="AVLO-DAA-MDeNet")
            plt.xticks(X + 0.10, ('35', '45', '55', '65', '755', '88'))
            plt.xlabel('Learning Percentage')
            plt.ylabel(Terms[Graph_Term[j]])

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_bar-learnperc.png" % (i + 1, Terms[Graph_Term[j]])
            # path1 = "./Results/_%s_bar.png" % (Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'DO', 'EOO', 'AVOA', 'LO', 'PROPOSED']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(1):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Report ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='DO-DAA-MDeNet')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='EOO-DAA-MDeNet')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='AVOA-DAA-MDeNet')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='LO-DAA-MDeNet')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='AVLO-DAA-MDeNet')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Plot_ROC_Curve():
    lw = 2
    cls = ['TERMS', 'Resnet', 'VGG16', 'Densenet', 'MDeNet', 'AVLO-DAA-MDeNet']

    # Classifier = ['TERMS', 'Xgboost', 'DT', 'NN', 'FUZZY', 'KNN', 'PROPOSED']
    for a in range(1):  # For 5 Datasets
        # Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')
        Actual = np.load('Targets_1.npy', allow_pickle=True)

        colors = cycle(["blue", "darkorange", "cornflowerblue", "deeppink", "black"])  # "aqua",
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i], )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


no_of_dataset = 10


def Image_Results():
    Plant = ['Apple', 'Cherry', 'Citrus', 'Corn', 'Grape', 'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato']
    for n in range((no_of_dataset)):
        Images = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        SegImg1 = np.load('Fuzzy_' + str(n + 1) + '.npy', allow_pickle=True)
        Image = [7]
        for i in range(len(Image)):
            fig, ax = plt.subplots(2, 1)
            plt.suptitle(Plant[n], fontsize=20)
            plt.subplot(1, 2, 1)
            plt.title('Original Image')
            plt.imshow(Images[Image[i]])
            plt.subplot(1, 2, 2)
            plt.title('Segmented(Abnormal Part) Image')
            # gt1 = SegImg1[Image[i]]
            plt.imshow(SegImg1[Image[i]])
            # path1 = "./Results_1/Class5/Dataset_%simage.png" % (i + 1)
            # plt.savefig(path1)
            plt.show()
            cv.imwrite('./Results/Image_Results/Plant' + str(n + 1) + '-orig-' + str(i + 1) + '.png', Images[Image[i]])
            cv.imwrite('./Results/Image_Results/Plant' + str(n + 1) + '-Seg-Abnormal' + str(i + 1) + '.png',
                       SegImg1[Image[i]])


def Sample_images():
    Plant = ['Apple', 'Cherry', 'Citrus', 'Corn', 'Grape', 'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato']
    for n in range((no_of_dataset)):
        Images = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        Image = [8, 9, 10, 11, 12, 13]
        # for i in range(len(Image)):
        fig, ax = plt.subplots(2, 3)
        plt.suptitle("Sample Images from " + Plant[n], fontsize=20)
        plt.subplot(2, 3, 1)
        plt.title('Image-1')
        plt.imshow(Images[Image[0]])
        plt.subplot(2, 3, 2)
        plt.title('Image-2')
        plt.imshow(Images[Image[1]])
        plt.subplot(2, 3, 3)
        plt.title('Image-3')
        plt.imshow(Images[Image[2]])
        plt.subplot(2, 3, 4)
        plt.title('Image-4')
        plt.imshow(Images[Image[3]])
        plt.subplot(2, 3, 5)
        plt.title('Image-5')
        plt.imshow(Images[Image[4]])
        plt.subplot(2, 3, 6)
        plt.title('Image-6')
        plt.imshow(Images[Image[5]])
        # path1 = "./Results_1/Class5/Dataset_%simage.png" % (i + 1)
        # plt.savefig(path1)
        plt.show()
        # cv.imwrite('./Results/Image_Results/Plant' + str(n + 1) + '-orig-' + str(i + 1) + '.png', Images[Image[i]])
        # cv.imwrite('./Results/Image_Results/Plant' + str(n + 1) + '-Seg-Abnormal' + str(i + 1) + '.png',
        #            SegImg1[Image[i]])


import seaborn as sns


def Plot_Confusion():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    no_of_Dataset = 10
    for n in range(no_of_Dataset):
        ax = plt.subplot()
        cm = confusion_matrix(np.asarray(Actual[n]).argmax(axis=1), np.asarray(Predict[n]).argmax(axis=1))
        sns.heatmap(cm, annot=True, fmt='g',
                    ax=ax)
        path = "./Results/Confusion_%s.png" % (n + 1)
        plt.title('Accuracy')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig(path)
        plt.show()


def plot_results_Accuracy():
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all_ACC.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0]
    Algorithm = ['TERMS', 'DO', 'EOO', 'AVOA', 'LO', 'AVOA+LO']
    Classifier = ['Resnet', 'VGG16', 'Densenet', 'Multiscale densenet', 'PROPOSED']
    Classes = ['Apple', 'Cherry', 'Citrus', 'Corn', 'Grape', 'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato']
    # for i in range(eval.shape[0]):
    value = eval[0:11, 5, 5:, 4]
    Table = PrettyTable()
    Table.add_column('Classes', Classes)
    Table.add_column(Classifier[0], value[:, 0])
    Table.add_column(Classifier[1], value[:, 1])
    Table.add_column(Classifier[2], value[:, 2])
    Table.add_column(Classifier[3], value[:, 3])
    Table.add_column(Classifier[4], value[:, 4])
    print('-------------------------------------------------- Accuracy Report ',
          '--------------------------------------------------')
    print(Table)


def plot_Segmentation_results():
    Eval_all = np.load('Eval_all_Segmentation.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD', 'VARIANCE']
    Algorithm = ['TERMS', 'DO', 'EOO', 'AVOA', 'LO', 'AVOA+LO']
    Classifier = ['TERMS', 'Resnet', 'VGG16', 'Densenet', 'Multiscale densenet', 'PROPOSED']
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]

        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            X = np.arange(stats.shape[2])

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 0, :] * 100, color='r', width=0.10, label="DO-DAA-MRCNN")
            ax.bar(X + 0.10, stats[i, 1, :] * 100, color='g', width=0.10, label="EOO-DAA-MRCNN")
            ax.bar(X + 0.20, stats[i, 2, :] * 100, color='b', width=0.10, label="AVOA-DAA-MRCNN")
            ax.bar(X + 0.30, stats[i, 3, :] * 100, color='m', width=0.10, label="LO-DAA-MRCNN")
            ax.bar(X + 0.40, stats[i, 4, :] * 100, color='k', width=0.10, label="AVLO-DAA-MRCNN")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16),
                       ncol=2, fancybox=True, shadow=True)
            # plt.legend(loc=10)
            path1 = "./Results/Dataset_%s_%s_alg-segmentation.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 5, :] * 100, color='r', width=0.10, label="CNN")
            ax.bar(X + 0.10, stats[i, 6, :] * 100, color='g', width=0.10, label="MCNN")
            ax.bar(X + 0.20, stats[i, 7, :] * 100, color='m', width=0.10, label="MRCNN")
            ax.bar(X + 0.30, stats[i, 8, :] * 100, color='k', width=0.10, label="AVLO-DAA-MRCNN")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            # plt.legend(loc=10)
            path1 = "./Results/Dataset_%s_%s_met-segmentation.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path1)
            plt.show()


if __name__ == '__main__':
    # plot_results()
    # plotConvResults()
    # Plot_ROC_Curve()
    plot_results_1()
    # Plot_Confusion()
    # plot_results_Accuracy()
    # Image_Results()
    # Sample_images()
    # plot_Segmentation_results()
