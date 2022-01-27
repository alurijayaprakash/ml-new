def sample():
    return "Hello LG Normal Mode Running.....!"
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import string
import random



def lg_model_normal(mydata, file_ext):
    print("lg_model_normal Normal Mode Running.....!")
    print(mydata.head())
    if file_ext == ".txt":
        print("HAS==========>", mydata[0].iloc[0])
        myheadval = mydata[0].iloc[0]
        hasHeader = isinstance(myheadval, np.float64)
        print("TXT HAS HEADER", hasHeader)
        # if it has header return Flase other wise True
    elif file_ext == ".csv":
        firstval = list(mydata.columns)[0]
        hasHeader = isinstance(firstval, np.float64)
        print("CSV HAS HEADER", hasHeader)
        # if it has header return Flase other wise True

    if not hasHeader:
        X_name = mydata.columns[0]
        Y_name = mydata.columns[1]
        X = mydata[X_name].values.reshape(-1,1)
        Y = mydata[Y_name]
    else:
        X_name = "X-Axis" 
        Y_name = "Y-Axis"
        X = mydata[0].values.reshape(-1,1)
        Y = mydata[1] 
    print(X_name, Y_name, "=======")

    # print(X, Y, "=======")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
    lg_model = LogisticRegression()
    lg_model_final = lg_model  # for user input
    lg_model.fit(X_train, Y_train)
    Y_pred = lg_model.predict(X_test)
    # Y_pred = lg_model.predict(np.array([17.5]).reshape(1, 1))
    
    result = Y_pred[0]
    print("lg model plotting & result ", result)


    # Plot outputs
    # print(X_test.shape) # numpy
    # print("lg model plotting")
    # print(Y_test.shape) #panda
    # print("lg model plotting")
    # print(Y_pred) # numpy
    # print("lg model plotting")
    # plt.xlabel(X_name, fontsize=20)
    # plt.ylabel(Y_name, fontsize=20)

    # plt.scatter(X_test, Y_test,  color='black')
    # plt.plot(X_test, Y_pred, color='red', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())
    
    # fig1 = plt.gcf()
    # plt.draw()



    fig1 = plt.gcf()
    ranimg = ''.join(random.choices(string.ascii_lowercase + string.digits, k = 6))
    imgpath = 'static\\logistic_uni_normal_pic' + ranimg + ".png"
    imgpath_html = 'static\logistic_uni_normal_pic' + ranimg + ".png"

    fig1.savefig(imgpath, dpi=100)
    # plt.show()
    print("lg model return")

    print("lg model plotting")
    # from sklearn import metrics
    # Mean_Absolute_Error = metrics.mean_absolute_error(Y_test, Y_pred)
    # Mean_Squared_Error = metrics.mean_squared_error(Y_test, Y_pred)
    # Root_Mean_Squared_Error = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
    # print('Mean Absolute Error:', Mean_Absolute_Error)
    # print('Mean Squared Error:', Mean_Squared_Error)
    # print('Root Mean Squared Error:', Root_Mean_Squared_Error)
    # from sklearn.metrics import r2_score
    # score = r2_score(Y_test, Y_pred)
    # return result, imgpath_html, Mean_Absolute_Error, Mean_Squared_Error, Root_Mean_Squared_Error, score, lg_model_final
    return result, imgpath_html, lg_model_final