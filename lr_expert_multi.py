def sample():
    return "Hello Expert Mode Running.....!"
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import string
import random



def lr_expert_multi(mydata, file_ext, Fit_Intercept_Val, copy_X_Val, n_jobs_Val, positive_Val):
    print("lr_expert_multi Normal Mode Running.....!")
    print(mydata.head())
    print(mydata.describe())
    if file_ext == ".txt":
        print("HAS==========>", mydata[0].iloc[0])
        myheadval = mydata[0].iloc[0]
        hasHeader = isinstance(myheadval, np.float64) or isinstance(myheadval, np.int64)
        print("TXT HAS HEADER", hasHeader)
        # if it has header return Flase other wise True
    elif file_ext == ".csv":
        firstval = list(mydata.columns)[0]
        hasHeader = isinstance(firstval, np.float64)
        print("CSV HAS HEADER", hasHeader)
        # if it has header return Flase other wise True
    if not hasHeader:
        pass
        # has Header
        # X_name, Y_name these are used for plotting graph
        # X_name = mydata.columns[0]
        # Y_name = mydata.columns[1]
        # # for Model inputs
        # X = mydata.iloc[1:, :-1]
        # Y = mydata.iloc[1:, mydata.shape[1]]
        # print("has Header X and Y shapes", X.shape, Y.shape)
    else:
        # not has Header
        # X_name, Y_name these are used for plotting graph
        X_name = "X-Axis" 
        Y_name = "Y-Axis"
        print("I'm working...................!")
        # for Model inputs
        colnum = mydata.shape[1]
        X = mydata.iloc[:, 0:2]
        
        print("shape", )
        Y = mydata.iloc[:, -1]
        print("I'm working2222...................!")
        print("not has HeaderX and Y shapes", X.shape, Y.shape)
        print("not has HeaderX and Y shapes", X, Y) 

    print(X_name, Y_name, "=======")

    # print(X, Y, "=======")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=1)
    lr_model = LinearRegression(fit_intercept=Fit_Intercept_Val, normalize='deprecated', copy_X=copy_X_Val, n_jobs=n_jobs_Val, positive=positive_Val)
    
    lr_expert_multi_final = lr_model  # for user input

    lr_model.fit(X_train, Y_train)
    Y_pred = lr_model.predict(X_test)
    # Y_pred = lr_model.predict(np.array([[1203,3]]))
    
    result = Y_pred[0]
    print("lr model plotting")

    # Plot outputs
    print(X_test.shape)
    print(Y_test.shape)
    print(Y_pred.shape)
    print(X_test)
    print("---------------------------")
    print(Y_test)
    print("---------------------------")
    print(Y_pred)
    plt.xlabel(X_name, fontsize=20)
    plt.ylabel(Y_name, fontsize=20)

    # plt.scatter(X_test, Y_test,  color='black')
    plt.plot(X_test, Y_pred, color='red', linewidth=3)

    plt.xticks(())
    plt.yticks(())
    
    fig1 = plt.gcf()
    plt.draw()


    ranimg = ''.join(random.choices(string.ascii_lowercase + string.digits, k = 6))
    imgpath = 'static\\linear_uni_expert_pic' + ranimg + ".png"
    imgpath_html = 'static\linear_uni_expert_pic' + ranimg + ".png"

    fig1.savefig(imgpath, dpi=100)
    # plt.show()
    print("lr model return")
    from sklearn import metrics
    Mean_Absolute_Error = metrics.mean_absolute_error(Y_test, Y_pred)
    Mean_Squared_Error = metrics.mean_squared_error(Y_test, Y_pred)
    Root_Mean_Squared_Error = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
    print('Mean Absolute Error:', Mean_Absolute_Error)
    print('Mean Squared Error:', Mean_Squared_Error)
    print('Root Mean Squared Error:', Root_Mean_Squared_Error)
    from sklearn.metrics import r2_score
    score = r2_score(Y_test, Y_pred)
    return result, imgpath_html, Mean_Absolute_Error, Mean_Squared_Error, Root_Mean_Squared_Error, score, lr_expert_multi_final
    # return result, imgpath_html, 1, 1, 1