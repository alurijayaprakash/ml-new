# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
import random

# # Importing the dataset
# dataset = pd.read_csv('Social_Network_Ads.csv')

def lg_model_expert(mydata, file_ext):
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
    X = mydata.iloc[:, [2,3]].values
    y = mydata.iloc[:, 4].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)



    #fitting logisitic regression to the training set
    from sklearn.linear_model import LogisticRegression
    classifier=LogisticRegression(random_state=0)

    lg_expert_final = classifier.fit(X_train,y_train)


    #Predicting the test set results
    y_pred=lg_expert_final.predict(X_test)

    result = y_pred

    #making the confusion matrix
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y_test, y_pred)


    # Visualising the Training set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
            np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, lg_expert_final.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Multi Varient (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    # plt.show()

    # Visualising the Test set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
            np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, lg_expert_final.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                    alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Multi Varient (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

    fig1 = plt.gcf()
    ranimg = ''.join(random.choices(string.ascii_lowercase + string.digits, k = 6))
    imgpath = 'static\\logistic_multi_expert_pic' + ranimg + ".png"
    imgpath_html = 'static\logistic_multi_expert_pic' + ranimg + ".png"

    fig1.savefig(imgpath, dpi=100)
    # plt.show()
    print("lg model return")
    return result, imgpath_html, lg_expert_final
