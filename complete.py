import matplotlib.pyplot as plt #Used to create graphs
import seaborn as sns #Also used to create graphs
import numpy as np #Used for working with arrays which will be
                   #useful when working with the data
import pandas as pd #Used to open CSV files
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #Used to encode data
from sklearn.preprocessing import StandardScaler #Used to scale the data


#Used to assign a numerical value to each feature
#in the data that we are using to find the most
#important and useful features.
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Used to create test and train data sets.
from sklearn.model_selection import train_test_split

#Used to predict the status
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

#Used to predict the salary
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


def getPred(data):
    # Create the dataframe from the data
    #userdf = pd.read_csv("./TestData.csv")
    userdf = pd.DataFrame.from_dict(data)
    formatdf = pd.read_csv("./FormattingData.csv")
    #Add The formatting datset to the user dataset so that the
    #data can be easily cleaned.
    userdf = pd.concat([userdf, formatdf], ignore_index=True)
    #Clean the data
    userdf = cleandata(userdf)
    #Drop the format formatting dataset.
    userdf = userdf.drop(index=[1,2,3])
    

    """Get the dataset"""
    #Read the data from the CSV file
    df = pd.read_csv("./Data.csv")

    #Remove the sl_no column
    df = df.drop(['sl_no'], axis=1)

    #Fill the null values with 0
    df['salary'] = df['salary'].fillna(0)

    df = cleandata(df)


    """Cleaning the data"""
    #Create variables to predict a student's status
    x1=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,12,13,14,15,16,17]] #independent variable for status prediction
    y1=df.iloc[:,10] #dependent variable for status prediction
    x2=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17]] #independent variable for salary prediction
    y2=df.iloc[:,11] #dependent variable for salary prediction

    #Select the factors that will be useful and drop the ones that aren't for the y1 data.
    y1=pd.DataFrame(y1)
    #Use SelectKBest to select the top 10 features.
    bestfeatures = SelectKBest(score_func=chi2, k=12)
    fit = bestfeatures.fit(x1,y1)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x1.columns)
    #Concatenate the two dataframes to have better visualization.
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns

    #Select the factors that will be useful and drop the ones that aren't for the y2 data.
    y2=pd.DataFrame(y2)
    #Use SelectKBest to select the top 10 features.
    bestfeatures = SelectKBest(score_func=chi2, k=12)
    fit = bestfeatures.fit(x2,y2)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x2.columns)
    #Concatenate the two dataframes to have better visualization.
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns



    #Create two x and a y data sets to train and test the data.
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size = 0.20, random_state = 42)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size = 0.20, random_state = 42)




    """Scaling the data"""
    #Transofrm train1 and test1
    sc_x=StandardScaler()
    x_train1=sc_x.fit_transform(x_train1)
    x_test1=sc_x.transform(x_test1)

    #Transform train2 and test2
    sc_x1=StandardScaler()
    sc_y=StandardScaler()
    x_train2=sc_x1.fit_transform(x_train2)
    x_test2=sc_x1.transform(x_test2)
    y_train2=sc_y.fit_transform(y_train2)




    """Remove all employees whose salary is 0"""
    #Remove teh X value and store it
    X_extra = df.drop(df[df['salary'] == 0].index)

    y_extra = X_extra.iloc[:,11].values

    X_extra = X_extra.drop(['salary'],axis=1)

    y_extra = y_extra.reshape(-1,1)    


    """Making a prediction"""
    #Create new varaibles to train and test the data.
    X_train_extra, X_test_extra, y_train_extra, y_test_extra = train_test_split(X_extra, y_extra, test_size = 0.2, random_state = 42)

    #Use the user data to make tests
    X_test_extra = userdf.drop(['salary'],axis=1)
    y_test_extra = userdf.iloc[:,11].values.reshape(-1,1)

    sc_X_extra=StandardScaler()
    X_train_extra=sc_X_extra.fit_transform(X_train_extra)
    X_test_extra=sc_X_extra.transform(X_test_extra)
    sc_y_extra = StandardScaler()
    y_train_extra = sc_y_extra.fit_transform(y_train_extra)
    

    #Multiple Linear Regression
    mlr_extra = LinearRegression()
    mlr_extra.fit(X_train_extra, y_train_extra)
    #Predicting the Test set results
    y_pred_mlr_extra = sc_y_extra.inverse_transform(mlr_extra.predict(X_test_extra))

    #Support Vector Regression
    svr = SVR(kernel='poly')
    svr.fit(X_train_extra, y_train_extra)
    #Predicting a new result
    y_pred_svr = sc_y_extra.inverse_transform(svr.predict(X_test_extra))

    #Random Forest Regression
    rfr = RandomForestRegressor(n_estimators = 300, random_state = 42)
    rfr.fit(X_train_extra, y_train_extra)
    #Predicting a new result
    y_pred_rfr = sc_y.inverse_transform(rfr.predict(X_test_extra))

    #Return the data
    data = pd.DataFrame({'MLR_Salary':list(y_pred_mlr_extra),'SVR_Salary':list(y_pred_svr),'RFR_Salary':list(y_pred_rfr)})
    return y_pred_mlr_extra[0][0], y_pred_svr[0], y_pred_rfr[0]


def cleandata(df):
    #Check for unique values in the data.
    uniqueValues = df.nunique()

    #Convert Nan type to string or float
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].fillna(' ')
        else:
            df[col] = df[col].fillna(0)

    #Create obects to encode the data
    labelencoder = LabelEncoder()
    onehotencoder = OneHotEncoder()

    #Encode the data and send that encoded data to the table.
    df['gender'] = labelencoder.fit_transform(df['gender'])
    df['ssc_b'] = labelencoder.fit_transform(df['ssc_b'])
    df['hsc_b'] = labelencoder.fit_transform(df['hsc_b'])
    df['hsc_s'] = labelencoder.fit_transform(df['hsc_s'])
    df['degree_t'] = labelencoder.fit_transform(df['degree_t'])
    df['workex'] = labelencoder.fit_transform(df['workex'])
    df['specialisation'] = labelencoder.fit_transform(df['specialisation'])
    df['status'] = labelencoder.fit_transform(df['status'])

    #Get new data based on the values in hsc_s and degree_t
    enc = pd.DataFrame(onehotencoder.fit_transform(df[['hsc_s']]).toarray())
    enc1 = pd.DataFrame(onehotencoder.fit_transform(df[['degree_t']]).toarray())
    
    #Add the new data to the main table.
    df = df.join(enc)
    df = df.rename(columns={0:'hsc_s1',1:'hsc_s2',2:'hsc_s3'})
    df = df.join(enc1)
    df = df.rename(columns={0:'degree_t1',1:'degree_t2',2:'degree_t3'})

    #Drop the rows that are no longer needed
    df = df.drop(['hsc_s'],axis=1)
    df = df.drop(['degree_t'],axis=1)

    #Return the dataframe
    return df
