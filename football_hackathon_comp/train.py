from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from tables import Column
import xgboost as xg
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
def categorical_encode(x):
      #encode to numerical data
    temp_cols = pd.get_dummies(x.winner,prefix='result')
    x.drop('winner',axis=1,inplace=True)
    x.join(temp_cols)

    #change teams to 0 and 1
    x['team'] = (x['team']=='team2').astype(dtype=int)
    return x
def test (model,x_scaler,y_scaler):
    test_data_path='./dataset/test.csv'
    x_test = pd.read_csv(test_data_path)
    print ( x_test.shape)
  
    res =pd.DataFrame(x_test.row_id)
   
    x_test.drop('row_id',axis=1,inplace=True)
    x_test = categorical_encode(x_test)
    x_test = x_test.fillna(0)
    x_test = x_scaler.transform(x_test.values)
    
    y_pred = model.predict(x_test)
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1,1))
    res = res.join(pd.DataFrame(y_pred,columns=["rating_num"]))
    return res

def train():
    #load train data
    train_data_path  = 'dataset/train.csv'
    x_train = pd.read_csv(train_data_path)    
    #y_test = x_test ["rating_num"]

    #cleaning 
    #missing values in each column
    #print ( x_train.isnull().sum())
    #drop these rows with this missing info
    x_train=x_train.dropna(subset=['player_position_1','player_position_2','player_height','player_weight','team1_system_id','team2_system_id'])
    y_train = x_train["rating_num"]
    #drop columns that are not required
    x_train.drop('row_id',axis=1,inplace=True)
    x_train.drop('rating_num',axis=1,inplace=True)


    #encode to numerical data
    temp_cols = pd.get_dummies(x_train.winner,prefix='result')
    x_train.drop('winner',axis=1,inplace=True)
    x_train.join(temp_cols)

    #change teams to 0 and 1
    x_train['team'] = (x_train['team']=='team2').astype(dtype=int)
    #fill missing values with 0
    x_train = x_train.fillna(0)
    

    #scale x
    min_max_scaler_x = preprocessing.StandardScaler()
    min_max_scaler_y = preprocessing.StandardScaler()
    x_array = x_train.values
    x_train = min_max_scaler_x.fit_transform(x_array)
    #scale y 
    y_array = y_train.values

    y_train = min_max_scaler_y.fit_transform(y_array.reshape(-1,1))

    print(x_train.shape,y_train.shape)

    #train val split
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size= 0.85)

    #train
    xgb_r = xg.XGBRegressor(objective ='reg:linear',
                  n_estimators = 10)    
    reg = xgb_r.fit(x_train,y_train)
    print(f'train score : {reg.score(x_train,y_train)}')

    


    print (f'R2 score val {reg.score(x_val,y_val)}')


    return reg,min_max_scaler_x,min_max_scaler_y

def visualize():
    train_data_path  = 'dataset/train.csv'
    x_train = pd.read_csv(train_data_path)    
    #y_test = x_test ["rating_num"]

    #cleaning 
    #missing values in each column
    #print ( x_train.isnull().sum())
    #drop these rows with this missing info
    x_train=x_train.dropna(subset=['player_position_1','player_position_2','player_height','player_weight','team1_system_id','team2_system_id'])
    #y_train = x_train["rating_num"]
    #drop columns that are not required
    x_train.drop('row_id',axis=1,inplace=True)
    #x_train.drop('rating_num',axis=1,inplace=True)


    #encode to numerical data
    temp_cols = pd.get_dummies(x_train.winner,prefix='result')
    x_train.drop('winner',axis=1,inplace=True)
    x_train.join(temp_cols)

    #change teams to 0 and 1
    x_train['team'] = (x_train['team']=='team2').astype(dtype=int)
    #fill missing values with 0
    x_train = x_train.fillna(0)
    col_names = list(x_train)
    print(col_names)
    stdscaler = preprocessing.StandardScaler()
    x_train = pd.DataFrame(stdscaler.fit_transform(x_train.values),columns=col_names)

    print("plotting")
    sns.heatmap(x_train.corr(), annot=True, cmap="YlGnBu")
    plt.show()



if __name__ == '__main__':
    #visualize()
    
    model,x_scaler,y_scaler  = train()
    y_pred = test(model,x_scaler,y_scaler)
    
    #print (y_pred)
    y_pred.to_csv("result.csv",index=False)

    sample = pd.read_csv('./dataset/sample_submission_wBWLI0s.csv')
    print (sample.shape)
    test = pd.read_csv('result.csv')
    print ( test.shape)





