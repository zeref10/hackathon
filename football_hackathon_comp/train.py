
from unicodedata import category
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
#load train data
train_data_path  = './dataset/train.csv'
x_train = pd.read_csv(train_data_path)


 
#load test data
test_data_path = './dataset/test.csv'
x_test = pd.read_csv(test_data_path)
#y_test = x_test ["rating_num"]

#cleaning 
#missing values in each column
#print ( x_train.isnull().sum())
#drop these rows with this missing info
x_train=x_train.dropna(subset=['player_position_1','player_position_2','player_height','player_weight','team1_system_id','team2_system_id'])
y_train = x_train["rating_num"]

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
min_max_scaler = preprocessing.MinMaxScaler()
x_array = x_train.values
x_train = min_max_scaler.fit_transform(x_array)
#scale y 
y_array = y_train.values

y_train = min_max_scaler.fit_transform(y_array.reshape(-1,1))

print(x_train.shape,y_train.shape)

#train val split
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size= 0.85)

#train
reg = LinearRegression().fit(x_train,y_train)
print(f'train score : {reg.score(x_train,y_train)}')

preds = reg.predict(x_val)


print (f'R2 score val {reg.score(x_val,y_val)}')



#visualize
 
#x_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(x_array)


#
'''
category = ["offensive","defensive","positional","physical","general","other"]
type = ["derived","raw","ratio"]

for i in category:
    for j in type:
        temp = df.filter(regex=f'player_{i}_{j}_var_').sum(axis=0)
        df[f"player_{i}_{j}_var_aggergate"] = temp
        #drop statement
        df.drop(list(df.filter(regex = f'player_{i}_{j}_var_')), axis = 1, inplace = True)

'''



