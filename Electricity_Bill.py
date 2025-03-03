import pandas as pd 
df = pd.read_csv('/content/electricity_bill_dataset.csv')
df.head()
df.isnull().sum()

#separating categorical and numerical data
df.info()
cat_cols = df.select_dtypes(include='object').columns

#Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cat_cols:
  df[col] = le.fit_transform(df[col])

#Selection of target
x = df.drop(columns=['ElectricityBill'])
y = df['ElectricityBill']

#splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=61)

#Standerdization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Training model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

#prediction
y_pred = model.predict(x_test)

#evaluation
from sklearn.metrics import r2_score,mean_squared_error
accuracy = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
import numpy as np
mape = np.mean(np.abs(y_test-y_pred)/y_test)*100
print('Accuracy:',accuracy*100,'%')
print('Error:',mape,'%')
print('MSE:',mse)