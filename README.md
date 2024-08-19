# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:VAISHALI BALAMURUGAN
### Register Number:212222230164
```python
from google.colab import auth
import gspread
from oauth2client.client import GoogleCredentials
from google.auth import default
import pandas as pd

!pip install scikit-learn  

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)

auth.authenticate_user()
creds, _ = default()  # Get credentials using the newer google-auth library
gc = gspread.authorize(creds)

worksheet=gc.open('12.8.24 deeplearning').sheet1
data=worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])  # Capitalize 'F' in 'DataFrame'
dataset1 = dataset1.astype({'input': 'float'})  # Change 'Input' to 'input'
dataset1 = dataset1.astype({'output': 'float'})  # Change 'Output' to 'output'

dataset1.head()

x = dataset1[['input']].values  # Change 'Input' to 'input'
y = dataset1[['output']].values  # Change 'Output' to 'output'

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(X_train)

X_train1=Scaler.transform(X_train)

ai_brain=Sequential([
   # Dense(12,input_dim=1,activation='relu'),
    Dense(8,activation='relu'),
    Dense(1)
])

ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(X_train1,y_train,epochs=2000)

loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()

X_test1=Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)

X_n1=[[4]]
X_n1=Scaler.transform(X_n1)
ai_brain.predict(X_n1)



```
## Dataset Information


![image](https://github.com/user-attachments/assets/11f0f084-e12d-4d8d-8702-28a6c5cb8a04)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/c201e401-29a7-4793-aea5-978ae25036ce)


### Test Data Root Mean Squared Error

Find the test data root mean squared error

### New Sample Data Prediction


![image](https://github.com/user-attachments/assets/12e99c15-e434-42a3-b52c-0fc196a0c6d0)


## RESULT

Thus the Process of developing a neural network regression model for the created dataset is successfully executed.


