import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Import data file
# change the path to the path u saved this file
cardio = pd.read_csv("F:/winter/6131/ehospital/e-hospital-2023/cardio_dataset.csv")

# Conduct data cleaning
Pre_Cleaned = pd.DataFrame(cardio)
# Check missing value
print(Pre_Cleaned.isnull().any())
# Check outlier
Pre_Cleaned.describe().to_csv('DataDescribe.csv')
# Delete "height" value that is lower than 100CM and higher than 200CM
Pre_Cleaned = Pre_Cleaned.drop(Pre_Cleaned[Pre_Cleaned['height'] > 200].index)
Pre_Cleaned = Pre_Cleaned.drop(Pre_Cleaned[Pre_Cleaned['height'] < 100].index)
# Delete "weight" value that is lower than 30kg
Pre_Cleaned = Pre_Cleaned.drop(Pre_Cleaned[Pre_Cleaned['weight'] < 30].index)
# Delete "ap_hi" value that is lower than 50 and higher than 300
Pre_Cleaned = Pre_Cleaned.drop(Pre_Cleaned[Pre_Cleaned['ap_hi'] > 350].index)
Pre_Cleaned = Pre_Cleaned.drop(Pre_Cleaned[Pre_Cleaned['ap_hi'] < 50].index)
# Delete "ap_hi" value that is lower than 20 and higher than 250
Pre_Cleaned = Pre_Cleaned.drop(Pre_Cleaned[Pre_Cleaned['ap_lo'] > 250].index)
dataset_Cleaned = Pre_Cleaned.drop(Pre_Cleaned[Pre_Cleaned['ap_lo'] < 20].index)
# Double check the distribution and generate the csv file
# dataset_Cleaned.describe().to_csv('DataDes_Cleaned.csv')
# dataset_Cleaned.to_csv('Data_Cleaned.csv')

# Sampling testing dataset and training dataset using random Sampling methods
dataset_Cleaned = dataset_Cleaned.drop('id', axis = 1)
X = dataset_Cleaned.drop('cardio', axis = 1)
y = dataset_Cleaned['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model Development
# gradient Boosting Decision Tree model
gradient_model = GradientBoostingClassifier()
gradient_model.fit(X_train,y_train)



#保存模型
#save model
joblib.dump(gradient_model,"pred.m")

print(gradient_model.score(X_train,y_train),gradient_model.score(X_test,y_test))
