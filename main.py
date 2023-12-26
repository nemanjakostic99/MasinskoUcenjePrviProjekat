import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



# Create URL
filePath = 'hcvdat0.csv'

dataframe = pd.read_csv(filepath_or_buffer=filePath)

dataframe = dataframe.dropna()

scale_mapper = {
    "0=Blood Donor": 0,
    "0s=suspect Blood Donor": 2,
    "1=Hepatitis": 1,
    "2=Fibrosis": 1,
    "3=Cirrhosis": 1}

# Deljenje tipa krvi u dve klase
dataframe['Category'] = dataframe['Category'].replace(scale_mapper)
dataframe['Sex'] = dataframe['Category'].replace({"m":0, "f":1})
dataframe = dataframe[dataframe['Category'] != 2]

columns_to_scale = ['Age', 'ALB', 'ALP', 'ALB', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

# Create scaler
scaler = preprocessing.StandardScaler()
# Transform the feature
dataframe[columns_to_scale] = scaler.fit_transform(dataframe[columns_to_scale])

outlier_detector = EllipticEnvelope(contamination=.009)
# Fit detector
outlier_detector.fit(dataframe[columns_to_scale])
# Predict outliers
outliers = outlier_detector.predict(dataframe[columns_to_scale])
outliers_indices = outliers == -1
dataframe = dataframe[~outliers_indices]

# View first two rows
#print(dataframe.head(20))

features, target = dataframe[['Category', 'Age', 'ALB', 'ALP', 'ALB', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']], dataframe['Category']

# Split into training and test set
features_train, features_test, target_train, target_test = train_test_split(
features, target, random_state=0)

# Create dummy classifier
dummy = DummyClassifier(strategy='uniform', random_state=1)
# "Train" model
dummy.fit(features_train, target_train)
# Get accuracy score
print("Basic train test split, dummy classifier")
print(dummy.score(features_test, target_test))

# Create standardizer
standardizer = StandardScaler()
# Create logistic regression object
logit = LogisticRegression()
# Create a pipeline that standardizes, then runs logistic regression
pipeline = make_pipeline(standardizer, logit)

kf = KFold(n_splits=10, shuffle=True, random_state=1)
# Conduct k-fold cross-validation
cv_results = cross_val_score(
    pipeline, # Pipeline
    features, # Feature matrix
    target, # Target vector
    cv=kf, # Cross-validation technique
    scoring="accuracy", # Loss function
    n_jobs=-1) # Use all CPU scores

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# Conduct k-fold cross-validation
cv_results = cross_val_score(
    pipeline, # Pipeline
    features, # Feature matrix
    target, # Target vector
    cv=kf, # Cross-validation technique
    scoring="accuracy", # Loss function
    n_jobs=-1) # Use all CPU scores