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
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier



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

features, target = dataframe[['Age', 'ALB', 'ALP', 'ALB', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']], dataframe['Category']

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

# RANDOM FOREST cLASSIFIER

# Create standardizer
standardizer = StandardScaler()
# Create logistic regression object
# Create Random Forest Classifier object
rf_classifier = RandomForestClassifier()
# Create a pipeline that standardizes, then runs logistic regression
pipeline = make_pipeline(standardizer, rf_classifier)

kf = KFold(n_splits=10, shuffle=True, random_state=1)
# Conduct k-fold cross-validation
kf_rfc_cv_results = cross_val_score(
    pipeline, # Pipeline
    features, # Feature matrix
    target, # Target vector
    cv=kf, # Cross-validation technique
    scoring="accuracy", # Loss function
    n_jobs=-1) # Use all CPU scores

print("Random Forest Classifier Cross-Validation Mean Accuracy:", kf_rfc_cv_results.mean())

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# Conduct k-fold cross-validation
skf_rfc_cv_results = cross_val_score(
    pipeline, # Pipeline
    features, # Feature matrix
    target, # Target vector
    cv=skf, # Cross-validation technique
    scoring="accuracy", # Loss function
    n_jobs=-1) # Use all CPU scores

print("Random Forest Classifier Stratified Cross-Validation Mean Accuracy:", skf_rfc_cv_results.mean())

rf_classifier_basicSplit = RandomForestClassifier()
# Train model
rf_classifier_basicSplit.fit(features_train, target_train)
# Get accuracy score
print("Train Test Split Random Forest Classifier", rf_classifier_basicSplit.score(features_test, target_test))

# NAIVE BAYES

# Create Naive Bayes classifier object
nb_classifier = GaussianNB()

# Create a pipeline that standardizes, then runs Naive Bayes
pipeline_nb = make_pipeline(standardizer, nb_classifier)

# Use K-Fold cross-validation
kf_nb_cv_results = cross_val_score(
    pipeline_nb,  # Pipeline
    features,     # Feature matrix
    target,       # Target vector
    cv=kf,        # Cross-validation technique
    scoring="accuracy",  # Loss function
    n_jobs=-1     # Use all CPU cores
)
print("Naive Bayes Cross-Validation Mean Accuracy:", kf_nb_cv_results.mean())

skf_nb_cv_results = cross_val_score(
    pipeline_nb,  # Pipeline
    features,     # Feature matrix
    target,       # Target vector
    cv=skf,        # Cross-validation technique
    scoring="accuracy",  # Loss function
    n_jobs=-1     # Use all CPU cores
)
print("Naive Bayes Stratified Cross-Validation Mean Accuracy:", skf_nb_cv_results.mean())

nb_classifier_basicSplit = GaussianNB()
# Train model
nb_classifier_basicSplit.fit(features_train, target_train)
# Get accuracy score
print("Train Test Split Naive Bayes", nb_classifier_basicSplit.score(features_test, target_test))

# K-Nearest Neighbors

# Create K-Nearest Neighbors classifier object
knn_classifier = KNeighborsClassifier()

# Create a pipeline that standardizes, then runs KNN
pipeline_knn = make_pipeline(standardizer, knn_classifier)

# Use K-Fold cross-validation
kf_knn_cv_results = cross_val_score(
    pipeline_knn,  # Pipeline
    features,      # Feature matrix
    target,        # Target vector
    cv=kf,         # Cross-validation technique
    scoring="accuracy",   # Loss function
    n_jobs=-1      # Use all CPU cores
)
print("K-Nearest Neighbors Cross-Validation Mean Accuracy:", kf_knn_cv_results.mean())

skf_knn_cv_results = cross_val_score(
    pipeline_knn,  # Pipeline
    features,      # Feature matrix
    target,        # Target vector
    cv=skf,         # Cross-validation technique
    scoring="accuracy",   # Loss function
    n_jobs=-1      # Use all CPU cores
)
print("K-Nearest Neighbors Stratified Cross-Validation Mean Accuracy:", skf_knn_cv_results.mean())

knn_classifier_basicSplit = KNeighborsClassifier()
# Train model
knn_classifier_basicSplit.fit(features_train, target_train)
# Get accuracy score
print("Train Test Split K-Nearest Neighbors", knn_classifier_basicSplit.score(features_test, target_test))


# GRADIENT BOOSTING

# Create Gradient Boosting classifier object
gb_classifier = GradientBoostingClassifier()

# Create a pipeline that standardizes, then runs Gradient Boosting
pipeline_gb = make_pipeline(standardizer, gb_classifier)

# Use K-Fold cross-validation
kf_gb_cv_results = cross_val_score(
    pipeline_gb,   # Pipeline
    features,      # Feature matrix
    target,        # Target vector
    cv=kf,         # Cross-validation technique
    scoring="accuracy",   # Loss function
    n_jobs=-1      # Use all CPU cores
)
print("Gradient Boosting Cross-Validation Mean Accuracy:", kf_gb_cv_results.mean())

skf_gb_cv_results = cross_val_score(
    pipeline_gb,   # Pipeline
    features,      # Feature matrix
    target,        # Target vector
    cv=skf,         # Cross-validation technique
    scoring="accuracy",   # Loss function
    n_jobs=-1      # Use all CPU cores
)
print("Gradient Boosting Stratified Cross-Validation Mean Accuracy:", skf_gb_cv_results.mean())

gb_classifier_basicSplit = KNeighborsClassifier()
# Train model
gb_classifier_basicSplit.fit(features_train, target_train)
# Get accuracy score
print("Train Test Split Gradient Boosting", gb_classifier_basicSplit.score(features_test, target_test))

# LOGISTIC REGRESSION

# Create Logistic Regression classifier object
log_reg_classifier = LogisticRegression()

# Create a pipeline that standardizes, then runs Logistic Regression
pipeline_log_reg = make_pipeline(standardizer, log_reg_classifier)

# Use K-Fold cross-validation
kf_log_reg_cv_results = cross_val_score(
    pipeline_log_reg,  # Pipeline
    features,          # Feature matrix
    target,            # Target vector
    cv=kf,             # Cross-validation technique
    scoring="accuracy",   # Loss function
    n_jobs=-1          # Use all CPU cores
)
print("Logistic Regression Cross-Validation Mean Accuracy:", kf_log_reg_cv_results.mean())

skf_log_reg_cv_results = cross_val_score(
    pipeline_log_reg,  # Pipeline
    features,          # Feature matrix
    target,            # Target vector
    cv=skf,             # Cross-validation technique
    scoring="accuracy",   # Loss function
    n_jobs=-1          # Use all CPU cores
)
print("Logistic Regression Stratified Cross-Validation Mean Accuracy:", skf_log_reg_cv_results.mean())

log_reg__basicSplit = KNeighborsClassifier()
# Train model
log_reg__basicSplit.fit(features_train, target_train)
# Get accuracy score
print("Train Test Split Logistic Regression", log_reg__basicSplit.score(features_test, target_test))