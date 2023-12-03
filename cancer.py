import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib

cancer = pd.read_csv('survey_lung_cancer.csv')

# Preprocessing 
lung_cancer_dict = {
    'YES': 1,
    'NO': 0
}
cancer['LUNG_CANCER'] = cancer.LUNG_CANCER.map(lung_cancer_dict)

gender_dict = {'F': 0, 'M': 1}
cancer['GENDER'] = cancer.GENDER.map(gender_dict)


mode_imputer = SimpleImputer(strategy='most_frequent')
cancer['SMOKING'] = mode_imputer.fit_transform(cancer[['SMOKING']])
cancer['GENDER'] = mode_imputer.fit_transform(cancer[['GENDER']])

cancer = pd.DataFrame(cancer, columns=['GENDER', 'AGE', 'SMOKING','YELLOW_FINGERS', 'ANXIETY','PEER_PRESSURE', 'CHRONIC_DISEASE','FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING','SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'LUNG_CANCER'])


data_n = 300
cancer = cancer.sample(n=data_n, random_state=1)
feature_cols = ['GENDER', 'AGE', 'SMOKING','YELLOW_FINGERS', 'ANXIETY','PEER_PRESSURE', 'CHRONIC_DISEASE','FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING','SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
X = cancer[feature_cols]
y = cancer['LUNG_CANCER']



scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X = pd.DataFrame(X, columns=['GENDER', 'AGE', 'SMOKING','YELLOW_FINGERS', 'ANXIETY','PEER_PRESSURE', 'CHRONIC_DISEASE','FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING','SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 70% training and 30% test


# Tree

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=2)

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

# #Predict the response for test dataset
# y_pred = clf.predict(X_test)

# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# Import 
joblib.dump(clf, 'cancer.pkl')
joblib.dump(scaler, 'scaler.pkl')