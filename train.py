import pandas as pd
from numpy import median
from numpy import std
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import skew
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and Read the data
df = pd.read_csv("Admission_Predict.csv")
print(df.head())

# Check the info
print(df.info())

#Check for skewness in the data
for col in df:
    print(skew(df[col]))


# Check for null values
print(df.isnull().sum())

x = df.drop(["Chance of Admit", "Serial No."], axis=1)
y = df["Chance of Admit"]
# scalar=StandardScaler()
# scaled_df = scalar.fit_transform(df)

# print(scaled_df)

plt.scatter(df["GRE Score"], y)
plt.scatter(df["TOEFL Score"], y)
plt.scatter(df["CGPA"], y)
# plt.show()


train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.4, random_state=100
)

scalar_train = StandardScaler().fit(train_x)
scaled_train_x = scalar_train.transform(train_x)

scalar_test = StandardScaler().fit(test_x)
scaled_test_x = scalar_test.transform(test_x)
reg = linear_model.LinearRegression()
scores = cross_val_score(reg, scaled_train_x, train_y, scoring="r2", cv=5)
print(scores)
model = reg.fit(scaled_train_x, train_y)

# Cross Validation and Hyperparameter tuning

folds = KFold(n_splits=10, shuffle=True, random_state=100)

hyper_params = [{"n_features_to_select": list(range(1, 14))}]

lm = linear_model.LinearRegression()
lm.fit(scaled_train_x, train_y)
rfe = RFE(lm)

model_cv = GridSearchCV(
    estimator=rfe,
    param_grid=hyper_params,
    scoring="r2",
    cv=folds,
    verbose=1,
    return_train_score=True,
)

# fit the model
model_cv.fit(scaled_train_x, train_y)

# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.to_csv("cv_results.csv")
#print(cv_results)

# plotting cv results
plt.figure(figsize=(16, 6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel("number of features")
plt.ylabel("r-squared")
plt.title("Optimal Number of Features")
plt.legend(["test score", "train score"], loc="upper left")
plt.show()
# score=r2_score(reg.predict(scaled_test_x),test_y)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# print(score)
# print('Accuracy: %.3f (%.3f)' % (median(n_scores), std(n_scores)))


# final model
n_features_optimal = 8

lm = linear_model.LinearRegression()
lm.fit(scaled_train_x, train_y)

rfe = RFE(lm, n_features_to_select=n_features_optimal)
rfe = rfe.fit(scaled_train_x, train_y)

# predict prices of X_test
y_pred = lm.predict(scaled_test_x)
r2 = r2_score(test_y, y_pred)
print(r2)

file_name='Admission_Selection_model.pickle'
pickle.dump(lm,open(file_name,'wb'))


