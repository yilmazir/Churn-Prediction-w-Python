import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel('churn_table_final v1.xlsx')
data['Churn'] = data['Churn'].astype(str)

print(data['Churn'].value_counts(),'\n')
print('rows - columns: ', data.shape,'\n')
print(data.dtypes,'\n') 

#bivariate analysis numeric - categorical
numeric_var = data[['days_of_usage', 'period_of_usage', 'PrimaryLastWeek', 'Primary_Avg_Weekly', 'OpenLastWeek', 'Open_Avg_Weekly']]

for var in numeric_var:
  sns.boxplot(x = var, y = data['Churn'], data = data)
  plt.show()

#bivariate analysis categorical - categorical
categorical_var = data['Subscription Offer Duration'], data['Subscription Offer Type'], data['region'], data['continent'], data['device.model'], data['device.iOSVersion'], data['language'], data['appVersion']

for var in categorical_var:
    sns.countplot(x = 'Churn', hue = var, data = data)
    plt.show()

############

from sklearn.model_selection import train_test_split

X = numeric_var
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc,'\n')

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: ', cm)


from sklearn.metrics import roc_curve, auc

y_test = y_test.astype(float)
y_pred = y_pred.astype(float)

fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, color='red', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
