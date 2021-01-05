google colap 에서 작업



Try - LinearRegression

import pandas as pd
df= pd.read_excel('USA_Housing.csv.xslx')
y = df['Price']
x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
'Avg. Area Number of Bedrooms', 'Area Population']]
from sklearn.model_selection import train_test_split # Cross Validation
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)
lr.score(x_test, y_test)
from sklearn import metrics
metrics.r2_score(y_test, predictions)

-----------------------------------------------------------------------------------------

Try - Gradient Descent
~$ 03.LinearRegressionWithUSAHousing.ipynb
❖ train_test_split(*arrays, **options) : Split arrays or matrices into random
from sklearn.model_selection import train_test_split
from sklearn import datasets
boston_dataset = datasets.load_boston()
boston_data = boston_dataset.data
boston_target = boston_dataset.target
train_boston_data, test_boston_data, train_boston_target, test_boston_target
= train_test_split(boston_data, boston_target, test_size=0.2)
❖ LinearRegression(*, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None) : Ordinary least squares
from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()
linearRegression.fit(train_boston_data, train_boston_target)
linearRegression.score(test_boston_data, test_boston_target)
lr.intercept_, lr.coef_
coef_df = pd.DataFrame(linearRegression.coef_, columns=['Coefficient'], index = boston_dataset.feature_names)


-------------------------------------------------------------------------------------------------------


Try - LinearRegression
❖ Download 'EcommerceCustomers.csv.xlsx'
@ vi 03.LinearRegressionWithboston.ipynb
import pandas as pd
df= pd.read_excel('EcommerceCustomers.csv.xlsx')
y = df['Yearly Amount Spent']
x = df[['Avg. ... Length', 'Time ... App', 'Time ... Website', 'Length ... Membership']]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
lr.intercept_, lr.coef_
predictions = lr.predict(x_test)
plt.scatter(y_test, predictions)
lr.score(x_test, y_test)
from sklearn import metrics
metrics.r2_score(y_test, predictions)