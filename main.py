import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import zscore


df = pd.read_csv('top250-00-19.csv')


df=df.dropna()
df = df[df['Age'] >= 16]


label_encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype not in ['int64', 'float64']:
        df[col] = label_encoder.fit_transform(df[col])


X = df.drop(['Transfer_fee','Name','Season','League_from','League_to','Position'], axis=1)
y = df['Transfer_fee']


z_scores = zscore(df)
df_no_outliers = df[(z_scores < 3).all(axis=1)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#fitTheModel
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)


y_pred_scaled = linear_reg.predict(X_test_scaled)


r2_scaled = r2_score(y_test, y_pred_scaled)
print(f"Scaled R-squared: {r2_scaled}")
#visulaize
plt.scatter(y_test, y_pred_scaled, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Line of Equality')
plt.xlabel("Actual Values")
plt.ylabel("Scaled Predicted Values")
plt.title("Scaled Actual vs Predicted Values")
plt.legend()
plt.show()
