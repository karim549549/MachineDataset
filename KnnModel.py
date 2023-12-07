import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from scipy.stats import zscore

df = pd.read_csv('top250-00-19.csv')


df=df.dropna()
df = df[df['Age'] >= 16]


print(df['Position'].value_counts())

label_encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype not in ['int64', 'float64']:
        df[col] = label_encoder.fit_transform(df[col])

X = df.drop(['Transfer_fee','Name','Season','League_from','League_to','Position'], axis=1)
y = df['Transfer_fee']

z_scores = zscore(df)
df_no_outliers = df[(z_scores < 3).all(axis=1)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




knn = KNeighborsRegressor(n_neighbors=5)


knn.fit(X_train_scaled, y_train)


y_pred_knn = knn.predict(X_test_scaled)


r2_knn = r2_score(y_test, y_pred_knn)
print(f"KNN R-squared: {r2_knn}")


variance_explained_knn = r2_knn * 100
print(f"Percentage of Variance Explained (KNN): {variance_explained_knn:.2f}%")


plt.scatter(y_test, y_pred_knn, alpha=0.5, label='_nolegend_')  # Using label='_nolegend_' removes the legend entry for the scatter plot
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Line of Equality')
plt.xlabel("Actual Values")
plt.ylabel("KNN Predicted Values")
plt.title("KNN Actual vs Predicted Values")
plt.legend()
plt.show()

print(X.columns)