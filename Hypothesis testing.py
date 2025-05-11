import pandas as pd

# Load the dataset
df = pd.read_csv("C:\\Users\\shaun\\Downloads\\CarSharing.csv")

# Check for duplicate rows
print("Number of duplicate rows before dropping:", df.duplicated().sum())

# Drop duplicate rows
df = df.drop_duplicates()
print("Number of duplicate rows after dropping:", df.duplicated().sum())

# Check for missing values
print("Missing values per column before handling:\n", df.isnull().sum())

# Handle missing values
# Step 1: Forward-fill to propagate last valid observation forward
df.ffill(inplace=True)

# Step 2: If any missing values remain (e.g., at the start), fill with column mean for numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Step 3: For non-numeric columns (if still missing), fill with mode
non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
for col in non_numeric_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Verify no missing values remain
print("Missing values per column after handling:\n", df.isnull().sum())

# Save the cleaned dataset (optional)
df.to_csv("CarSharing_cleaned.csv", index=False)
print("Dataset cleaned and saved as 'CarSharing_cleaned.csv'")

import scipy.stats as stats

# Define the columns to analyze
numerical_cols = ['temp', 'temp_feel', 'humidity', 'windspeed']
categorical_cols = ['season', 'holiday', 'workingday', 'weather']

print("----- HYPOTHESIS TESTING RESULTS (One-Line Format) -----\n")

# Pearson Correlation Results for Numerical Variables
for col in numerical_cols:
    r, p = stats.pearsonr(df[col], df['demand'])
    print(f"{col}: Pearson r = {r:.3f}, p-value = {p:.2e}")

# ANOVA Results for Categorical Variables
for col in categorical_cols:
    groups = [df[df[col] == category]['demand'] for category in df[col].unique()]
    if len(groups) > 1:
        F_stat, p_value = stats.f_oneway(*groups)
        print(f"{col}: ANOVA F = {F_stat:.2f}, p-value = {p_value:.2e}")
    else:
        print(f"{col}: Not enough groups for ANOVA analysis")
        

import calendar
import matplotlib.pyplot as plt

# Ensure the 'timestamp' column is in datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter the dataset for the year 2017
df_2017 = df[df['timestamp'].dt.year == 2017]

# Aggregate monthly averages for key variables: temp, humidity, windspeed, and demand
monthly_stats = df_2017.groupby(df_2017['timestamp'].dt.month).agg({
    'temp': 'mean',
    'humidity': 'mean',
    'windspeed': 'mean',
    'demand': 'mean'
}).reset_index()

# Map month numbers to month names for better readability
monthly_stats['month_name'] = monthly_stats['timestamp'].apply(lambda x: calendar.month_name[x])

# Reorder columns for clarity
monthly_stats = monthly_stats[['month_name', 'temp', 'humidity', 'windspeed', 'demand']]

# Print the monthly averages to observe seasonal trends
print("Seasonal and Cyclic Patterns in 2017:")
print(monthly_stats.to_string(index=False))

# ----------------------------
# Plotting the Seasonal Trends
# ----------------------------
plt.figure(figsize=(10, 6))
plt.plot(monthly_stats['month_name'], monthly_stats['temp'], marker='o', label='Temperature (°C)')
plt.plot(monthly_stats['month_name'], monthly_stats['humidity'], marker='s', label='Humidity (%)')
plt.plot(monthly_stats['month_name'], monthly_stats['windspeed'], marker='^', label='Windspeed (km/h)')
plt.plot(monthly_stats['month_name'], monthly_stats['demand'], marker='d', label='Demand')

plt.xlabel("Month")
plt.ylabel("Average Value")
plt.title("Seasonal Trends in 2017: Temperature, Humidity, Windspeed, and Demand")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Convert 'timestamp' to datetime and filter for 2017 data
df['timestamp'] = pd.to_datetime(df['timestamp'])
df_2017 = df[df['timestamp'].dt.year == 2017]

# Resample to compute weekly average demand (using only 'demand')
weekly_demand = df_2017.set_index('timestamp')['demand'].resample('W').mean().reset_index()

# Split the data into training (70%) and testing (30%) sets
train_size = int(len(weekly_demand) * 0.7)
train = weekly_demand.iloc[:train_size]
test = weekly_demand.iloc[train_size:]

# Fit the ARIMA model on the training data (order chosen as an example)
model = ARIMA(train['demand'], order=(5, 1, 2))
model_fit = model.fit()

# Forecast for the test period
forecast = model_fit.forecast(steps=len(test))

# Print a concise summary in the console
print("Training set size:", len(train))
print("Testing set size:", len(test))
forecast_df = test.copy()
forecast_df['Forecast'] = forecast.values
print("\nForecasted Weekly Demand:")
print(forecast_df[['timestamp', 'Forecast']].to_string(index=False))

# Plot actual vs. forecasted weekly demand in one graph
plt.figure(figsize=(10, 6))
plt.plot(weekly_demand['timestamp'], weekly_demand['demand'], marker='o', label='Actual Demand')
plt.plot(test['timestamp'], forecast, marker='s', linestyle='--', color='red', label='Forecasted Demand')
plt.xlabel("Date")
plt.ylabel("Weekly Average Demand")
plt.title("ARIMA Forecasting of Weekly Demand (2017)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# Assume 'df' is your cleaned DataFrame already loaded
features = ['temp', 'temp_feel', 'humidity', 'windspeed']
target = 'demand'
X = df[features]
y = df[target]

# Split the data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----- Random Forest Regressor -----
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)

# ----- MLP Regressor (Deep Neural Network) -----
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                         solver='adam', max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)
mlp_predictions = mlp_model.predict(X_test)
mlp_mse = mean_squared_error(y_test, mlp_predictions)

# Print concise summary of model performance
print("----- Demand Prediction Model Comparison -----")
print("Random Forest Regressor MSE: {:.4f}".format(rf_mse))
print("MLP Regressor MSE: {:.4f}".format(mlp_mse))

# Create a small comparison table of Actual vs. Predicted values (first 10 samples)
comparison = pd.DataFrame({
    "Actual Demand": y_test.values,
    "RF Predicted Demand": rf_predictions,
    "MLP Predicted Demand": mlp_predictions
})
print("\nComparison of Actual vs. Predicted Demand (first 10 rows):")
print(comparison.head(10))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Create binary demand labels: label 1 if demand > average, else label 2
avg_demand = df['demand'].mean()
df['demand_label'] = df['demand'].apply(lambda x: 1 if x > avg_demand else 2)

# Print overall average demand and label distribution
print("Overall Average Demand: {:.2f}".format(avg_demand))
print("Label Distribution:")
print(df['demand_label'].value_counts(), "\n")

# Select features and target
features = ['temp', 'temp_feel', 'humidity', 'windspeed']
target = 'demand_label'
X = df[features]
y = df[target]

# Split data into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
lr_cm = confusion_matrix(y_test, lr_pred)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
rf_cm = confusion_matrix(y_test, rf_pred)

# K-Nearest Neighbors Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, knn_pred)
knn_cm = confusion_matrix(y_test, knn_pred)

# Print concise classification summary
print("----- Classification Summary -----")
print(f"Logistic Regression: Accuracy = {lr_acc*100:.2f}%, Confusion Matrix = {lr_cm.tolist()}")
print(f"Random Forest:       Accuracy = {rf_acc*100:.2f}%, Confusion Matrix = {rf_cm.tolist()}")
print(f"KNN:                 Accuracy = {knn_acc*100:.2f}%, Confusion Matrix = {knn_cm.tolist()}")

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

# Convert timestamp to datetime and filter for 2017 data
df['timestamp'] = pd.to_datetime(df['timestamp'])
df_2017 = df[df['timestamp'].dt.year == 2017]

# Extract temperature data as a 2D array for clustering
X_temp = df_2017[['temp']].values

# Define the k values to test
k_values = [2, 3, 4, 12]

print("----- K-Means Clustering (Temperature Data, 2017) -----")
kmeans_results = {}
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_temp)
    counts = np.bincount(labels)
    std_dev = np.std(counts)
    kmeans_results[k] = (counts, std_dev)
    print(f"k = {k}: Cluster Counts = {counts.tolist()}, Std Dev = {std_dev:.2f}")

# Identify the most uniform clustering for K-Means (lowest std dev)
best_k_kmeans = min(kmeans_results, key=lambda x: kmeans_results[x][1])
best_counts, best_std = kmeans_results[best_k_kmeans]
print(f"\nMost uniform K-Means clustering: k = {best_k_kmeans} with counts = {best_counts.tolist()} and Std Dev = {best_std:.2f}")

print("\n----- Agglomerative Clustering (Temperature Data, 2017) -----")
agg_results = {}
for k in k_values:
    agg = AgglomerativeClustering(n_clusters=k)
    labels = agg.fit_predict(X_temp)
    counts = np.bincount(labels)
    std_dev = np.std(counts)
    agg_results[k] = (counts, std_dev)
    print(f"k = {k}: Cluster Counts = {counts.tolist()}, Std Dev = {std_dev:.2f}")

# Identify the most uniform clustering for Agglomerative Clustering
best_k_agg = min(agg_results, key=lambda x: agg_results[x][1])
best_counts_agg, best_std_agg = agg_results[best_k_agg]
print(f"\nMost uniform Agglomerative clustering: k = {best_k_agg} with counts = {best_counts_agg.tolist()} and Std Dev = {best_std_agg:.2f}")




















