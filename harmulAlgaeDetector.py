import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from geopy.geocoders import Nominatim

# Load the dataset
df = pd.read_csv('/Users/sairaghavganesh/saiProjects/habsos_20240430.csv')

# Display the column names to understand its structure
print(df.columns)

# Filter for relevant genus/species indicating harmful algal blooms
df = df[(df['GENUS'] == 'Karenia') & (df['SPECIES'] == 'brevis')]

# Select relevant columns - adjust based on actual available columns
df = df[['LATITUDE', 'LONGITUDE', 'WATER_TEMP', 'SALINITY']]  # Adjusted: Removed 'CHLOROPHYLL'

# Drop rows with missing values
df = df.dropna()

# Add a column for HAB (harmful algal bloom presence)
df['HAB'] = 1

# Preprocess the data
X = df[['LATITUDE', 'LONGITUDE', 'WATER_TEMP', 'SALINITY']]  # Features
y = df['HAB']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and model
best_params = grid_search.best_params_
model = grid_search.best_estimator_

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Function to predict HAB based on user input
def predict_hab(latitude, longitude, water_temp, salinity):
    input_data = scaler.transform([[latitude, longitude, water_temp, salinity]])
    prediction = model.predict(input_data)[0]
    return bool(prediction)

# Function to convert location name to latitude and longitude
def get_coordinates(location):
    geolocator = Nominatim(user_agent="hab_predictor")
    location = geolocator.geocode(location)
    if location:
        return location.latitude, location.longitude
    else:
        print("Location not found. Please enter a valid location.")
        return None, None

# Function to send email alert with heatmap
def send_alert(receiver_email, location, fig_path):
    sender_email = "algaeguard@gmail.com"
    app_password = "kbed nbsh tatl thjk"
    
    message = MIMEMultipart("alternative")
    message["Subject"] = "Harmful Algal Bloom Alert"
    message["From"] = sender_email
    message["To"] = receiver_email

    text = f"A harmful algal bloom has been detected at the following location: {location}"
    part1 = MIMEText(text, "plain")

    message.attach(part1)

    with open(fig_path, 'rb') as attachment:
        part2 = MIMEBase('application', 'octet-stream')
        part2.set_payload(attachment.read())

    encoders.encode_base64(part2)
    part2.add_header(
        "Content-Disposition",
        f"attachment; filename= {os.path.basename(fig_path)}",
    )

    message.attach(part2)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        print(f"Email sent to {receiver_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to get validated float input
def get_float_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# User input for prediction
print("\nEnter water quality parameters to predict Harmful Algal Blooms (HAB):")
print("Location Guide: Enter the name of the location in a format recognized by geocoding services. Examples: 'Tampa Bay, FL', 'Gulf of Mexico'")
location = input("Enter the location name: ")
latitude, longitude = get_coordinates(location)
if latitude is None or longitude is None:
    raise ValueError("Invalid location. Please run the script again with a valid location.")

water_temp = get_float_input("Water Temperature (°C): ")
salinity = get_float_input("Salinity: ")
receiver_email = input("Enter the receiver's email for alerts: ")

# Perform prediction
hab_present = predict_hab(latitude, longitude, water_temp, salinity)
location_info = f"{location} (Latitude: {latitude}, Longitude: {longitude})"

# Generate and save heatmap
fig_path = "heatmap.png"
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Water Quality Parameters")
plt.savefig(fig_path)

# Display prediction and send alert
if hab_present:
    print("Warning: Harmful Algal Bloom (HAB) detected.")
    send_alert(receiver_email, location_info, fig_path)
else:
    print("No Harmful Algal Bloom (HAB) detected.")
