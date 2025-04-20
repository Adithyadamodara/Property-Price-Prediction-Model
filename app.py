import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import pickle
import os

# Function to load and clean the dataset
def load_and_clean_data():
    # Load the dataset
    df = pd.read_csv('melb_data.csv')
    
    # Select relevant features based on the report
    features = ['Rooms', 'Type', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 
                'Landsize', 'BuildingArea', 'YearBuilt', 'Price']
    df = df[features]
    
    # Handle missing values
    numeric_cols = ['Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 
                    'BuildingArea', 'YearBuilt']
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Categorical column: impute with mode
    df['Type'] = df['Type'].fillna(df['Type'].mode()[0])
    
    # Remove outliers (using IQR method for Price)
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Price'] >= Q1 - 1.5 * IQR) & (df['Price'] <= Q3 + 1.5 * IQR)]
    
    return df

# Function to build and train the neural network
def build_and_train_model(X, y):
    # Define categorical and numerical columns
    categorical_cols = ['Type']
    numerical_cols = ['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 
                     'Landsize', 'BuildingArea', 'YearBuilt']
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply preprocessing
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    # Save preprocessor
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Build neural network
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    # Compile model with proper loss and metric objects
    model.compile(
        optimizer='adam',
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError()]
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        verbose=0
    )
    
    # Save model
    model.save('price_prediction_model.keras')  # Use .keras format for compatibility
    
    return model, preprocessor, history

# Function to load model and preprocessor
def load_model_and_preprocessor():
    if os.path.exists('price_prediction_model.keras') and os.path.exists('preprocessor.pkl'):
        try:
            model = tf.keras.models.load_model('price_prediction_model.keras')
            with open('preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
            return model, preprocessor
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None
    return None, None

# Streamlit app
def main():
    st.title('Melbourne Property Price Prediction')
    st.write('Predict property prices using a neural network based on structured data.')
    
    # Load and clean data
    df = load_and_clean_data()
    
    # Prepare features and target
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Load or train model
    model, preprocessor = load_model_and_preprocessor()
    if model is None or preprocessor is None:
        st.write('Training model...')
        model, preprocessor, history = build_and_train_model(X, y)
        st.write('Model trained successfully!')
    
    # User input form
    st.header('Enter Property Details')
    
    col1, col2 = st.columns(2)
    
    with col1:
        rooms = st.number_input('Rooms', min_value=1, max_value=10, value=3)
        distance = st.number_input('Distance from CBD (km)', min_value=0.0, value=10.0)
        bedroom2 = st.number_input('Bedrooms', min_value=0, max_value=10, value=3)
        bathroom = st.number_input('Bathrooms', min_value=1, max_value=10, value=2)
    
    with col2:
        car = st.number_input('Car Spaces', min_value=0, max_value=10, value=2)
        landsize = st.number_input('Land Size (sqm)', min_value=0.0, value=500.0)
        building_area = st.number_input('Building Area (sqm)', min_value=0.0, value=150.0)
        year_built = st.number_input('Year Built', min_value=1800, max_value=2025, value=2000)
    
    property_type = st.selectbox('Property Type', ['h (house)', 'u (unit)', 't (townhouse)'])
    property_type = property_type.split()[0]  # Extract 'h', 'u', or 't'
    
    # Predict button
    if st.button('Predict Price'):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Rooms': [rooms],
            'Type': [property_type],
            'Distance': [distance],
            'Bedroom2': [bedroom2],
            'Bathroom': [bathroom],
            'Car': [car],
            'Landsize': [landsize],
            'BuildingArea': [building_area],
            'YearBuilt': [year_built]
        })
        
        # Preprocess input
        input_processed = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_processed)[0][0]
        
        # Display result
        st.success(f'Predicted Property Price: ${prediction:,.2f}')

if __name__ == '__main__':
    main()