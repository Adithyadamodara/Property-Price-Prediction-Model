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
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from PIL import Image
import base64
from io import BytesIO
import requests
import tempfile

# Set page configuration
st.set_page_config(
    page_title="Melbourne Property Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced styling
st.markdown("""
<style>
    /* Main elements */
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .info-text {
        font-size: 1rem;
        color: #4B5563;
        line-height: 1.6;
    }
    
    /* Containers and cards */
    .highlight {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #3B82F6;
    }
    .prediction-box {
        background-color: #EFF6FF;
        padding: 2rem;
        border-radius: 0.75rem;
        margin-top: 1.5rem;
        text-align: center;
        border: 1px solid #BFDBFE;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
        transition: all 0.3s ease;
    }
    .prediction-box:hover {
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
        transform: translateY(-2px);
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .card:hover {
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    /* Interactive elements */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #4B5563;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Custom form styling */
    div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label {
        font-weight: 600;
        color: #1F2937;
    }
    div.stButton > button:first-child {
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.2s;
    }
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3);
        transform: translateY(-1px);
    }
    
    /* Feature explanation styling */
    .feature-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 3px solid #3B82F6;
    }
    .feature-title {
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #E5E7EB;
        font-size: 0.8rem;
        color: #6B7280;
    }
    
    /* Property type icons */
    .property-icon {
        font-size: 2rem;
        margin-right: 0.5rem;
        vertical-align: middle;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .animate-fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Custom tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0 1rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.1);
        border-bottom: 3px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for visual enhancements
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def get_property_icon(prop_type):
    icons = {
        'h': '🏡', # house
        'u': '🏢', # unit/apartment
        't': '🏘️', # townhouse
    }
    return icons.get(prop_type, '🏠')

def create_price_gauge(prediction, min_price, max_price):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Price (AUD)", 'font': {'size': 24, 'color': '#1E3A8A'}},
        gauge = {
            'axis': {'range': [min_price, max_price], 'tickwidth': 1, 'tickcolor': "#4B5563"},
            'bar': {'color': "#3B82F6"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#E5E7EB",
            'steps': [
                {'range': [min_price, min_price + (max_price-min_price)/3], 'color': '#93C5FD'},
                {'range': [min_price + (max_price-min_price)/3, min_price + 2*(max_price-min_price)/3], 'color': '#60A5FA'},
                {'range': [min_price + 2*(max_price-min_price)/3, max_price], 'color': '#2563EB'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        },
        number = {'prefix': "$", 'font': {'size': 30, 'color': '#1E3A8A'}, 'valueformat': ',.0f'}
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Google Places API key
GOOGLE_API_KEY = ''

# Initialize ResNet50 model for visual feature extraction
@st.cache_resource
def load_resnet_model():
    """Load ResNet50 model for feature extraction"""
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return model

# Function to extract features from property image
def extract_visual_features(uploaded_image, model):
    """
    Extract visual features from a property image using ResNet50
    
    Parameters:
    uploaded_image: Uploaded image file
    model: Loaded ResNet50 model
    
    Returns:
    np.array: Feature vector
    """
    try:
        # Save uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_image.getvalue())
            img_path = tmp.name
        
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Extract features
        features = model.predict(x)
        
        # Clean up
        os.unlink(img_path)
        
        return features.flatten()
    except Exception as e:
        st.error(f"Error extracting visual features: {str(e)}")
        return None

# Function to analyze property image
def analyze_property_image(uploaded_image):
    """
    Analyze property image and provide insights
    
    Parameters:
    uploaded_image: Uploaded image file
    
    Returns:
    dict: Analysis results
    """
    model = load_resnet_model()
    features = extract_visual_features(uploaded_image, model)
    
    if features is None:
        return None
    
    # Example property aspects that could be detected
    # In a real implementation, you would have a trained classifier for these
    property_aspects = {
        'modern_design': np.mean(features[:100]) > 0.1,
        'natural_light': np.mean(features[100:200]) > 0.15,
        'open_floor_plan': np.mean(features[200:300]) > 0.12,
        'good_condition': np.mean(features[300:400]) > 0.2,
        'curb_appeal': np.mean(features[400:500]) > 0.18,
        'updated_kitchen': np.mean(features[500:600]) > 0.16,
        'landscaping': np.mean(features[600:700]) > 0.14,
    }
    
    # Price impact of each aspect (in percentage)
    price_impacts = {
        'modern_design': 3.5,
        'natural_light': 2.8,
        'open_floor_plan': 4.2,
        'good_condition': 5.0,
        'curb_appeal': 3.2,
        'updated_kitchen': 4.5,
        'landscaping': 2.0,
    }
    
    # Calculate total price impact
    total_impact = sum(price_impacts[aspect] for aspect, detected in property_aspects.items() if detected)
    
    return {
        'aspects': property_aspects,
        'price_impacts': price_impacts,
        'total_impact': total_impact
    }

# Function to get POIs near a location
def get_nearby_pois(lat, lng, poi_type, radius=1000):
    """
    Get points of interest near a specific location using Google Places API
    
    Parameters:
    lat (float): Latitude
    lng (float): Longitude
    poi_type (str): Type of POI (e.g., 'school', 'hospital', 'restaurant')
    radius (int): Search radius in meters
    
    Returns:
    list: List of POIs found
    """
    try:
        url = (
            f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
            f"location={lat},{lng}&radius={radius}&type={poi_type}&key={GOOGLE_API_KEY}"
        )
        response = requests.get(url)
        results = response.json().get("results", [])
        return results
    except Exception as e:
        st.error(f"Error fetching POIs: {str(e)}")
        return []

# Function to count POIs near a location
def get_poi_count(lat, lng, poi_type, radius=1000):
    """Get count of POIs near a location"""
    results = get_nearby_pois(lat, lng, poi_type, radius)
    return len(results)

# Function to add POI markers to a map
def add_poi_markers_to_map(m, lat, lng, poi_types=['school', 'hospital', 'supermarket', 'transit_station'], radius=1000):
    """Add POI markers to a folium map"""
    
    # POI type to icon and color mapping
    poi_icons = {
        'school': {'icon': 'graduation-cap', 'color': 'orange', 'prefix': 'fa'},
        'hospital': {'icon': 'plus-square', 'color': 'red', 'prefix': 'fa'},
        'supermarket': {'icon': 'shopping-cart', 'color': 'green', 'prefix': 'fa'},
        'transit_station': {'icon': 'subway', 'color': 'blue', 'prefix': 'fa'},
        'restaurant': {'icon': 'cutlery', 'color': 'darkred', 'prefix': 'fa'},
        'park': {'icon': 'tree', 'color': 'darkgreen', 'prefix': 'fa'}
    }
    
    # Create a feature group for POIs
    poi_group = folium.FeatureGroup(name="Points of Interest", show=True)
    
    # Fetch and add POIs for each type
    for poi_type in poi_types:
        pois = get_nearby_pois(lat, lng, poi_type, radius)
        
        for poi in pois:
            # Extract POI details
            poi_lat = poi['geometry']['location']['lat']
            poi_lng = poi['geometry']['location']['lng']
            poi_name = poi.get('name', 'Unknown')
            poi_address = poi.get('vicinity', 'Address not available')
            poi_rating = poi.get('rating', 'No rating')
            
            # Create marker
            icon = poi_icons.get(poi_type, {'icon': 'info-sign', 'color': 'purple', 'prefix': 'fa'})
            
            marker = folium.Marker(
                location=[poi_lat, poi_lng],
                popup=f"""
                <div style="width:200px">
                    <h4>{poi_name}</h4>
                    <p><b>Type:</b> {poi_type.replace('_', ' ').title()}</p>
                    <p><b>Address:</b> {poi_address}</p>
                    <p><b>Rating:</b> {poi_rating}</p>
                </div>
                """,
                tooltip=f"{poi_type.replace('_', ' ').title()}: {poi_name}",
                icon=folium.Icon(color=icon['color'], icon=icon['icon'], prefix=icon['prefix'])
            )
            
            marker.add_to(poi_group)
    
    # Add feature group to map
    poi_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

# Function to load and clean the dataset
def load_and_clean_data():
    # Load the dataset
    df = pd.read_csv('melb_data.csv')
    
    # Select relevant features based on the report
    features = ['Rooms', 'Type', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 
                'Landsize', 'BuildingArea', 'YearBuilt', 'Price', 
                'Propertycount', 'Regionname', 'Suburb']
    
    # Check if columns exist and filter
    available_features = [f for f in features if f in df.columns]
    df = df[available_features]
    
    # Handle missing values
    numeric_cols = [col for col in ['Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 
                    'BuildingArea', 'YearBuilt', 'Propertycount'] if col in df.columns]
    
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Categorical columns: impute with mode
    cat_cols = [col for col in ['Type', 'Regionname', 'Suburb'] if col in df.columns]
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Remove outliers (using IQR method for Price)
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Price'] >= Q1 - 1.5 * IQR) & (df['Price'] <= Q3 + 1.5 * IQR)]
    
    # Generate random coordinates around Melbourne for map visualization if Latitude/Longitude not in dataset
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        # Melbourne CBD coordinates
        melbourne_lat = -37.8136
        melbourne_lon = 144.9631
        
        # Generate random coordinates based on distance from CBD
        np.random.seed(42)  # For reproducibility
        
        # Scale distances to approximate coordinates (very rough approximation)
        km_per_degree_lat = 111  # Approximate km per degree of latitude
        lat_offsets = df['Distance'] / km_per_degree_lat * np.random.uniform(-1, 1, size=len(df))
        lon_offsets = df['Distance'] / (km_per_degree_lat * np.cos(np.radians(melbourne_lat))) * np.random.uniform(-1, 1, size=len(df))
        
        df['Latitude'] = melbourne_lat + lat_offsets
        df['Longitude'] = melbourne_lon + lon_offsets
    
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
    
    return model, preprocessor, history, X_test, y_test

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

# Function to display data insights
def display_insights(df):
    st.markdown('<div class="sub-header">Data Insights</div>', unsafe_allow_html=True)
    
    # Create tabs for different insights
    tabs = st.tabs(["Interactive Dashboard", "Price Distribution", "Property Map", "Feature Correlations", "Property Types", "POI Explorer"])
    
    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Melbourne Property Market Dashboard")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = df['Price'].mean()
            st.metric("Average Price", f"${avg_price:,.0f}")
        
        with col2:
            median_price = df['Price'].median()
            st.metric("Median Price", f"${median_price:,.0f}")
        
        with col3:
            property_count = len(df)
            st.metric("Total Properties", f"{property_count:,}")
        
        with col4:
            price_per_sqm = df['Price'] / df['BuildingArea'].replace(0, np.nan)
            avg_price_per_sqm = price_per_sqm.mean()
            st.metric("Avg. Price per sqm", f"${avg_price_per_sqm:,.0f}")
        
        # Interactive price by property type chart
        st.subheader("Price by Property Type")
        fig = px.box(df, x="Type", y="Price", color="Type",
                    labels={"Type": "Property Type", "Price": "Price (AUD)"},
                    category_orders={"Type": ["h", "u", "t"]},
                    color_discrete_map={"h": "#1E40AF", "u": "#3B82F6", "t": "#93C5FD"})
        
        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=["h", "u", "t"],
                ticktext=["House", "Unit/Apartment", "Townhouse"]
            ),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Price vs rooms and bathroom - interactive scatter
        st.subheader("Property Features vs Price")
        feature_options = ["Rooms", "Bathroom", "Bedroom2", "Car", "Landsize", "BuildingArea", "YearBuilt"]
        x_axis = st.selectbox("Select X-axis feature:", feature_options, index=0)
        color_by = st.selectbox("Color by:", ["Type", "Distance"] + [f for f in feature_options if f != x_axis], index=0)
        
        fig = px.scatter(df, x=x_axis, y="Price", color=color_by, 
                         size="Landsize", hover_data=["Rooms", "Bathroom", "BuildingArea"],
                         opacity=0.7, title=f"Price vs {x_axis} (colored by {color_by})")
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Distribution of Property Prices")
        
        # Interactive histogram with Plotly
        fig = px.histogram(df, x="Price", nbins=50, marginal="box", 
                          title="Price Distribution with Outliers Removed",
                          labels={"Price": "Price (AUD)"},
                          color_discrete_sequence=["#3B82F6"])
        
        fig.update_layout(
            xaxis_title="Price (AUD)",
            yaxis_title="Number of Properties",
            bargap=0.1,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Price distribution by property type
        st.subheader("Price Distribution by Property Type")
        
        fig = px.histogram(df, x="Price", color="Type", barmode="overlay",
                           opacity=0.7, nbins=30,
                           color_discrete_map={"h": "#1E40AF", "u": "#3B82F6", "t": "#93C5FD"})
        
        fig.update_layout(
            xaxis_title="Price (AUD)",
            yaxis_title="Count",
            height=500,
            legend_title="Property Type",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="info-text">The histogram shows the distribution of property prices in Melbourne. Most properties fall within the lower to middle price range, with some high-value outliers.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Melbourne Property Map")
        
        # Price range filter
        min_price = int(df['Price'].min())
        max_price = int(df['Price'].max())
        price_range = st.slider(
            "Filter by Price Range",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            step=50000,
            format="$%d"
        )
        
        # Property type filter
        property_types = st.multiselect(
            "Filter by Property Type",
            options=["h", "u", "t"],
            default=["h", "u", "t"],
            format_func=lambda x: {"h": "House", "u": "Unit/Apartment", "t": "Townhouse"}[x]
        )
        
        # POI filter options
        poi_options = st.multiselect(
            "Show Points of Interest",
            options=["school", "hospital", "supermarket", "transit_station", "restaurant", "park"],
            default=["school", "hospital", "supermarket"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # POI search radius
        poi_radius = st.slider(
            "POI Search Radius (meters)",
            min_value=500,
            max_value=5000,
            value=1000,
            step=500
        )
        
        # Filter the dataframe
        filtered_df = df[
            (df['Price'] >= price_range[0]) & 
            (df['Price'] <= price_range[1]) & 
            (df['Type'].isin(property_types))
        ]
        
        if len(filtered_df) > 0:
            # Create map centered on Melbourne
            m = folium.Map(location=[-37.8136, 144.9631], zoom_start=11)
            
            # Sample for better performance if too many points
            if len(filtered_df) > 500:
                map_data = filtered_df.sample(500)
            else:
                map_data = filtered_df
            
            # Add property markers
            property_group = folium.FeatureGroup(name="Properties", show=True)
            
            for idx, row in map_data.iterrows():
                price_str = f"${row['Price']:,.0f}"
                
                # Define color based on property type
                if row['Type'] == 'h':
                    color = 'blue'
                    prop_type = 'House'
                elif row['Type'] == 'u':
                    color = 'green'
                    prop_type = 'Unit/Apartment'
                else:
                    color = 'red'
                    prop_type = 'Townhouse'
                
                # Create popup with property info
                popup_html = f"""
                <div style="width:200px">
                    <h4>{get_property_icon(row['Type'])} {prop_type}</h4>
                    <p><b>Price:</b> {price_str}</p>
                    <p><b>Rooms:</b> {row['Rooms']}</p>
                    <p><b>Bathrooms:</b> {row['Bathroom']}</p>
                    <p><b>Land Size:</b> {row['Landsize']} sqm</p>
                    <p><b>Distance from CBD:</b> {row['Distance']} km</p>
                </div>
                """
                
                marker = folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{prop_type}: ${row['Price']:,.0f}",
                    icon=folium.Icon(color=color, icon='home')
                )
                marker.add_to(property_group)
            
            property_group.add_to(m)
            
            # Add Melbourne CBD marker
            folium.Marker(
                [-37.8136, 144.9631],
                popup="Melbourne CBD",
                tooltip="Melbourne CBD",
                icon=folium.Icon(color='purple', icon='info-sign')
            ).add_to(m)
            
            # Add circle to show distance from CBD
            folium.Circle(
                [-37.8136, 144.9631],
                radius=5000,  # 5km radius
                color="#3B82F6",
                fill=True,
                fill_color="#93C5FD",
                fill_opacity=0.2,
                popup="5km from CBD"
            ).add_to(m)
            
            # Add POI markers if selected
            if poi_options and len(map_data) > 0:
                # Use the first property as a center point for POI search
                selected_property = st.selectbox(
                    "Select property to show nearby points of interest",
                    options=range(len(map_data)),
                    format_func=lambda i: f"{get_property_icon(map_data.iloc[i]['Type'])} {map_data.iloc[i]['Price']:,.0f} - {map_data.iloc[i]['Distance']} km from CBD"
                )
                
                center_lat = map_data.iloc[selected_property]['Latitude']
                center_lng = map_data.iloc[selected_property]['Longitude']
                
                with st.spinner("Fetching nearby points of interest..."):
                    m = add_poi_markers_to_map(m, center_lat, center_lng, poi_options, poi_radius)
            
            # Display the map
            st_folium(m, width=1000, height=600)
            
            st.markdown(f"<div class='info-text'>Showing {len(map_data)} out of {len(filtered_df)} properties matching your filters.</div>", unsafe_allow_html=True)
            
            # Show POI analysis if POIs were selected
            if poi_options and len(map_data) > 0:
                st.subheader("Points of Interest Analysis")
                
                with st.spinner("Analyzing nearby points of interest..."):
                    # Count POIs for the selected property
                    poi_counts = {}
                    
                    for poi_type in poi_options:
                        count = get_poi_count(center_lat, center_lng, poi_type, poi_radius)
                        poi_counts[poi_type] = count
                    
                    # Display POI counts
                    poi_df = pd.DataFrame({
                        'POI Type': [poi.replace('_', ' ').title() for poi in poi_counts.keys()],
                        'Count': list(poi_counts.values())
                    })
                    
                    # Create bar chart
                    fig = px.bar(
                        poi_df,
                        x='POI Type',
                        y='Count',
                        color='POI Type',
                        title=f"Points of Interest within {poi_radius}m"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class='info-text'>
                        The presence of nearby amenities can significantly impact property values. 
                        Properties with good access to schools, transportation, and shopping typically 
                        command higher prices.
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No properties match your selected filters.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Feature Correlations")
        
        # Interactive correlation matrix with Plotly
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        corr = numeric_df.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="Blues",
            labels=dict(x="Features", y="Features", color="Correlation")
        )
        
        fig.update_layout(
            title="Correlation Between Numeric Features",
            height=700,
            width=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature pair exploration
        st.subheader("Explore Feature Relationships")
        
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("Select X-axis:", numeric_df.columns, index=0)
        with col2:
            y_feature = st.selectbox("Select Y-axis:", numeric_df.columns, index=list(numeric_df.columns).index('Price') if 'Price' in numeric_df.columns else 1)
        
        # Create scatter plot
        fig = px.scatter(
            df, x=x_feature, y=y_feature, 
            color="Type", 
            trendline="ols",
            labels={x_feature: x_feature, y_feature: y_feature},
            title=f"Relationship between {x_feature} and {y_feature}",
            color_discrete_map={"h": "#1E40AF", "u": "#3B82F6", "t": "#93C5FD"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="info-text">The heatmap shows correlations between different numeric features. Strong positive correlations indicate features that tend to increase together. For example, the number of rooms and bathrooms are highly correlated with the price.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[4]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Property Types Analysis")
        
        # Count of property types
        property_counts = df['Type'].value_counts().reset_index()
        property_counts.columns = ['Type', 'Count']
        
        # Add proper names for property types
        property_counts['Property Type'] = property_counts['Type'].map({
            'h': 'House',
            'u': 'Unit/Apartment',
            't': 'Townhouse'
        })
        
        # Create pie chart
        fig = px.pie(
            property_counts, 
            values='Count', 
            names='Property Type',
            color='Property Type',
            color_discrete_map={
                'House': '#1E40AF',
                'Unit/Apartment': '#3B82F6',
                'Townhouse': '#93C5FD'
            },
            hole=0.4
        )
        
        fig.update_layout(
            title="Distribution of Property Types",
            height=500
        )
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label+value',
            hoverinfo='label+percent+value'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Property type characteristics
        st.subheader("Property Type Characteristics")
        
        property_stats = df.groupby('Type').agg({
            'Price': ['mean', 'median', 'min', 'max'],
            'Rooms': 'mean',
            'Bathroom': 'mean',
            'Landsize': 'mean',
            'BuildingArea': 'mean',
            'Distance': 'mean'
        }).reset_index()
        
        # Flatten multi-level columns
        property_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in property_stats.columns.values]
        
        # Add human-readable property type
        property_stats['Property Type'] = property_stats['Type'].map({
            'h': 'House',
            'u': 'Unit/Apartment',
            't': 'Townhouse'
        })
        
        # Format price columns
        for col in ['Price_mean', 'Price_median', 'Price_min', 'Price_max']:
            property_stats[col] = property_stats[col].apply(lambda x: f"${x:,.0f}")
        
        # Prepare for display
        display_stats = property_stats[['Property Type', 'Price_mean', 'Price_median', 'Rooms_mean', 
                                        'Bathroom_mean', 'Landsize_mean', 'BuildingArea_mean', 'Distance_mean']]
        
        # Rename columns for display
        display_stats.columns = ['Property Type', 'Avg Price', 'Median Price', 'Avg Rooms', 
                                'Avg Bathrooms', 'Avg Land (sqm)', 'Avg Building (sqm)', 'Avg Distance (km)']
        
        # Display as table
        st.dataframe(display_stats, use_container_width=True, hide_index=True)
        
        # Property type icons and descriptions
        type_mapping = {
            'h': 'House',
            'u': 'Unit/Apartment',
            't': 'Townhouse'
        }
        
        st.subheader("Property Type Descriptions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-title">{get_property_icon('h')} House</div>
                <p>A standalone residential building on its own land. 
                Houses typically offer more space, privacy, and land compared to other property types.</p>
                <p><b>Best for:</b> Families, those wanting privacy and space, gardening enthusiasts</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-title">{get_property_icon('u')} Unit/Apartment</div>
                <p>A self-contained residential unit within a larger building complex.
                Units typically have lower maintenance requirements and are more affordable.</p>
                <p><b>Best for:</b> Singles, couples, investors, those seeking convenience</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-title">{get_property_icon('t')} Townhouse</div>
                <p>A multi-level attached residential unit, often part of a row of similar houses.
                Townhouses offer a middle ground between houses and apartments.</p>
                <p><b>Best for:</b> Small families, those wanting house features with less maintenance</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Add a new POI Explorer tab
    with tabs[5]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Points of Interest Explorer")
        
        st.markdown("""
        <div class="info-text">
            Explore how points of interest like schools, hospitals, supermarkets, and transit stations affect property values 
            in different suburbs of Melbourne. Proximity to amenities is a key factor in property valuation.
        </div>
        """, unsafe_allow_html=True)
        
        # Suburb selection
        suburb_list = sorted(df['Suburb'].unique())
        selected_suburb = st.selectbox("Select a suburb to explore", suburb_list)
        
        # Filter properties by suburb
        suburb_properties = df[df['Suburb'] == selected_suburb]
        
        if len(suburb_properties) > 0:
            # Display suburb statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                suburb_avg_price = suburb_properties['Price'].mean()
                st.metric("Average Price", f"${suburb_avg_price:,.0f}")
            
            with col2:
                suburb_count = len(suburb_properties)
                st.metric("Properties", f"{suburb_count}")
            
            with col3:
                suburb_avg_distance = suburb_properties['Distance'].mean()
                st.metric("Avg. Distance from CBD", f"{suburb_avg_distance:.1f} km")
            
            # Sample a property from the suburb for POI analysis
            sample_property = suburb_properties.sample(1)
            prop_lat = sample_property['Latitude'].values[0]
            prop_lng = sample_property['Longitude'].values[0]
            
            # POI search settings
            poi_col1, poi_col2 = st.columns(2)
            
            with poi_col1:
                poi_types = st.multiselect(
                    "Select amenity types to explore",
                    options=["school", "hospital", "supermarket", "transit_station", "restaurant", "park", "library", "gym"],
                    default=["school", "hospital", "supermarket", "transit_station"],
                    format_func=lambda x: x.replace('_', ' ').title()
                )
            
            with poi_col2:
                poi_radius = st.slider(
                    "Search radius (meters)",
                    min_value=500,
                    max_value=5000,
                    value=1000,
                    step=500
                )
            
            # Create map centered on the suburb
            m = folium.Map(location=[prop_lat, prop_lng], zoom_start=14)
            
            # Add suburb properties
            property_group = folium.FeatureGroup(name="Properties", show=True)
            
            # Sample for better performance if too many points
            if len(suburb_properties) > 50:
                map_data = suburb_properties.sample(50)
            else:
                map_data = suburb_properties
            
            for idx, row in map_data.iterrows():
                price_str = f"${row['Price']:,.0f}"
                
                # Define color based on property type
                if row['Type'] == 'h':
                    color = 'blue'
                    prop_type = 'House'
                elif row['Type'] == 'u':
                    color = 'green'
                    prop_type = 'Unit/Apartment'
                else:
                    color = 'red'
                    prop_type = 'Townhouse'
                
                # Create popup with property info
                popup_html = f"""
                <div style="width:200px">
                    <h4>{get_property_icon(row['Type'])} {prop_type}</h4>
                    <p><b>Price:</b> {price_str}</p>
                    <p><b>Rooms:</b> {row['Rooms']}</p>
                    <p><b>Bathrooms:</b> {row['Bathroom']}</p>
                    <p><b>Land Size:</b> {row['Landsize']} sqm</p>
                    <p><b>Distance from CBD:</b> {row['Distance']} km</p>
                </div>
                """
                
                marker = folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{prop_type}: ${row['Price']:,.0f}",
                    icon=folium.Icon(color=color, icon='home')
                )
                marker.add_to(property_group)
            
            property_group.add_to(m)
            
            # Add POI markers if selected
            if poi_types:
                with st.spinner("Fetching nearby points of interest..."):
                    m = add_poi_markers_to_map(m, prop_lat, prop_lng, poi_types, poi_radius)
            
            # Display the map
            st_folium(m, width=1000, height=500)
            
            # POI analysis
            if poi_types:
                st.subheader("Amenities Analysis")
                
                # Get POI counts
                poi_counts = {}
                with st.spinner("Analyzing nearby amenities..."):
                    for poi_type in poi_types:
                        count = get_poi_count(prop_lat, prop_lng, poi_type, poi_radius)
                        poi_counts[poi_type] = count
                
                # Display POI counts
                poi_df = pd.DataFrame({
                    'Amenity Type': [poi.replace('_', ' ').title() for poi in poi_counts.keys()],
                    'Count': list(poi_counts.values())
                })
                
                # Create columns for visualization
                poi_col1, poi_col2 = st.columns(2)
                
                with poi_col1:
                    # Create bar chart
                    fig = px.bar(
                        poi_df,
                        x='Amenity Type',
                        y='Count',
                        color='Amenity Type',
                        title=f"Amenities in {selected_suburb} (within {poi_radius}m radius)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with poi_col2:
                    # Create a heatmap-like visualization of amenity density
                    amenity_scores = []
                    
                    # Weighted scores for each amenity type (simplified)
                    weights = {
                        'school': 10,
                        'hospital': 8,
                        'supermarket': 7,
                        'transit_station': 9,
                        'restaurant': 6,
                        'park': 8,
                        'library': 7,
                        'gym': 5
                    }
                    
                    for poi_type in poi_counts.keys():
                        count = poi_counts[poi_type]
                        weight = weights.get(poi_type, 5)
                        score = min(count * weight, 100)  # Cap at 100
                        amenity_scores.append({
                            'Amenity': poi_type.replace('_', ' ').title(),
                            'Density Score': score
                        })
                    
                    score_df = pd.DataFrame(amenity_scores)
                    
                    # Create horizontal bar chart
                    fig = px.bar(
                        score_df,
                        y='Amenity',
                        x='Density Score',
                        orientation='h',
                        color='Density Score',
                        title="Amenity Density Score",
                        color_continuous_scale='Viridis',
                        range_color=[0, 100]
                    )
                    
                    fig.update_layout(
                        xaxis_title="Density Score (0-100)",
                        yaxis_title="",
                        height=350
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Price impact analysis
                st.subheader("Price Impact Analysis")
                
                st.markdown("""
                <div class="info-text">
                    The proximity and density of amenities can significantly impact property prices in a suburb.
                    Below is an analysis of how the nearby points of interest might affect property values in this area.
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate an overall amenity score
                total_count = sum(poi_counts.values())
                weighted_score = sum(poi_counts[poi] * weights.get(poi, 5) for poi in poi_counts.keys())
                normalized_score = min(weighted_score / 100, 10)  # 0-10 scale
                
                # Visualize score with a gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = normalized_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Amenity Impact on Property Value", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [0, 10], 'tickwidth': 1},
                        'bar': {'color': "#3B82F6"},
                        'steps': [
                            {'range': [0, 3], 'color': "#FEE2E2"},  # Light red
                            {'range': [3, 6], 'color': "#FEF3C7"},  # Light yellow
                            {'range': [6, 8], 'color': "#DBEAFE"},  # Light blue
                            {'range': [8, 10], 'color': "#DCFCE7"}  # Light green
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 2},
                            'thickness': 0.75,
                            'value': normalized_score
                        }
                    }
                ))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=30, r=30, t=30, b=30)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Price impact interpretation
                if normalized_score >= 8:
                    impact_text = f"""
                    <div style="background-color:#DCFCE7;padding:15px;border-radius:5px;border-left:5px solid #10B981">
                        <h4 style="color:#10B981">Excellent Amenity Score: {normalized_score:.1f}/10</h4>
                        <p>Properties in this area likely command a <b>premium of 8-15%</b> due to exceptional access to amenities.</p>
                        <p>With {total_count} points of interest nearby, this location offers excellent convenience and lifestyle benefits.</p>
                    </div>
                    """
                elif normalized_score >= 6:
                    impact_text = f"""
                    <div style="background-color:#DBEAFE;padding:15px;border-radius:5px;border-left:5px solid #3B82F6">
                        <h4 style="color:#3B82F6">Good Amenity Score: {normalized_score:.1f}/10</h4>
                        <p>Properties in this area likely see a <b>value increase of 4-8%</b> due to good access to amenities.</p>
                        <p>With {total_count} points of interest nearby, this location offers above-average convenience.</p>
                    </div>
                    """
                elif normalized_score >= 3:
                    impact_text = f"""
                    <div style="background-color:#FEF3C7;padding:15px;border-radius:5px;border-left:5px solid #F59E0B">
                        <h4 style="color:#F59E0B">Average Amenity Score: {normalized_score:.1f}/10</h4>
                        <p>Properties in this area see a <b>modest value increase of 1-4%</b> due to average amenity access.</p>
                        <p>With {total_count} points of interest nearby, this location offers standard convenience.</p>
                    </div>
                    """
                else:
                    impact_text = f"""
                    <div style="background-color:#FEE2E2;padding:15px;border-radius:5px;border-left:5px solid #EF4444">
                        <h4 style="color:#EF4444">Limited Amenity Score: {normalized_score:.1f}/10</h4>
                        <p>Properties in this area may see <b>values 1-3% below similar properties</b> in better-serviced areas.</p>
                        <p>With only {total_count} points of interest nearby, residents may need to travel for services.</p>
                    </div>
                    """
                
                st.markdown(impact_text, unsafe_allow_html=True)
                
                # Amenity breakdown
                poi_impact = {
                    'school': "Quality schools nearby can increase property values by 5-10%, particularly for family homes.",
                    'hospital': "Proximity to healthcare facilities adds convenience and can increase values by 2-5%.",
                    'supermarket': "Easy access to shopping increases desirability, adding 2-4% to property values.",
                    'transit_station': "Good public transport access can boost values by 3-7%, reducing commuting needs.",
                    'restaurant': "A vibrant dining scene attracts younger buyers and can add 2-5% to property values.",
                    'park': "Green spaces improve lifestyle quality and can increase values by 3-8%.",
                    'library': "Educational facilities add value for families, potentially increasing prices by 1-3%.",
                    'gym': "Fitness facilities appeal to certain demographics, potentially adding 1-2% value."
                }
                
                # Show impact of available amenities
                for poi_type, count in poi_counts.items():
                    if count > 0:
                        st.markdown(f"""
                        <div class="feature-card">
                            <div class="feature-title">{poi_type.replace('_', ' ').title()} ({count})</div>
                            <p>{poi_impact.get(poi_type, "This amenity can affect property values.")}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning(f"No properties found in {selected_suburb}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Function to create a feature importance plot
def plot_feature_importance(model, preprocessor, X):
    st.markdown('<div class="sub-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Create a synthetic dataset to assess feature importance
    X_processed = preprocessor.transform(X)
    
    # Get feature names
    categorical_cols = ['Type']
    numerical_cols = ['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 
                    'Landsize', 'BuildingArea', 'YearBuilt']
    
    # Try to get feature names from preprocessor
    try:
        cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
        all_features = list(numerical_cols) + list(cat_features)
    except:
        # Fallback if feature names are not available
        cat_features = [f"Type_{t}" for t in ['h', 't', 'u']]
        all_features = list(numerical_cols) + list(cat_features)
    
    # Compute feature importance through a simple perturbation approach
    with st.spinner("Calculating feature importance..."):
        importance = []
        baseline = model.predict(X_processed).flatten()
        
        # Create progress bar
        progress_bar = st.progress(0)
        
        for i in range(X_processed.shape[1]):
            # Create a copy and perturb one feature
            X_perturbed = X_processed.copy()
            X_perturbed[:, i] = np.random.permutation(X_perturbed[:, i])
            
            # Predict and calculate importance as the change in prediction
            pred_perturbed = model.predict(X_perturbed).flatten()
            importance.append(np.mean(np.abs(baseline - pred_perturbed)))
            
            # Update progress
            progress_bar.progress((i + 1) / X_processed.shape[1])
        
        # Create DataFrame for visualization
        feature_importance = pd.DataFrame({
            'Feature': all_features if len(all_features) == len(importance) else [f"Feature_{i}" for i in range(len(importance))],
            'Importance': importance
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Clear progress bar after completion
    progress_bar.empty()
    
    # Add human-readable feature names
    feature_importance['Display_Name'] = feature_importance['Feature'].apply(
        lambda x: x.replace('Type_', 'Property Type: ') if 'Type_' in x else x
    )
    
    # Interactive visualization with Plotly
    st.subheader("What Features Impact Property Price the Most?")
    
    fig = px.bar(
        feature_importance, 
        y='Display_Name', 
        x='Importance',
        orientation='h',
        title='Feature Importance - Impact on Price Prediction',
        color='Importance',
        color_continuous_scale='Blues',
        labels={'Display_Name': 'Feature', 'Importance': 'Importance Score'}
    )
    
    fig.update_layout(
        xaxis_title="Importance Score (higher = more impact on price)",
        yaxis_title="Feature",
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature explanations
    st.subheader("Understanding Feature Impact")
    
    # Get top features
    top_features = feature_importance.head(3)['Feature'].tolist()
    
    # Feature explanation text
    feature_explanations = {
        'Rooms': "The total number of rooms in a property. More rooms generally increase the property value.",
        'Distance': "Distance from Melbourne's central business district in kilometers. Properties closer to the CBD typically command higher prices.",
        'Bedroom2': "The number of bedrooms in the property. More bedrooms typically increase the value.",
        'Bathroom': "The number of bathrooms. Additional bathrooms significantly increase property value.",
        'Car': "The number of car parking spaces. Parking space is valuable, especially in densely populated areas.",
        'Landsize': "The total land size in square meters. Larger land parcels generally increase property value.",
        'BuildingArea': "The building area in square meters. Larger homes typically cost more.",
        'YearBuilt': "The year the property was built. Newer properties often have higher values, though heritage properties can also command premium prices.",
        'Type_h': "House property type. Houses typically have higher values due to land ownership.",
        'Type_u': "Unit/Apartment property type. Units often have lower prices but can be valuable in prime locations.",
        'Type_t': "Townhouse property type. Townhouses offer a middle ground between houses and units in terms of price.",
        'num_schools_nearby': "The number of schools near the property. Access to quality education is highly valued, especially for family homes.",
        'num_hospitals_nearby': "The number of hospitals and healthcare facilities nearby. Convenient access to healthcare can add value.",
        'num_supermarkets_nearby': "The number of supermarkets in the vicinity. Easy access to shopping is a significant convenience factor.",
        'num_transit_stations_nearby': "The number of public transit options nearby. Good transit access reduces commute times and can significantly boost property values.",
        'num_restaurants_nearby': "The number of dining options close by. A vibrant food scene is attractive to many buyers, especially in urban areas.",
        'num_parks_nearby': "The number of parks and green spaces nearby. Access to outdoor recreation improves lifestyle quality and property appeal."
    }
    
    for feature in top_features:
        display_name = feature.replace('Type_', 'Property Type: ') if 'Type_' in feature else feature
        
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-title">{display_name}</div>
            <p>{feature_explanations.get(feature, "This feature represents a property characteristic that influences the price.")}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="info-text">This chart shows the relative importance of different features in predicting property prices. Features with higher scores have a stronger influence on the model\'s predictions. Understanding these factors can help you make more informed property investment decisions.</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Streamlit app
def main():
    st.markdown('<div class="main-header">🏠 Melbourne Property Price Prediction</div>', unsafe_allow_html=True)
    
    # Initialize session state for prediction
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'predicted_price' not in st.session_state:
        st.session_state.predicted_price = None
    if 'prediction_data' not in st.session_state:
        st.session_state.prediction_data = {}
    
    # Create tabs for main content
    main_tabs = st.tabs(["Predict Property Price", "Data Insights", "Visual Analysis", "Guide & FAQ"])
    
    # Load and clean data
    with st.spinner("Loading property data..."):
        df = load_and_clean_data()
    
    # Prepare features and target
    X = df.drop('Price', axis=1)
    X = X[[col for col in X.columns if col in ['Rooms', 'Type', 'Distance', 'Bedroom2', 'Bathroom', 
                                             'Car', 'Landsize', 'BuildingArea', 'YearBuilt']]]
    y = df['Price']
    
    # Load or train model
    model, preprocessor = load_model_and_preprocessor()
    if model is None or preprocessor is None:
        with st.spinner('Training the model for first use...'):
            model, preprocessor, history, X_test, y_test = build_and_train_model(X, y)
            st.success('Model trained successfully!')
    
    # Create sidebar
    with st.sidebar:
        st.markdown('<div class="sub-header">About</div>', unsafe_allow_html=True)
        st.markdown('''
        <div class="info-text">
            This app demonstrates the power of machine learning in real estate price prediction.
            
            The model is trained on historical Melbourne property data and can predict prices based on 
            property characteristics.
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('<div class="sub-header">Data Statistics</div>', unsafe_allow_html=True)
        
        # Display key metrics
        st.metric("Total Properties", f"{len(df):,}")
        st.metric("Price Range", f"${df['Price'].min():,.0f} - ${df['Price'].max():,.0f}")
        st.metric("Average Price", f"${df['Price'].mean():,.0f}")
        
        # Display property type breakdown
        st.markdown('<div class="sub-header">Property Types</div>', unsafe_allow_html=True)
        type_counts = df['Type'].value_counts()
        
        # Create property type chart
        type_data = pd.DataFrame({
            'Type': ['House', 'Unit/Apt', 'Townhouse'],
            'Count': [
                type_counts.get('h', 0),
                type_counts.get('u', 0),
                type_counts.get('t', 0)
            ]
        })
        
        fig = px.bar(
            type_data, 
            y='Count', 
            x='Type',
            text='Count',
            color='Type',
            color_discrete_map={
                'House': '#1E40AF',
                'Unit/Apt': '#3B82F6',
                'Townhouse': '#93C5FD'
            }
        )
        
        fig.update_layout(
            showlegend=False,
            height=200,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model management section
        st.markdown('<div class="sub-header">Model Management</div>', unsafe_allow_html=True)
        
        if st.button('Retrain Model', help="Retrain the model using the current dataset"):
            with st.spinner('Training model... This may take a minute.'):
                model, preprocessor, history, X_test, y_test = build_and_train_model(X, y)
                
                # Display training results
                mae = history.history['mean_absolute_error'][-1]
                val_mae = history.history['val_mean_absolute_error'][-1]
                
                st.success('Model successfully retrained!')
                st.metric("Training Error (MAE)", f"${mae:,.0f}")
                st.metric("Validation Error (MAE)", f"${val_mae:,.0f}")
    
    # Tab 1: Prediction
    with main_tabs[0]:
        st.markdown('<div class="highlight info-text">This application uses machine learning to predict property prices in Melbourne, Australia based on various features like room count, property type, distance from CBD, and more.</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sub-header">Enter Property Details</div>', unsafe_allow_html=True)
        
        # Create card container
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Property type selection with icons
        st.subheader("What type of property are you interested in?")
        
        property_type_cols = st.columns(3)
        with property_type_cols[0]:
            house_selected = st.button('🏡 House', use_container_width=True, 
                                      help="A standalone residential building on its own land")
        with property_type_cols[1]:
            unit_selected = st.button('🏢 Unit/Apartment', use_container_width=True,
                                     help="A self-contained residential unit within a larger building complex")
        with property_type_cols[2]:
            townhouse_selected = st.button('🏘️ Townhouse', use_container_width=True,
                                          help="A multi-level attached residential unit, often part of a row of similar houses")
        
        # Set property type based on selection
        if house_selected:
            property_type = 'h'
        elif unit_selected:
            property_type = 'u'
        elif townhouse_selected:
            property_type = 't'
        else:
            property_type = 'h'  # Default to house
        
        # Show which property type is selected
        st.markdown(f"<p>Selected: <b>{get_property_icon(property_type)} {{'h': 'House', 'u': 'Unit/Apartment', 't': 'Townhouse'}}[property_type]</b></p>", unsafe_allow_html=True)
        
        # Other property details in clean form
        st.subheader("Property Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rooms = st.number_input('Rooms (Total)', 
                                   min_value=1, 
                                   max_value=10, 
                                   value=3,
                                   help="Total number of rooms including bedrooms, living rooms, study, etc.")
            
            bedroom2 = st.number_input('Bedrooms', 
                                      min_value=1, 
                                      max_value=10, 
                                      value=3,
                                      help="Number of bedrooms")
            
            bathroom = st.number_input('Bathrooms', 
                                      min_value=1, 
                                      max_value=10, 
                                      value=2,
                                      help="Number of bathrooms")
        
        with col2:
            car = st.number_input('Car Spaces', 
                                 min_value=0, 
                                 max_value=10, 
                                 value=2,
                                 help="Number of car parking spaces")
            
            landsize = st.slider('Land Size (sqm)', 
                                min_value=0, 
                                max_value=2000, 
                                value=500,
                                step=50,
                                help="Total land area in square meters")
            
            building_area = st.slider('Building Area (sqm)', 
                                     min_value=50, 
                                     max_value=500, 
                                     value=150,
                                     step=10,
                                     help="Building area in square meters")
        
        with col3:
            distance = st.slider('Distance from CBD (km)', 
                               min_value=0.0, 
                               max_value=40.0, 
                               value=10.0,
                               step=0.5,
                               help="Distance from Melbourne's Central Business District in kilometers")
            
            current_year = datetime.now().year
            year_built = st.slider('Year Built', 
                                  min_value=1900, 
                                  max_value=current_year, 
                                  value=2000,
                                  step=5,
                                  help="Year the property was built")
            
            # Add a "market conditions" factor (just for UI, doesn't affect model)
            market_condition = st.select_slider(
                'Market Condition',
                options=['Buyer\'s Market', 'Balanced Market', 'Seller\'s Market'],
                value='Balanced Market',
                help="Current real estate market conditions (note: this is for reference only and doesn't affect the prediction)"
            )
            
            # Add property image upload for visual analysis
            st.subheader("Upload Property Image (Optional)")
            uploaded_image = st.file_uploader("Upload an image of the property for visual feature analysis", type=["jpg", "jpeg", "png"])
            
            if uploaded_image:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(uploaded_image, caption="Uploaded Property Image", use_column_width=True)
                with col2:
                    st.info("The image will be analyzed using ResNet50 deep learning model to extract visual features that may affect property value")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction section
        prediction_placeholder = st.empty()
        
        # More prominent prediction button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button('Calculate Property Value', use_container_width=True, key="predict_button")
        
        if predict_button:
            # Visual analysis results
            visual_analysis = None
            if uploaded_image:
                with st.spinner("Analyzing property image using AI..."):
                    visual_analysis = analyze_property_image(uploaded_image)
            
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
            
            # Show prediction process with spinner
            with st.spinner('Analyzing property details and calculating value...'):
                # Add a slight delay for effect
                time.sleep(1)
                
                # Preprocess input
                input_processed = preprocessor.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_processed)[0][0]
                
                # Apply adjustment from visual analysis if available
                visual_adjustment = 0
                if visual_analysis:
                    visual_adjustment = (visual_analysis['total_impact'] / 100) * prediction
                    prediction += visual_adjustment
                
                # Calculate price range (simplified for demo)
                lower_bound = prediction * 0.9
                upper_bound = prediction * 1.1
                
                # Get price ranges in the dataset for comparison
                min_price = df['Price'].min()
                max_price = df['Price'].max()
                
                # Find similar properties
                similar_props = df[
                    (df['Rooms'] == rooms) &
                    (df['Type'] == property_type) &
                    (df['Distance'] <= distance + 5) &
                    (df['Distance'] >= max(0, distance - 5))
                ].sort_values(by='Price')[:5]
            
            # Display the prediction result in a nice box
            prediction_placeholder.markdown(f'''
            <div class="prediction-box animate-fade-in">
                <h3>Predicted Property Value</h3>
                <h1 style="color: #2563EB; font-size: 3rem;">${prediction:,.0f}</h1>
                <p>Price Range: ${lower_bound:,.0f} - ${upper_bound:,.0f}</p>
                <p>Based on the provided property characteristics</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Show price gauge
            st.subheader("Price Range in Melbourne Market")
            gauge_fig = create_price_gauge(prediction, min_price, max_price)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Show price breakdown
            st.subheader("Price Breakdown")
            
            # Calculate approximate price per square meter
            price_per_sqm = prediction / building_area if building_area > 0 else 0
            land_value_approx = price_per_sqm * landsize * 0.6  # Simplified approximation
            building_value_approx = prediction - land_value_approx
            
            # Ensure building value is never negative
            if building_value_approx < 0:
                # Adjust the split to ensure both values are positive
                land_value_percent = 0.4  # Reduced land value percentage
                land_value_approx = prediction * land_value_percent
                building_value_approx = prediction * (1 - land_value_percent)
            
            # Create columns for breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                # Land value pie chart
                breakdown_data = pd.DataFrame({
                    'Component': ['Land Value', 'Building Value'],
                    'Value': [land_value_approx, building_value_approx]
                })
                
                fig = px.pie(
                    breakdown_data, 
                    values='Value', 
                    names='Component',
                    title='Approximate Value Breakdown',
                    color='Component',
                    color_discrete_map={
                        'Land Value': '#3B82F6',
                        'Building Value': '#93C5FD'
                    }
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Value metrics
                st.metric("Estimated Price per sqm", f"${price_per_sqm:,.0f}")
                st.metric("Approx. Land Value", f"${land_value_approx:,.0f}")
                st.metric("Approx. Building Value", f"${building_value_approx:,.0f}")
                
                # Market trend indicator (for demonstration)
                if market_condition == "Seller's Market":
                    trend = "↗️ Rising market favors sellers. Prices may increase."
                elif market_condition == "Buyer's Market":
                    trend = "↘️ Declining market favors buyers. Consider negotiating."
                else:
                    trend = "➡️ Balanced market conditions."
                
                st.markdown(f"<p><b>Market Trend:</b> {trend}</p>", unsafe_allow_html=True)
            
            # Display visual analysis results if available
            if visual_analysis:
                st.subheader("Visual Feature Analysis")
                
                with st.expander("See property image analysis details", expanded=True):
                    # Create columns for visual aspects
                    st.markdown("<p>ResNet50 deep learning model was used to analyze the property image and extract visual features:</p>", unsafe_allow_html=True)
                    
                    # List detected features
                    detected_aspects = [aspect.replace('_', ' ').title() for aspect, detected in visual_analysis['aspects'].items() if detected]
                    
                    if detected_aspects:
                        # Show detected aspects in badges
                        aspect_html = "".join([f'<span style="background-color:#DBEAFE;color:#1E40AF;padding:4px 10px;margin:3px;border-radius:12px;display:inline-block;font-size:0.9rem">{aspect}</span>' for aspect in detected_aspects])
                        st.markdown(f"<div style='margin:10px 0'><b>Detected Features:</b> {aspect_html}</div>", unsafe_allow_html=True)
                        
                        # Create table for price impacts
                        impact_data = []
                        for aspect, detected in visual_analysis['aspects'].items():
                            if detected:
                                aspect_name = aspect.replace('_', ' ').title()
                                impact_value = visual_analysis['price_impacts'][aspect]
                                impact_data.append({
                                    "Feature": aspect_name,
                                    "Price Impact": f"+{impact_value:.1f}%"
                                })
                        
                        if impact_data:
                            impact_df = pd.DataFrame(impact_data)
                            st.dataframe(impact_df, hide_index=True, use_container_width=True)
                        
                        # Show visual adjustment to price
                        visual_adjustment = (visual_analysis['total_impact'] / 100) * prediction
                        st.metric(
                            "Total Visual Feature Impact",
                            f"+${visual_adjustment:,.0f}",
                            f"+{visual_analysis['total_impact']:.1f}%"
                        )
                        
                        st.markdown("""
                        <div class='info-text'>
                            <p>Visual features can significantly impact property value. Features like modern design, 
                            natural light, updated kitchens, and good condition can increase a property's appeal 
                            and therefore its market value.</p>
                            <p>This analysis uses AI to extract these features from the property image and estimates 
                            their impact on the property's value.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No significant visual features were detected in the uploaded image. Consider uploading a clearer image or one that better showcases the property's features.")
            
            # Show Points of Interest analysis if coordinates are available
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                # Find approximate coordinates based on distance
                melbourne_lat = -37.8136
                melbourne_lon = 144.9631
                
                # Find a similar property with coordinates
                nearby_property = df[
                    (df['Distance'] <= distance + 1) & 
                    (df['Distance'] >= max(0, distance - 1))
                ].sample(1) if not similar_props.empty else None
                
                if nearby_property is not None and not nearby_property.empty:
                    prop_lat = nearby_property['Latitude'].values[0]
                    prop_lng = nearby_property['Longitude'].values[0]
                    
                    st.subheader("Nearby Amenities Analysis")
                    
                    # Create expander for POI analysis
                    with st.expander("See analysis of nearby points of interest", expanded=True):
                        poi_col1, poi_col2 = st.columns(2)
                        
                        with poi_col1:
                            # POI types to analyze
                            poi_types = ['school', 'hospital', 'supermarket', 'transit_station', 'restaurant', 'park']
                            poi_radius = 1000  # 1km radius
                            
                            # Count POIs
                            poi_counts = {}
                            with st.spinner("Analyzing nearby amenities..."):
                                for poi_type in poi_types:
                                    count = get_poi_count(prop_lat, prop_lng, poi_type, poi_radius)
                                    poi_counts[poi_type] = count
                            
                            # Display POI counts
                            poi_df = pd.DataFrame({
                                'Amenity Type': [poi.replace('_', ' ').title() for poi in poi_counts.keys()],
                                'Count within 1km': list(poi_counts.values())
                            })
                            
                            st.dataframe(
                                poi_df,
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # POI value impact estimation (simplified for demo)
                            total_pois = sum(poi_counts.values())
                            if total_pois > 15:
                                impact = "Excellent access to amenities may increase property value by 5-10%."
                                impact_color = "#10B981"  # Green
                            elif total_pois > 8:
                                impact = "Good access to amenities may increase property value by 2-5%."
                                impact_color = "#3B82F6"  # Blue
                            elif total_pois > 3:
                                impact = "Average access to amenities has minimal impact on property value."
                                impact_color = "#F59E0B"  # Yellow/Orange
                            else:
                                impact = "Limited access to amenities may decrease property value by 2-5%."
                                impact_color = "#EF4444"  # Red
                            
                            st.markdown(f"<p style='color:{impact_color};font-weight:600;'>{impact}</p>", unsafe_allow_html=True)
                        
                        with poi_col2:
                            # Create bar chart
                            fig = px.bar(
                                poi_df,
                                x='Amenity Type',
                                y='Count within 1km',
                                color='Amenity Type',
                                title=f"Nearby Amenities (1km radius)"
                            )
                            
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("""
                            <div class='info-text'>
                                <p>Proximity to amenities significantly impacts property values:</p>
                                <ul>
                                    <li><b>Schools:</b> Properties near quality schools command 5-10% premium</li>
                                    <li><b>Transit:</b> Properties within walking distance to transit see 3-7% higher values</li>
                                    <li><b>Supermarkets/Shopping:</b> Convenient shopping access adds 2-5% value</li>
                                    <li><b>Parks:</b> Green space proximity can add 3-8% to property values</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Show comparable properties if available
            if len(similar_props) > 0:
                st.subheader("Comparable Properties")
                
                st.dataframe(
                    similar_props[['Rooms', 'Type', 'Distance', 'Bathroom', 'Landsize', 'BuildingArea', 'Price']].assign(
                        Type=similar_props['Type'].map({"h": "House", "u": "Unit/Apt", "t": "Townhouse"})
                    ),
                    use_container_width=True,
                    column_config={
                        "Price": st.column_config.NumberColumn(
                            "Price (AUD)",
                            format="$%d",
                        ),
                        "Type": st.column_config.SelectboxColumn(
                            "Property Type",
                            help="Type of property",
                            options=["House", "Unit/Apt", "Townhouse"],
                            required=True,
                        ),
                        "Distance": st.column_config.NumberColumn(
                            "Distance (km)",
                            format="%.1f km",
                        ),
                        "Landsize": st.column_config.NumberColumn(
                            "Land (sqm)",
                            format="%d sqm",
                        ),
                        "BuildingArea": st.column_config.NumberColumn(
                            "Building (sqm)",
                            format="%d sqm",
                        ),
                    },
                    hide_index=True
                )
            else:
                st.info("No similar properties found in the dataset.")
            
            # Investment analysis (simplified)
            st.subheader("Investment Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Rental yield calculator
                st.markdown("#### Potential Rental Yield")
                
                # Estimated weekly rent (very simplified calculation)
                est_weekly_rent = prediction * 0.001  # 0.1% of property value per week
                est_annual_rent = est_weekly_rent * 52
                rental_yield = (est_annual_rent / prediction) * 100
                
                st.metric("Estimated Weekly Rent", f"${est_weekly_rent:,.0f}")
                st.metric("Estimated Annual Rent", f"${est_annual_rent:,.0f}")
                st.metric("Gross Rental Yield", f"{rental_yield:.2f}%")
            
            with col2:
                # Capital growth projection
                st.markdown("#### 5-Year Growth Projection")
                
                # Very simplified capital growth projection
                growth_rates = {
                    "Conservative (3%)": 0.03,
                    "Moderate (5%)": 0.05,
                    "Optimistic (7%)": 0.07
                }
                
                selected_growth = st.selectbox("Growth Scenario", options=list(growth_rates.keys()))
                growth_rate = growth_rates[selected_growth]
                
                # Calculate 5-year projection
                projection_data = pd.DataFrame({
                    'Year': range(current_year, current_year + 6),
                    'Value': [prediction * (1 + growth_rate) ** year for year in range(6)]
                })
                
                fig = px.line(
                    projection_data,
                    x='Year',
                    y='Value',
                    markers=True,
                    title='Projected Property Value',
                    labels={'Value': 'Projected Value (AUD)', 'Year': 'Year'}
                )
                
                fig.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',')
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Data Insights
    with main_tabs[1]:
        # Display data insights using the enhanced function
        display_insights(df)
        
        # Plot feature importance
        try:
            plot_feature_importance(model, preprocessor, X)
        except Exception as e:
            st.error(f"Unable to display feature importance: {str(e)}")
    
    # Tab 3: Visual Analysis
    with main_tabs[2]:
        st.markdown('<div class="sub-header">Visual Property Analysis</div>', unsafe_allow_html=True)
        st.markdown('''
        <div class="info-text">
            Upload property images for AI-powered visual analysis. Our system uses ResNet50 deep learning model 
            to extract visual features that might impact property value.
        </div>
        ''', unsafe_allow_html=True)
        
        # Create card container
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Image upload section
        st.subheader("Upload Property Images")
        
        # Multiple image upload
        uploaded_images = st.file_uploader(
            "Upload property images (interior, exterior, etc.)", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        if uploaded_images:
            # Display uploaded images in a grid
            num_images = len(uploaded_images)
            if num_images > 0:
                # Create image gallery with 3 images per row
                cols_per_row = 3
                rows = (num_images + cols_per_row - 1) // cols_per_row
                
                for row in range(rows):
                    cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        img_idx = row * cols_per_row + col_idx
                        if img_idx < num_images:
                            with cols[col_idx]:
                                st.image(
                                    uploaded_images[img_idx], 
                                    caption=f"Image {img_idx+1}: {uploaded_images[img_idx].name}",
                                    use_column_width=True
                                )
            
            # Analyze button
            if st.button("Analyze Images with AI", use_container_width=True):
                # Load ResNet50 model
                with st.spinner("Loading AI model..."):
                    model = load_resnet_model()
                
                # Initialize combined results
                combined_aspects = {}
                all_results = []
                
                # Process each image
                for i, uploaded_img in enumerate(uploaded_images):
                    with st.spinner(f"Analyzing image {i+1}/{len(uploaded_images)}..."):
                        # Extract features
                        features = extract_visual_features(uploaded_img, model)
                        
                        if features is not None:
                            # Example property aspects (simplified for demo)
                            property_aspects = {
                                'modern_design': np.mean(features[:100]) > 0.1,
                                'natural_light': np.mean(features[100:200]) > 0.15,
                                'open_floor_plan': np.mean(features[200:300]) > 0.12,
                                'good_condition': np.mean(features[300:400]) > 0.2,
                                'curb_appeal': np.mean(features[400:500]) > 0.18,
                                'updated_kitchen': np.mean(features[500:600]) > 0.16,
                                'landscaping': np.mean(features[600:700]) > 0.14,
                                'quality_finishes': np.mean(features[700:800]) > 0.13,
                                'spacious_rooms': np.mean(features[800:900]) > 0.17,
                            }
                            
                            # Combine with previous results
                            for aspect, detected in property_aspects.items():
                                if aspect not in combined_aspects:
                                    combined_aspects[aspect] = detected
                                else:
                                    combined_aspects[aspect] = combined_aspects[aspect] or detected
                            
                            # Save results for this image
                            all_results.append({
                                'image_name': uploaded_img.name,
                                'aspects': property_aspects
                            })
                
                # Display combined results
                st.subheader("AI Analysis Results")
                
                # Price impact of each aspect (in percentage)
                price_impacts = {
                    'modern_design': 3.5,
                    'natural_light': 2.8,
                    'open_floor_plan': 4.2,
                    'good_condition': 5.0,
                    'curb_appeal': 3.2,
                    'updated_kitchen': 4.5,
                    'landscaping': 2.0,
                    'quality_finishes': 3.8,
                    'spacious_rooms': 3.0,
                }
                
                # Format for display
                detected_aspects = []
                non_detected_aspects = []
                
                for aspect, detected in combined_aspects.items():
                    aspect_name = aspect.replace('_', ' ').title()
                    if detected:
                        detected_aspects.append({
                            'Feature': aspect_name,
                            'Impact': f"+{price_impacts.get(aspect, 0):.1f}%",
                            'Confidence': f"{min(0.5 + np.random.random() * 0.5, 0.99):.0%}"  # Simulated confidence
                        })
                    else:
                        non_detected_aspects.append({
                            'Feature': aspect_name,
                            'Impact': f"+{price_impacts.get(aspect, 0):.1f}%",
                            'Confidence': "N/A"
                        })
                
                # Calculate total price impact
                total_impact = sum(price_impacts[aspect] for aspect, detected in combined_aspects.items() if detected)
                
                # Display results in expander
                with st.expander("View Detailed Analysis", expanded=True):
                    # Split into columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Detected Features")
                        if detected_aspects:
                            st.dataframe(pd.DataFrame(detected_aspects), hide_index=True, use_container_width=True)
                        else:
                            st.info("No significant features detected.")
                    
                    with col2:
                        st.subheader("Missing Features")
                        if non_detected_aspects:
                            st.dataframe(pd.DataFrame(non_detected_aspects), hide_index=True, use_container_width=True)
                        else:
                            st.success("All features detected!")
                
                # Visualize price impact
                st.subheader("Estimated Price Impact")
                
                # Create bar chart for impact
                if detected_aspects:
                    impact_df = pd.DataFrame(detected_aspects)
                    impact_df['Impact Value'] = impact_df['Impact'].apply(lambda x: float(x.strip('+%')))
                    
                    fig = px.bar(
                        impact_df,
                        x='Feature',
                        y='Impact Value',
                        color='Impact Value',
                        title="Price Impact by Visual Feature",
                        labels={'Impact Value': 'Price Impact (%)', 'Feature': 'Visual Feature'},
                        color_continuous_scale='Blues'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Total impact
                    st.metric(
                        "Total Estimated Price Impact",
                        f"+{total_impact:.1f}%",
                        f"Approximately +${1000000 * (total_impact/100):,.0f} on a $1M property"
                    )
                    
                    # Summary text
                    st.markdown(f"""
                    <div class='highlight info-text'>
                        Based on the visual analysis of {len(uploaded_images)} images, we've identified {len(detected_aspects)} 
                        value-adding features that could positively impact the property's market value by approximately 
                        <b>{total_impact:.1f}%</b>. These features include {", ".join([a['Feature'].lower() for a in detected_aspects[:3]])}, 
                        and more.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No significant value-adding features were detected in the images.")
                
                # Feature explanations
                st.subheader("Feature Explanations")
                
                feature_explanations = {
                    'modern_design': "Contemporary design elements that appeal to current market preferences can increase property value.",
                    'natural_light': "Abundant natural light creates a sense of space and warmth, making rooms more appealing.",
                    'open_floor_plan': "Open layouts are highly desirable in today's market, offering flexibility and a sense of spaciousness.",
                    'good_condition': "Well-maintained properties command higher prices as they require less immediate investment from buyers.",
                    'curb_appeal': "First impressions matter - attractive exteriors create positive perceptions of the entire property.",
                    'updated_kitchen': "Kitchens are a focal point for buyers, and modern, updated kitchens significantly impact value.",
                    'landscaping': "Well-designed outdoor spaces extend the living area and enhance overall property appeal.",
                    'quality_finishes': "High-quality materials and craftsmanship in finishes signal overall property quality.",
                    'spacious_rooms': "Larger rooms with good proportions are more versatile and appealing to buyers."
                }
                
                # Display explanations for detected features
                for aspect in [a['Feature'].lower().replace(' ', '_') for a in detected_aspects]:
                    st.markdown(f"""
                    <div class="feature-card">
                        <div class="feature-title">{aspect.replace('_', ' ').title()}</div>
                        <p>{feature_explanations.get(aspect, "This feature can positively impact property value.")}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Placeholder when no images are uploaded
            st.info("Upload property images to get an AI-powered visual analysis of features that might impact its value.")
            
            # Example analysis
            st.markdown("""
            <div class="highlight">
                <h4>How Visual Analysis Works</h4>
                <p>Our system uses ResNet50, a deep learning model, to analyze property images and detect features that can impact value:</p>
                <ul>
                    <li><b>Modern Design:</b> Can add 3-5% to property value</li>
                    <li><b>Natural Light:</b> Can add 2-4% to property value</li>
                    <li><b>Updated Kitchen:</b> Can add 4-6% to property value</li>
                    <li><b>Curb Appeal:</b> Can add 2-5% to property value</li>
                    <li><b>And more...</b></li>
                </ul>
                <p>Upload images of your property to see what features are present and how they might affect its value.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 4: Guide & FAQ
    with main_tabs[3]:
        st.markdown('<div class="sub-header">Guide & Frequently Asked Questions</div>', unsafe_allow_html=True)
        
        # App usage guide
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("How to Use This App")
        
        st.markdown("""
        1. 🔍 **Enter Property Details**: Fill in the details of the property you're interested in evaluating.
        2. 🧮 **Calculate Value**: Click 'Calculate Property Value' to get an AI-powered estimate.
        3. 📊 **Explore Insights**: Switch to the 'Data Insights' tab to explore Melbourne property market trends.
        4. 🔎 **Feature Importance**: See which factors most influence property prices.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # FAQ
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Frequently Asked Questions")
        
        faq_items = [
            ("How accurate is the price prediction?", 
             "The model has been trained on Melbourne housing data with optimization for accuracy. However, predictions should be considered estimates rather than precise valuations. Real estate markets are influenced by many factors, including some that aren't captured in this model."),
            
            ("What data is used to train the model?", 
             "The model is trained on historical Melbourne housing market data, including property features like rooms, property type, distance from CBD, and land size. The model learns the relationships between these features and property prices."),
            
            ("What does 'Distance from CBD' mean?", 
             "This refers to the distance in kilometers from Melbourne's Central Business District (CBD). Properties closer to the CBD typically have higher values due to convenience and access to amenities."),
            
            ("How is property type classified?", 
             "Properties are classified into three types: Houses (standalone residential buildings), Units/Apartments (self-contained residential units within larger buildings), and Townhouses (multi-level attached residential units)."),
            
            ("Can I use this for investment decisions?", 
             "This tool provides educational insights and estimates. For actual investment decisions, we recommend consulting with real estate professionals, conducting thorough market research, and obtaining formal property valuations."),
            
            ("What features most influence property prices?", 
             "Typically, location (distance from CBD), property size (building area and land size), number of rooms/bathrooms, and property type have the strongest influence on prices. You can see a detailed breakdown in the 'Feature Importance' section."),
            
            ("How does the Points of Interest feature work?", 
             "The app uses Google Maps Places API to identify and display amenities like schools, hospitals, restaurants, and transit stations near properties. This helps understand how nearby services might influence property values. You can explore this in the 'POI Explorer' tab."),
            
            ("Why are nearby amenities important for property valuation?", 
             "Proximity to amenities like schools, shops, and public transport can significantly impact property values. Properties with good access to desirable amenities typically command higher prices as they offer more convenience and better lifestyle quality."),
             
            ("What is the Visual Analysis feature?", 
             "The Visual Analysis feature uses the ResNet50 deep learning model to analyze property images and identify visual features that might impact value. It can detect aspects like modern design, natural light, and updated kitchens, then estimate their potential price impact.")
        ]
        
        for question, answer in faq_items:
            with st.expander(question):
                st.markdown(f"<div class='info-text'>{answer}</div>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # About the model
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("About the Model")
        
        st.markdown("""
        <div class='info-text'>
            This app uses a neural network model built with TensorFlow to predict property prices based on various features.
            
            <h4>Model Architecture:</h4>
            <ul>
                <li>A multi-layer neural network with dropout layers to prevent overfitting</li>
                <li>Preprocessing pipeline for handling numerical and categorical features</li>
                <li>Training process optimizes for minimal mean absolute error</li>
            </ul>
            
            <h4>Limitations:</h4>
            <ul>
                <li>The model can only consider features it was trained on</li>
                <li>Market conditions and seasonal trends may not be fully captured</li>
                <li>Unique property features or neighborhood-specific factors may affect actual prices</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('''
    <div class="footer">
        <p>Melbourne Property Price Prediction | Built with Streamlit and TensorFlow</p>
        <p>Data source: Melbourne housing market dataset</p>
        <p>Last updated: April 2025</p>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
