import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random

# Set page config
st.set_page_config(
    page_title="AI Skincare Recommendation System",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .brand-recommendation {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Generate comprehensive skincare dataset
@st.cache_data
def generate_skincare_dataset():
    np.random.seed(42)
    random.seed(42)
    
    # Define categories
    skin_types = ['Oily', 'Dry', 'Combination', 'Sensitive', 'Normal']
    age_groups = ['Teens', 'Young Adult', 'Adult', 'Mature']
    concerns = ['Acne', 'Aging', 'Hyperpigmentation', 'Dryness', 'Sensitivity', 'Oiliness', 'Pores', 'Dullness']
    climates = ['Humid', 'Dry', 'Moderate', 'Cold']
    lifestyles = ['Active', 'Moderate', 'Sedentary']
    
    # Skincare routines
    routines = {
        'Basic Cleansing': ['Gentle Cleanser', 'Moisturizer', 'Sunscreen'],
        'Acne Treatment': ['Salicylic Acid Cleanser', 'Benzoyl Peroxide Treatment', 'Oil-Free Moisturizer', 'Sunscreen'],
        'Anti-Aging': ['Gentle Cleanser', 'Retinol Serum', 'Hyaluronic Acid', 'Anti-Aging Moisturizer', 'Sunscreen'],
        'Hydration Focus': ['Cream Cleanser', 'Hyaluronic Acid Serum', 'Rich Moisturizer', 'Face Oil', 'Sunscreen'],
        'Sensitive Care': ['Gentle Cleanser', 'Calming Toner', 'Fragrance-Free Moisturizer', 'Mineral Sunscreen'],
        'Brightening': ['Gentle Cleanser', 'Vitamin C Serum', 'Niacinamide', 'Moisturizer', 'Sunscreen'],
        'Oil Control': ['Foaming Cleanser', 'BHA Toner', 'Lightweight Moisturizer', 'Mattifying Sunscreen']
    }
    
    # Brand recommendations
    brands = {
        'Budget': ['CeraVe', 'The Ordinary', 'Neutrogena', 'Aveeno', 'Olay'],
        'Mid-Range': ['Paula\'s Choice', 'Drunk Elephant', 'Glossier', 'Tatcha', 'Fresh'],
        'Luxury': ['SK-II', 'La Mer', 'Estée Lauder', 'Clinique', 'Kiehl\'s']
    }
    
    # Generate dataset
    data = []
    for i in range(5000):
        skin_type = random.choice(skin_types)
        age_group = random.choice(age_groups)
        primary_concern = random.choice(concerns)
        climate = random.choice(climates)
        lifestyle = random.choice(lifestyles)
        budget = random.choice(['Budget', 'Mid-Range', 'Luxury'])
        
        # Logic for routine recommendation
        if skin_type == 'Oily' and primary_concern == 'Acne':
            routine = 'Acne Treatment'
        elif skin_type == 'Dry' or primary_concern == 'Dryness':
            routine = 'Hydration Focus'
        elif age_group == 'Mature' or primary_concern == 'Aging':
            routine = 'Anti-Aging'
        elif skin_type == 'Sensitive' or primary_concern == 'Sensitivity':
            routine = 'Sensitive Care'
        elif primary_concern == 'Hyperpigmentation':
            routine = 'Brightening'
        elif skin_type == 'Oily' and primary_concern == 'Oiliness':
            routine = 'Oil Control'
        else:
            routine = 'Basic Cleansing'
        
        # Get routine steps and brand
        routine_steps = routines[routine]
        recommended_brands = random.sample(brands[budget], min(3, len(brands[budget])))
        
        data.append({
            'skin_type': skin_type,
            'age_group': age_group,
            'primary_concern': primary_concern,
            'climate': climate,
            'lifestyle': lifestyle,
            'budget': budget,
            'recommended_routine': routine,
            'routine_steps': ', '.join(routine_steps),
            'recommended_brands': ', '.join(recommended_brands)
        })
    
    return pd.DataFrame(data)

# Train ML model
@st.cache_data
def train_model(df):
    # Prepare features for ML
    le_skin = LabelEncoder()
    le_age = LabelEncoder()
    le_concern = LabelEncoder()
    le_climate = LabelEncoder()
    le_lifestyle = LabelEncoder()
    le_budget = LabelEncoder()
    le_routine = LabelEncoder()
    
    # Encode features
    X = pd.DataFrame({
        'skin_type': le_skin.fit_transform(df['skin_type']),
        'age_group': le_age.fit_transform(df['age_group']),
        'primary_concern': le_concern.fit_transform(df['primary_concern']),
        'climate': le_climate.fit_transform(df['climate']),
        'lifestyle': le_lifestyle.fit_transform(df['lifestyle']),
        'budget': le_budget.fit_transform(df['budget'])
    })
    
    y = le_routine.fit_transform(df['recommended_routine'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store label encoders for later use
    encoders = {
        'skin_type': le_skin,
        'age_group': le_age,
        'primary_concern': le_concern,
        'climate': le_climate,
        'lifestyle': le_lifestyle,
        'budget': le_budget,
        'routine': le_routine
    }
    
    return model, encoders, accuracy

# Main application
def main():
    st.markdown('<h1 class="main-header">🧴 Skincare Recommendation System</h1>', unsafe_allow_html=True)
    
    # Load data and train model
    with st.spinner('Loading skincare database and training AI model...'):
        df = generate_skincare_dataset()
        model, encoders, accuracy = train_model(df)
    
    # Sidebar for user input
    st.sidebar.header("Tell us about your skin")
    
    skin_type = st.sidebar.selectbox(
        "What's your skin type?",
        ['Oily', 'Dry', 'Combination', 'Sensitive', 'Normal']
    )
    
    age_group = st.sidebar.selectbox(
        "What's your age group?",
        ['Teens', 'Young Adult', 'Adult', 'Mature']
    )
    
    primary_concern = st.sidebar.selectbox(
        "What's your primary skin concern?",
        ['Acne', 'Aging', 'Hyperpigmentation', 'Dryness', 'Sensitivity', 'Oiliness', 'Pores', 'Dullness']
    )
    
    climate = st.sidebar.selectbox(
        "What's your climate like?",
        ['Humid', 'Dry', 'Moderate', 'Cold']
    )
    
    lifestyle = st.sidebar.selectbox(
        "How active is your lifestyle?",
        ['Active', 'Moderate', 'Sedentary']
    )
    
    budget = st.sidebar.selectbox(
        "What's your budget preference?",
        ['Budget', 'Mid-Range', 'Luxury']
    )
    
    if st.sidebar.button("Get My Skincare Recommendations", type="primary"):
        # Make prediction
        user_input = pd.DataFrame({
            'skin_type': [encoders['skin_type'].transform([skin_type])[0]],
            'age_group': [encoders['age_group'].transform([age_group])[0]],
            'primary_concern': [encoders['primary_concern'].transform([primary_concern])[0]],
            'climate': [encoders['climate'].transform([climate])[0]],
            'lifestyle': [encoders['lifestyle'].transform([lifestyle])[0]],
            'budget': [encoders['budget'].transform([budget])[0]]
        })
        
        prediction = model.predict(user_input)[0]
        recommended_routine = encoders['routine'].inverse_transform([prediction])[0]
        
        # Get routine details
        routine_info = df[df['recommended_routine'] == recommended_routine].iloc[0]
        
        # Display recommendations
        st.markdown('<h2 class="sub-header">🎯 Your Personalized Skincare Recommendations</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f'''
            <div class="recommendation-card">
                <h3>🌟 Recommended Routine: {recommended_routine}</h3>
                <p><strong>Perfect for:</strong> {skin_type} skin with {primary_concern} concerns</p>
                <p><strong>Steps:</strong> {routine_info['routine_steps']}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Detailed routine steps
            st.markdown('<h3 class="sub-header">📋 Step-by-Step Routine</h3>', unsafe_allow_html=True)
            
            routine_steps = routine_info['routine_steps'].split(', ')
            for i, step in enumerate(routine_steps, 1):
                st.write(f"**{i}.** {step}")
            
            # Brand recommendations
            st.markdown('<h3 class="sub-header">🏷️ Recommended Brands</h3>', unsafe_allow_html=True)
            
            brands = routine_info['recommended_brands'].split(', ')
            for brand in brands:
                st.markdown(f'''
                <div class="brand-recommendation">
                    <strong>{brand}</strong> - Great for {budget.lower()} conscious consumers
                </div>
                ''', unsafe_allow_html=True)
        
        with col2:
            # User profile summary
            st.markdown("### 👤 Your Profile")
            st.write(f"**Skin Type:** {skin_type}")
            st.write(f"**Age Group:** {age_group}")
            st.write(f"**Primary Concern:** {primary_concern}")
            st.write(f"**Climate:** {climate}")
            st.write(f"**Lifestyle:** {lifestyle}")
            st.write(f"**Budget:** {budget}")
            
            # Model accuracy
            st.markdown("### 🤖 AI Model Info")
            st.write(f"**Accuracy:** {accuracy:.1%}")
            st.write(f"**Training Data:** {len(df)} samples")
    
    # Data Analytics Tab
    st.markdown('<h2 class="sub-header">📊 Skincare Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Skin Type Analysis", "🎯 Concerns Distribution", "💰 Budget Preferences"])
    
    with tab1:
        fig1 = px.pie(df, names='skin_type', title='Distribution of Skin Types')
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.bar(df.groupby(['skin_type', 'recommended_routine']).size().reset_index(name='count'), 
                     x='skin_type', y='count', color='recommended_routine',
                     title='Recommended Routines by Skin Type')
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        fig3 = px.histogram(df, x='primary_concern', title='Most Common Skin Concerns')
        st.plotly_chart(fig3, use_container_width=True)
        
        concern_routine = df.groupby(['primary_concern', 'recommended_routine']).size().reset_index(name='count')
        fig4 = px.sunburst(concern_routine, path=['primary_concern', 'recommended_routine'], values='count',
                          title='Concern-Routine Relationship')
        st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        fig5 = px.box(df, x='budget', y='age_group', title='Budget Preferences by Age Group')
        st.plotly_chart(fig5, use_container_width=True)
        
        budget_dist = df['budget'].value_counts()
        fig6 = px.bar(x=budget_dist.index, y=budget_dist.values, title='Budget Distribution')
        st.plotly_chart(fig6, use_container_width=True)
    
    # Dataset overview
    st.markdown('<h2 class="sub-header">🗂️ Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Unique Routines", df['recommended_routine'].nunique())
    with col3:
        st.metric("Skin Types", df['skin_type'].nunique())
    with col4:
        st.metric("Model Accuracy", f"{accuracy:.1%}")
    
    # Show sample data
    if st.checkbox("Show Sample Data"):
        st.dataframe(df.head(100))
    
    # Download dataset
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Dataset",
        data=csv,
        file_name='skincare_dataset.csv',
        mime='text/csv'
    )

if __name__ == "__main__":
    main()
