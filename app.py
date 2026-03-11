import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Student Performance Analytics", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished look
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    [data-testid="stSidebar"] {
        background-color: #1E1E1E;
        color: white;
    }
    .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16txtl3 h2 {
        font-family: 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# CACHED FUNCTIONS
# ==========================================
@st.cache_data
def load_data(file_or_path):
    try:
        return pd.read_csv(file_or_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def prepare_data_and_train_models(df):
    """Encodes data, splits it, and trains predictive models."""
    df_processed = df.copy()
    
    # Calculate TotalScore for reading & writing only to prevent data leakage when predicting math score!
    if "reading score" in df_processed.columns and "writing score" in df_processed.columns:
        df_processed["ReadingWritingScore"] = df_processed["reading score"] + df_processed["writing score"]
    
    categorical_cols = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
    encoders = {}
    
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            encoders[col] = le
            
    if "math score" not in df_processed.columns:
        return None, None, None, None, None, {"error": "Target column 'math score' not found."}
        
    X = df_processed.drop("math score", axis=1)
    y = df_processed["math score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        predictions[name] = model.predict(X_test)
        
    # Get feature importances from Random Forest
    rf_importances = trained_models["Random Forest"].feature_importances_
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": rf_importances
    }).sort_values(by="Importance", ascending=True)
        
    return df_processed, encoders, trained_models, predictions, y_test, importance_df

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135810.png", width=60)
st.sidebar.title("EduMetrics Pro")
st.sidebar.markdown("---")

nav_selection = st.sidebar.radio(
    "Navigation", 
    ["🏠 Home & Upload", "📊 Exploratory Analysis", "🧠 Machine Learning", "🔮 Predict Score"]
)
st.sidebar.markdown("---")

# File Upload Handling in Sidebar
st.sidebar.subheader("Dataset Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Upload your StudentsPerformance.csv")

# Use local file if available and no file is uploaded
local_file_path = "StudentsPerformance.csv"
df = None

if uploaded_file is not None:
    df = load_data(uploaded_file)
elif os.path.exists(local_file_path):
    df = load_data(local_file_path)
    if nav_selection == "🏠 Home & Upload":
        st.sidebar.success(f"Loaded local `{local_file_path}` automatically.")
else:
    if nav_selection != "🏠 Home & Upload":
        st.warning("⚠️ Please navigate to 'Home' and upload a dataset to proceed.")
        st.stop()

# ==========================================
# PAGE ROUTING
# ==========================================

if nav_selection == "🏠 Home & Upload":
    st.title("🎓 Student Performance Analytics")
    st.markdown("""
        Welcome to **EduMetrics Pro**, a modern, data-driven dashboard for analyzing student demographic and academic performance.
        
        * **Explore insights** visually with interactive charts.
        * **Evaluate Machine Learning models** trained to predict academic outcomes.
        * **Simulate scenarios** to forecast math scores based on student profiles.
    """)
    
    if df is None:
        st.info("👈 Please upload the `StudentsPerformance.csv` file in the sidebar to begin building the dashboard.")
    else:
        st.success("✅ Dataset loaded successfully! You can now navigate through the dashboard sections via the sidebar.")
        
        st.markdown("### 📋 Quick Glance at the Data")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", f"{df.shape[0]:,}")
        col2.metric("Total Features", df.shape[1])
        if "math score" in df.columns:
            col3.metric("Avg Math Score", f"{df['math score'].mean():.1f}")
        if "reading score" in df.columns:
            col4.metric("Avg Reading Score", f"{df['reading score'].mean():.1f}")
            
        st.dataframe(df.head(10), use_container_width=True)
        
        with st.expander("Show Data Summary"):
            st.dataframe(df.describe().T, use_container_width=True)

elif nav_selection == "📊 Exploratory Analysis":
    if df is None: st.stop()
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Discover patterns and relationships within the student population.")
    
    tab1, tab2 = st.tabs(["Distributions", "Categorical Impacts"])
    
    with tab1:
        st.subheader("Score Distributions")
        score_cols = [col for col in ["math score", "reading score", "writing score"] if col in df.columns]
        
        if score_cols:
            selected_score = st.selectbox("Select Score to view", score_cols)
            fig = px.histogram(df, x=selected_score, marginal="box", nbins=30, 
                               color_discrete_sequence=['#636EFA'], 
                               title=f"Distribution of {selected_score.title()}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation Heatmap
            st.subheader("Feature Correlations")
            numeric_df = df.select_dtypes(include=np.number)
            corr_matrix = numeric_df.corr().round(2)
            fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                                 title="Correlation Matrix of Numerical Features")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("No score columns found in dataset.")
            
    with tab2:
        st.subheader("Demographic Impacts on Performance")
        cat_cols = [col for col in ["gender", "race/ethnicity", "lunch", "parental level of education", "test preparation course"] if col in df.columns]
        
        if score_cols and cat_cols:
            col1, col2 = st.columns([1, 3])
            with col1:
                demographic = st.radio("Select Demographic", cat_cols)
                y_metric = st.selectbox("Select Performance Metric", score_cols)
            with col2:
                fig_box = px.box(df, x=demographic, y=y_metric, color=demographic, 
                                 title=f"{y_metric.title()} by {demographic.title()}")
                # Update layout for horizontal labels
                fig_box.update_xaxes(tickangle=45)
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Required categorical columns not available for analysis.")

elif nav_selection == "🧠 Machine Learning":
    if df is None: st.stop()
    st.title("🧠 Predictive Modeling")
    st.markdown("Evaluating our robust ensemble and regression models to predict **Math Scores**.")
    
    with st.spinner("Training models..."):
        df_processed, encoders, models, predictions, y_test, importance_df = prepare_data_and_train_models(df)
        
    if isinstance(importance_df, dict) and "error" in importance_df:
        st.error(importance_df["error"])
        st.stop()
        
    st.subheader("Model Validation Performance")
    col1, col2, col3 = st.columns(3)
    
    metrics_display = zip([col1, col2, col3], predictions.keys())
    
    for col, model_name in metrics_display:
        pred = predictions[model_name]
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        
        with col:
            st.markdown(f"**{model_name}**")
            st.metric("R² Score (Accuracy)", f"{r2*100:.2f}%")
            st.metric("RMSE (Avg. Error)", f"{rmse:.2f} points")
            
    st.markdown("---")
    
    col_feat, col_scatter = st.columns(2)
    with col_feat:
        st.subheader("Feature Importance")
        st.markdown("What drives a student's math score the most? (According to Random Forest)")
        fig_imp = px.bar(importance_df, x="Importance", y="Feature", orientation='h',
                         color="Importance", color_continuous_scale="Viridis")
        fig_imp.update_layout(showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col_scatter:
        st.subheader("Prediction vs Actual (RF)")
        st.markdown("Visualizing the accuracy of our best model.")
        rf_pred = predictions["Random Forest"]
        fig_scat = px.scatter(x=y_test, y=rf_pred, opacity=0.6,
                              labels={'x': 'Actual Math Score', 'y': 'Predicted Math Score'})
        # Add ideal line
        fig_scat.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                      mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
        st.plotly_chart(fig_scat, use_container_width=True)

elif nav_selection == "🔮 Predict Score":
    if df is None: st.stop()
    st.title("🔮 Predictive Sandbox")
    st.markdown("Simulate a student's profile to forecast their expected **Math Score**.")
    
    # Needs trained models
    df_processed, encoders, models, predictions, y_test, importance_df = prepare_data_and_train_models(df)
    
    if models is None:
        st.error("Models not available.")
        st.stop()
        
    with st.container():
        st.markdown("### Profile Constructor")
        
        col1, col2 = st.columns(2)
        input_data = {}
        
        with col1:
            input_data["gender"] = st.selectbox("Gender", df["gender"].unique())
            input_data["race/ethnicity"] = st.selectbox("Race/Ethnicity", df["race/ethnicity"].unique())
            input_data["parental level of education"] = st.selectbox("Parental Education", df["parental level of education"].unique())
            
        with col2:
            input_data["lunch"] = st.selectbox("Lunch Program", df["lunch"].unique())
            input_data["test preparation course"] = st.selectbox("Test Prep Course", df["test preparation course"].unique())
            
            # Use mean values as default
            r_val = int(df["reading score"].mean()) if "reading score" in df.columns else 50
            w_val = int(df["writing score"].mean()) if "writing score" in df.columns else 50
            
            input_data["reading score"] = st.slider("Reading Score", 0, 100, r_val)
            input_data["writing score"] = st.slider("Writing Score", 0, 100, w_val)
            
        if st.button("🚀 Run Prediction Algorithm", type="primary", use_container_width=True):
            with st.spinner("Analyzing profile..."):
                # Create df from input
                input_df = pd.DataFrame([input_data])
                
                # Add engineered feature (ReadingWritingScore)
                input_df["ReadingWritingScore"] = input_df["reading score"] + input_df["writing score"]
                
                # Encode categoricals using fitted encoders
                for col in encoders:
                    if col in input_df.columns:
                        input_df[col] = encoders[col].transform(input_df[col])
                
                # Align columns
                feature_cols = models["Random Forest"].feature_names_in_
                input_df = input_df[feature_cols]
                
                # Predict
                st.markdown("### 📊 Forecast Results")
                res_col1, res_col2, res_col3 = st.columns(3)
                
                v_lr = models["Linear Regression"].predict(input_df)[0]
                v_rf = models["Random Forest"].predict(input_df)[0]
                v_gb = models["Gradient Boosting"].predict(input_df)[0]
                
                res_col1.metric("Linear Regression", f"{v_lr:.1f} / 100")
                res_col2.metric("Random Forest", f"{v_rf:.1f} / 100", delta="Best Fit", delta_color="normal")
                res_col3.metric("Gradient Boosting", f"{v_gb:.1f} / 100")
                
                st.success("Analysis complete! Experiment with different parameters to see how strongly reading/writing scores and test prep impact the math score.")
