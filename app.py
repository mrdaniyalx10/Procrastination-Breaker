import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. SET UP THE WEB PAGE ---
st.set_page_config(page_title="Procrastination Breaker", page_icon="📱")
st.title("📱 Personal Procrastination Pattern Breaker")
st.write("Welcome to your AI-powered digital wellbeing dashboard!")

# --- 2. LOAD THE AI BRAIN (model.pkl) ---
# We use @st.cache_resource so it only loads the model once, making the website super fast
@st.cache_resource 
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    brain = load_model()
    encoder = brain['encoder']
    scaler = brain['scaler']
    kmeans = brain['kmeans']
    tree_model = brain['tree_model']
    q_table = brain['q_table']
    actions = brain['actions']
except FileNotFoundError:
    st.error("Error: model.pkl not found! Make sure you ran the Jupyter Notebook first.")
    st.stop()

# --- 3. THE SIDEBAR (For CSV Upload) ---
st.sidebar.header("📁 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload ActionDash CSV (data.csv)", type=["csv"])

# Initialize session state for CSV data
if 'user_csv_data' not in st.session_state:
    st.session_state.user_csv_data = None

if uploaded_file is not None:
    st.sidebar.success("CSV Loaded Successfully!")
    # Read the CSV to show the teacher you can handle real data
    # Try multiple encodings to handle different CSV file formats
    df = None
    last_error = None
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding)
            break
        except Exception as exc:
            last_error = exc
            continue

    if df is not None and not df.empty:
        st.session_state.user_csv_data = df
        st.subheader("📊 Your Phone Usage Data")
        st.dataframe(df.head()) # Shows the top 5 rows
        st.sidebar.info(f"✅ Predictions will now use your {len(df)} real records")
    else:
        st.sidebar.error("Could not read CSV file. Please check the file format.")
        if last_error is not None:
            st.sidebar.text(f"Last error: {type(last_error).__name__}: {last_error}")
else:
    st.sidebar.info("Drop your data.csv here to view historical logs.")

# --- 4. THE LIVE AI TESTER ---
st.markdown("---")
st.subheader("🚨 Live Distraction Tester")
st.write("Enter your current situation, and the AI will predict your risk and suggest an action.")

# Create two columns for the inputs
col1, col2 = st.columns(2)

with col1:
    # Get the list of apps the AI already learned from the encoder
    known_apps = encoder.classes_
    selected_app = st.selectbox("What app did you just open?", known_apps)

with col2:
    # A slider for the 24-hour time
    selected_hour = st.slider("What time is it? (24-hour format)", 0, 23, 12)

# --- 5. THE PREDICTION BUTTON ---
if st.button("Check Risk & Get Intervention"):
    # Convert the text back into numbers for the AI
    app_code = encoder.transform([selected_app])[0]
    input_data = pd.DataFrame({'Hour': [selected_hour], 'App_Code': [app_code]})
    input_scaled = scaler.transform(input_data)
    
    # 1. K-Means: Find the Risk Cluster
    cluster = kmeans.predict(input_scaled)[0]
    
    # 2. Decision Tree: Predict if it is High, Medium, or Low Risk
    risk = tree_model.predict(input_data)[0]
    
    # 3. Q-Learning (RL): Find the best action with the highest reward in the Q-Table
    best_action_index = np.argmax(q_table[cluster])
    intervention = actions[best_action_index]
    
    # --- SYNC WITH USER'S CSV DATA ---
    user_data_context = ""
    if st.session_state.user_csv_data is not None:
        csv_df = st.session_state.user_csv_data
        # Filter data for this app
        app_usage = csv_df[csv_df['App_Name'] == selected_app]
        hour_usage = csv_df[csv_df['Hour'] == selected_hour]
        combined = csv_df[(csv_df['App_Name'] == selected_app) & (csv_df['Hour'] == selected_hour)]
        
        if not combined.empty:
            total_duration = combined['Duration_Minutes'].sum()
            avg_duration = combined['Duration_Minutes'].mean()
            user_data_context = f"\n📊 **Your Data:** {len(combined)} sessions of {selected_app} at {selected_hour}:00 (Total: {total_duration}min, Avg: {avg_duration:.1f}min)"
            
            # Adjust risk based on actual usage patterns
            if total_duration > 90:  # High usage duration
                if risk == 'Low':
                    risk = 'Medium'
            elif total_duration > 120:  # Very high usage
                risk = 'High'
        elif not app_usage.empty:
            user_data_context = f"\n📊 **Your Data:** You use {selected_app} regularly, but not at {selected_hour}:00. Last recorded at Hour {int(app_usage['Hour'].iloc[-1])}"
        elif not hour_usage.empty:
            user_data_context = f"\n📊 **Your Data:** At {selected_hour}:00, you typically use: {', '.join(hour_usage['App_Name'].unique())}"
    
    # --- DISPLAY THE RESULTS ---
    st.markdown("### 🧠 AI Analysis Complete:")
    st.markdown(f"**App:** {selected_app} | **Time:** {selected_hour}:00{user_data_context}")
    
    if risk == 'High':
        st.error(f"**WARNING: HIGH RISK ZONE DETECTED!** (Cluster {cluster})")
        st.warning(f"🛑 **Required Intervention:** {intervention}")
    elif risk == 'Medium':
        st.warning(f"**CAUTION: MEDIUM RISK.** (Cluster {cluster})")
        st.info(f"💡 **Suggestion:** Try {intervention} to stay focused.")
    else:
        st.success(f"**ALL CLEAR: LOW RISK.** (Cluster {cluster})")
        st.write("You are being productive. Keep it up!")