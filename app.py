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

if uploaded_file is not None:
    st.sidebar.success("CSV Loaded Successfully!")
    # Read the CSV to show the teacher you can handle real data
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Your Phone Usage Data")
    st.dataframe(df.head()) # Shows the top 5 rows
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
    
    # 1. K-Means: Find the Risk Cluster
    cluster = kmeans.predict(input_data)[0]
    
    # 2. Decision Tree: Predict if it is High, Medium, or Low Risk
    risk = tree_model.predict(input_data)[0]
    
    # 3. Q-Learning (RL): Find the best action with the highest reward in the Q-Table
    best_action_index = np.argmax(q_table[cluster])
    intervention = actions[best_action_index]
    
    # --- DISPLAY THE RESULTS ---
    st.markdown("### 🧠 AI Analysis Complete:")
    
    if risk == 'High':
        st.error(f"**WARNING: HIGH RISK ZONE DETECTED!** (Cluster {cluster})")
        st.warning(f"🛑 **Required Intervention:** {intervention}")
    elif risk == 'Medium':
        st.warning(f"**CAUTION: MEDIUM RISK.** (Cluster {cluster})")
        st.info(f"💡 **Suggestion:** Try {intervention} to stay focused.")
    else:
        st.success(f"**ALL CLEAR: LOW RISK.** (Cluster {cluster})")
        st.write("You are being productive. Keep it up!")