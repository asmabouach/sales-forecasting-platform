import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
import streamlit_authenticator as stauth
import httpx
from src.monitoring.drift_detection import run_all_drift_checks

# ------------------- MUST BE FIRST: Page Config -------------------
st.set_page_config(page_title="Sales Forecasting Monitor", layout="wide")

# ------------------- Custom CSS for Login Card -------------------
st.markdown("""
<style>
    .stForm {
        max-width: 400px;
        margin: 0 auto;
        padding: 40px;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        background: white;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)
# Add this CSS after your existing CSS section
st.markdown("""
<style>
    /* Make primary buttons definitely blue */
    button[kind="primary"] {
        background-color: #3b82f6 !important;
        border-color: #3b82f6 !important;
    }
    
    button[kind="primary"]:hover {
        background-color: #2563eb !important;
        border-color: #2563eb !important;
    }
    
    /* Make secondary buttons grey */
    button[kind="secondary"] {
        background-color: #6b7280 !important;
        border-color: #6b7280 !important;
        color: white !important;
    }
    
    button[kind="secondary"]:hover {
        background-color: #4b5563 !important;
        border-color: #4b5563 !important;
    }
</style>
""", unsafe_allow_html=True)
# ------------------- Load Authentication Config -------------------
auth_config_path = Path(__file__).parent / "configs" / "auth.yml"

if not auth_config_path.exists():
    st.error(f"Authentication config not found at {auth_config_path}")
    st.stop()

with open(auth_config_path) as file:
    config = yaml.safe_load(file)

# Create authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

# Read the timestamp
timestamp_path = Path(__file__).parent.parent / "output" / "last_update.txt"

if timestamp_path.exists():
    with open(timestamp_path, "r") as f:
        last_update_time = f.read().strip()
else:
    last_update_time = "Never"

# ------------------- Login Form (Centered in Main) -------------------
authentication_status = st.session_state.get("authentication_status")
name = st.session_state.get("name")
username = st.session_state.get("username")

# Only show login form if user is NOT logged in
if authentication_status not in {True, False}:
    st.markdown("<h2 style='text-align: center; margin-bottom: 40px;'>🔐 Sales Forecasting Monitor</h2>", 
                unsafe_allow_html=True)
    authenticator.login(location='main', fields={'Form name': 'Please Log In'})

# Refresh session state values
authentication_status = st.session_state.get("authentication_status")
name = st.session_state.get("name")
username = st.session_state.get("username")

# ------------------- Authentication Feedback (Main Area) -------------------
if authentication_status is False:
    st.error("Username or password is incorrect")
elif authentication_status is None:
    st.warning("Please enter your username and password")

# ------------------- Stop if not authenticated -------------------
if not authentication_status:
    st.stop()

# ------------------- Sidebar: Welcome + Logout + Navigation -------------------
with st.sidebar:
    # Title with reduced spacing
    st.markdown("<h3 style='margin-bottom: 0.25rem;'>📊 Real-time Analytics Dashboard</h3>", unsafe_allow_html=True)
    #st.markdown("<h4 style='margin-top: 0; margin-bottom: 1rem;'>Real-time Analytics Dashboard</h4>", unsafe_allow_html=True)

    # Welcome & user info
    st.markdown(f"**Welcome, {name}**")
    # Get user roles and display if relevant
    user_roles = config['credentials']['usernames'][username]['roles']
    #if user_roles:
    #    st.markdown(f"<small style='color: #6c757d;'>Role: {', '.join(user_roles)}</small>", unsafe_allow_html=True)
    
    # Initialize page in session state
    if 'page' not in st.session_state:
        st.session_state.page = "Business Dashboard"
    
    # Navigation buttons - Blue buttons with tighter spacing
    st.markdown("<div style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
    
    # Business Dashboard button
    if st.button("📈 Business Dashboard", 
                type="primary" if st.session_state.page == "Business Dashboard" else "secondary",
                use_container_width=True, 
                key="nav_business"):
        st.session_state.page = "Business Dashboard"
        st.rerun()  # Add this to immediately refresh

    # Data Monitoring button (only for authorized roles)
    if 'data_team' in user_roles or 'admin' in user_roles:
        if st.button("🔍 Data Monitoring",
                    type="primary" if st.session_state.page == "Data Monitoring" else "secondary",
                    use_container_width=True,
                    key="nav_monitoring"):
            st.session_state.page = "Data Monitoring"
            st.rerun()  # Add this to immediately refresh
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dynamic Status Info
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d  %H:%M")
    st.markdown(f"""
    <div style='background:#f8f9fa; padding:0.75rem; border-radius:8px; font-size:0.9rem; margin-top:0.5rem;'>
        <div><strong>Last update</strong><br>{last_update_time}</div>
        <div style='margin:0.5rem 0;'><strong>Pipeline status</strong><br>
        <span style='color:#28a745;'>✅ Running normally</span></div>
        <div><strong>Reference cutoff</strong><br>2025-09-30</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    # Logout button - Red button
    if st.button("👋🏻 Logout", type="primary", use_container_width=True, key="logout_btn"):
        authenticator.logout()

# ------------------- Main Content (after login) -------------------
powerbi_url = "https://app.powerbi.com/reportEmbed?reportId=20c465b7-62c2-4987-a7f5-81c687739f53&autoAuth=true&ctid=e030b0c2-7438-480c-8b0c-0d7c3ae5f098"

if st.session_state.page == "Business Dashboard":
    # Header with better design
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); 
                padding: 1.5rem; 
                border-radius: 10px; 
                margin-bottom: 1rem; 
                color: white;'>
        <h1 style='margin: 0; font-size: 2rem;'>📈 Business Dashboard</h1>
        <p style='margin: 0.5rem 0 0; opacity: 0.9;'>Real-time performance tracking and insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # No scroll, fixed height container
    st.markdown("""
    <div style='border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden;'>
    """, unsafe_allow_html=True)
    
    # Power BI iframe with no scrolling in the container
    st.components.v1.iframe(
        powerbi_url, 
        height=700, 
        scrolling=False  # Turn off scrolling
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "Data Monitoring":
    # Header with better design
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); 
                padding: 1.5rem; 
                border-radius: 10px; 
                margin-bottom: 1rem; 
                color: white;'>
        <h1 style='margin: 0; font-size: 2rem;'>🔍 Data Drift Monitoring</h1>
        <p style='margin: 0.5rem 0 0; opacity: 0.9;'>Track data distribution changes and model stability</p>
    </div>
    """, unsafe_allow_html=True)

    # Information card
    st.markdown("""
    <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem; border-left: 4px solid #3b82f6;'>
        <p style='margin: 0;'><strong>Reference period</strong>: All data up to <span style='color: #dc2626;'>2025-09-30</span> (training + stable period)</p>
        <p style='margin: 0.5rem 0 0;'><strong>Current period</strong>: October 2025 onwards</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ How do we detect drift? (Simple explanation)", expanded=True):
        st.markdown("""
        We compare old (reference) and new data using two trusted statistical tests:

        - **Kolmogorov-Smirnov (KS) Test** – for numerical features (e.g., Net Sales, Profit):  
          Checks if the distribution shape changed significantly.  
          **p-value < 0.05** → **Drift detected**.

        - **Total Variation Distance (TVD)** – for categories (e.g., Client_Category):  
          Measures how much category proportions shifted.  
          **TVD > 0.10** (10% total change) → **Drift detected**.

        These methods are industry standards and easy to explain.
        """)

    # Blue "Run Drift Checks" button
    if st.button("Run Drift Checks", type="primary", use_container_width=False, key="run_drift_button"):
        st.session_state.run_drift = True
    
    if st.session_state.get('run_drift', False):
        with st.spinner("Running drift analysis via backend API..."):
            try:
                # Explicitly use POST
                response = httpx.post(
                    "http://localhost:8000/run-drift-checks",
                    timeout=300.0  # Increase timeout to 5 minutes (in case drift takes time)
                )

                # Check for server errors
                if response.status_code == 405:
                    st.error("❌ API rejected request: Wrong method (GET instead of POST). This is a code bug.")
                    st.stop()
                if response.status_code == 500:
                    st.error("Backend error. Check API logs in logs/06_api_logs/api.log")
                    st.stop()

                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    st.error(f"API Error: {data['error']} - {data.get('details', '')}")
                    st.stop()

                reports = data["reports"]

            except httpx.ConnectError:
                st.error("❌ Cannot connect to backend API. Is uvicorn running on port 8000?")
                st.code("Run in terminal: uvicorn src.api.api:app --reload --port=8000")
                st.stop()
            except httpx.TimeoutException:
                st.error("⏰ Request timed out. Drift analysis is taking too long.")
                st.info("Check if your drift_detection function is hanging or processing huge data.")
                st.stop()
            except httpx.HTTPError as e:
                st.error(f"HTTP Error: {e}")
                st.stop()

        # --- Display results ---
        if not reports:
            st.error("No drift results — check if there is data after October 2025.")
        else:
            overall_drift = any(
                any(r["drift_detected"] for r in report["results"])
                for report in reports
            )
            
            # Status card
            if overall_drift:
                st.markdown("""
                <div style='background: #fee2e2; padding: 1rem; border-radius: 10px; border-left: 4px solid #dc2626; margin-bottom: 1.5rem;'>
                    <h3 style='color: #dc2626; margin: 0;'>⚠️ Drift Detected</h3>
                    <p style='margin: 0.5rem 0 0; color: #7f1d1d;'>Significant data distribution changes found in one or more models</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: #d1fae5; padding: 1rem; border-radius: 10px; border-left: 4px solid #10b981; margin-bottom: 1.5rem;'>
                    <h3 style='color: #065f46; margin: 0;'>✅ All Stable</h3>
                    <p style='margin: 0.5rem 0 0; color: #065f46;'>No significant drift detected across all monitored models</p>
                </div>
                """, unsafe_allow_html=True)

            for report in reports:
                model_name = report["model"]
                drift_in_model = any(r["drift_detected"] for r in report["results"])

                with st.expander(f"{'⚠️' if drift_in_model else '✅'} {model_name}", expanded=True):
                    # Status badge
                    status_color = "#dc2626" if drift_in_model else "#10b981"
                    st.markdown(f"""
                    <div style='display: inline-block; background: {status_color}; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 500; margin-bottom: 1rem;'>
                        {'DRIFT DETECTED' if drift_in_model else 'NO SIGNIFICANT DRIFT'}
                    </div>
                    """, unsafe_allow_html=True)

                    df_results = pd.DataFrame(report["results"])

                    if "Client_Category" in df_results["feature"].values:
                        num_df = df_results[~df_results["feature"].str.contains("Category")]
                        cat_df = df_results[df_results["feature"].str.contains("Category")]

                        if not num_df.empty:
                            st.subheader("Numerical Features (KS Test)")
                            styled = num_df[["feature", "ks_statistic", "p_value", "status"]].style \
                                .apply(lambda row: ["background: #ffcccc" if row.status == "DRIFT" else "background: #ccffcc"] * len(row), axis=1)
                            st.dataframe(styled, use_container_width=True)

                        if not cat_df.empty:
                            st.subheader("Target Distribution (TVD)")
                            styled = cat_df[["feature", "tvd", "status"]].style \
                                .apply(lambda row: ["background: #ffcccc" if row.status == "DRIFT" else "background: #ccffcc"] * len(row), axis=1)
                            st.dataframe(styled, use_container_width=True)
                    else:
                        styled = df_results[["feature", "ks_statistic", "p_value", "status"]].style \
                            .apply(lambda row: ["background: #ffcccc" if row.status == "DRIFT" else "background: #ccffcc"] * len(row), axis=1)
                        st.dataframe(styled, use_container_width=True)

# ------------------- Footer -------------------
st.markdown("---")
st.caption("Sales Forecasting Dashboard • BeeCoders 2025")
st.caption("v1.0 • Developed by Asma Bouach • International University of Tunis (UIT)")