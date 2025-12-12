# """
# Streamlit frontend for ICD code retrieval system.
# Beautiful, interactive UI for hackathon demo.
# """
# import streamlit as st
# import requests
# import json
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
# from typing import Dict, List

# # Page configuration
# st.set_page_config(
#     page_title="ICD Code Retrieval System",
#     page_icon="🏥",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 1rem;
#     }
#     .sub-header {
#         font-size: 1.2rem;
#         color: #666;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .icd-card {
#         background-color: #f8f9fa;
#         border-left: 4px solid #1f77b4;
#         padding: 1rem;
#         margin: 0.5rem 0;
#         border-radius: 0.5rem;
#     }
#     .confidence-high {
#         color: #28a745;
#         font-weight: bold;
#     }
#     .confidence-medium {
#         color: #ffc107;
#         font-weight: bold;
#     }
#     .confidence-low {
#         color: #dc3545;
#         font-weight: bold;
#     }
#     .metric-card {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 1.5rem;
#         border-radius: 1rem;
#         color: white;
#         text-align: center;
#     }
# </style>
# """, unsafe_allow_html=True)

# # API configuration
# API_URL = "http://localhost:5000"

# # Initialize session state
# if 'results' not in st.session_state:
#     st.session_state.results = None
# if 'example_loaded' not in st.session_state:
#     st.session_state.example_loaded = False


# def check_api_health():
#     """Check if Flask API is running."""
#     try:
#         response = requests.get(f"{API_URL}/health", timeout=2)
#         return response.json().get('model_loaded', False)
#     except:
#         return False


# def load_example_case():
#     """Load example patient case from API."""
#     try:
#         response = requests.get(f"{API_URL}/api/example")
#         if response.status_code == 200:
#             return response.json()['example']
#     except:
#         pass
#     return None


# def predict_icd_codes(clinical_note, lab_values, age, sex, top_k=20):
#     """Send prediction request to Flask API."""
#     payload = {
#         "clinical_note": clinical_note,
#         "lab_values": lab_values,
#         "age": age,
#         "sex": sex,
#         "top_k": top_k
#     }
    
#     try:
#         response = requests.post(
#             f"{API_URL}/api/predict",
#             json=payload,
#             timeout=30
#         )
        
#         if response.status_code == 200:
#             return response.json()
#         else:
#             return {"success": False, "error": response.json().get('error', 'Unknown error')}
#     except Exception as e:
#         return {"success": False, "error": str(e)}


# def render_confidence_badge(confidence):
#     """Render colored confidence badge."""
#     class_name = f"confidence-{confidence.lower()}"
#     return f'<span class="{class_name}">{confidence}</span>'


# def plot_top_codes_chart(results):
#     """Create interactive bar chart of top ICD codes."""
#     if not results:
#         return None
    
#     df = pd.DataFrame(results[:10])  # Top 10 for visualization
    
#     fig = go.Figure(data=[
#         go.Bar(
#             x=df['score'],
#             y=df['code'] + ': ' + df['title'].str[:40] + '...',
#             orientation='h',
#             marker=dict(
#                 color=df['score'],
#                 colorscale='Blues',
#                 showscale=True,
#                 colorbar=dict(title="Confidence Score")
#             ),
#             text=df['score'].round(3),
#             textposition='auto',
#         )
#     ])
    
#     fig.update_layout(
#         title="Top 10 ICD Codes by Confidence Score",
#         xaxis_title="Confidence Score",
#         yaxis_title="ICD Code",
#         height=500,
#         showlegend=False,
#         yaxis={'categoryorder': 'total ascending'}
#     )
    
#     return fig


# def plot_confidence_distribution(results):
#     """Create pie chart of confidence distribution."""
#     if not results:
#         return None
    
#     df = pd.DataFrame(results)
#     confidence_counts = df['confidence'].value_counts()
    
#     fig = go.Figure(data=[go.Pie(
#         labels=confidence_counts.index,
#         values=confidence_counts.values,
#         hole=0.4,
#         marker=dict(colors=['#28a745', '#ffc107', '#dc3545'])
#     )])
    
#     fig.update_layout(
#         title="Confidence Distribution",
#         height=400
#     )
    
#     return fig


# # Main UI
# st.markdown('<div class="main-header">🏥 ICD Code Retrieval System</div>', unsafe_allow_html=True)
# st.markdown(
#     '<div class="sub-header">AI-Powered Medical Coding using Two-Tower Neural Architecture</div>',
#     unsafe_allow_html=True
# )

# # Check API health
# api_healthy = check_api_health()

# if not api_healthy:
#     st.error("⚠️ Flask API is not running. Please start the Flask server first:")
#     st.code("python flask_app.py", language="bash")
#     st.stop()

# st.success("✅ Connected to Flask API")

# # Sidebar
# with st.sidebar:
#     st.header("⚙️ Configuration")
    
#     top_k = st.slider(
#         "Number of codes to retrieve",
#         min_value=5,
#         max_value=50,
#         value=20,
#         step=5
#     )
    
#     st.markdown("---")
    
#     if st.button("📋 Load Example Case", use_container_width=True):
#         example = load_example_case()
#         if example:
#             st.session_state.example_data = example
#             st.session_state.example_loaded = True
#             st.success("Example case loaded!")
#         else:
#             st.error("Failed to load example")
    
#     st.markdown("---")
    
#     st.markdown("""
#     ### 📊 About
    
#     This system uses a **two-tower neural architecture** to retrieve relevant ICD-10 codes:
    
#     - **Patient Tower**: Encodes clinical notes (ClinicalBERT) + structured EHR data
#     - **ICD Tower**: Encodes disease descriptions
#     - **Contrastive Learning**: Multi-positive InfoNCE loss
#     - **Fast Retrieval**: ANN search over 8K+ codes
    
#     ### 🎯 Model Details
#     - **Text Encoder**: BioClinicalBERT
#     - **Embedding Dim**: 768
#     - **Training**: Multi-positive contrastive learning
#     """)

# # Main content area
# tab1, tab2, tab3 = st.tabs(["📝 Patient Input", "📊 Results", "📈 Analytics"])

# with tab1:
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.subheader("Clinical Notes")
        
#         # Load example if button was clicked
#         default_note = ""
#         if st.session_state.example_loaded and 'example_data' in st.session_state:
#             default_note = st.session_state.example_data.get('clinical_note', '')
        
#         clinical_note = st.text_area(
#             "Enter patient's clinical notes, discharge summary, or medical history:",
#             value=default_note,
#             height=300,
#             placeholder="Patient presents with..."
#         )
    
#     with col2:
#         st.subheader("Demographics")
        
#         default_age = 0
#         default_sex = "M"
#         if st.session_state.example_loaded and 'example_data' in st.session_state:
#             default_age = st.session_state.example_data.get('age', 0)
#             default_sex = st.session_state.example_data.get('sex', 'M')
        
#         age = st.number_input("Age", min_value=0, max_value=120, value=default_age)
#         sex = st.selectbox("Sex", ["M", "F"], index=0 if default_sex == "M" else 1)
    
#     st.markdown("---")
#     st.subheader("Laboratory Values")
    
#     # Lab values in columns
#     lab_keys = [
#         "a1c", "glucose", "creatinine", "egfr", "ldl", "hdl",
#         "triglycerides", "wbc", "hgb", "platelets", "crp",
#         "troponin", "bnp", "alt", "ast"
#     ]
    
#     lab_labels = {
#         "a1c": "HbA1c (%)",
#         "glucose": "Glucose (mg/dL)",
#         "creatinine": "Creatinine (mg/dL)",
#         "egfr": "eGFR (mL/min)",
#         "ldl": "LDL (mg/dL)",
#         "hdl": "HDL (mg/dL)",
#         "triglycerides": "Triglycerides (mg/dL)",
#         "wbc": "WBC (×10³/μL)",
#         "hgb": "Hemoglobin (g/dL)",
#         "platelets": "Platelets (×10³/μL)",
#         "crp": "CRP (mg/L)",
#         "troponin": "Troponin (ng/mL)",
#         "bnp": "BNP (pg/mL)",
#         "alt": "ALT (U/L)",
#         "ast": "AST (U/L)"
#     }
    
#     lab_values = {}
    
#     # Create 3 columns for lab inputs
#     cols = st.columns(3)
#     for idx, lab in enumerate(lab_keys):
#         col_idx = idx % 3
#         with cols[col_idx]:
#             default_val = None
#             if st.session_state.example_loaded and 'example_data' in st.session_state:
#                 default_val = st.session_state.example_data.get('lab_values', {}).get(lab)
            
#             value = st.number_input(
#                 lab_labels.get(lab, lab),
#                 min_value=0.0,
#                 value=float(default_val) if default_val is not None else 0.0,
#                 format="%.2f",
#                 key=f"lab_{lab}"
#             )
#             lab_values[lab] = value if value > 0 else None
    
#     st.markdown("---")
    
#     # Predict button
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         if st.button("🔍 Retrieve ICD Codes", use_container_width=True, type="primary"):
#             if not clinical_note.strip():
#                 st.error("Please enter clinical notes")
#             else:
#                 with st.spinner("Running inference... This may take a moment..."):
#                     result = predict_icd_codes(
#                         clinical_note=clinical_note,
#                         lab_values=lab_values,
#                         age=age,
#                         sex=sex,
#                         top_k=top_k
#                     )
                    
#                     if result.get('success'):
#                         st.session_state.results = result['results']
#                         st.success(f"✅ Retrieved {len(result['results'])} ICD codes!")
#                         st.balloons()
#                     else:
#                         st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

# with tab2:
#     if st.session_state.results:
#         results = st.session_state.results
        
#         # Summary metrics
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("Total Codes Retrieved", len(results))
        
#         with col2:
#             high_conf = sum(1 for r in results if r['confidence'] == 'High')
#             st.metric("High Confidence", high_conf)
        
#         with col3:
#             med_conf = sum(1 for r in results if r['confidence'] == 'Medium')
#             st.metric("Medium Confidence", med_conf)
        
#         with col4:
#             avg_score = sum(r['score'] for r in results) / len(results)
#             st.metric("Avg Score", f"{avg_score:.3f}")
        
#         st.markdown("---")
        
#         # Display results
#         st.subheader("📋 Retrieved ICD Codes")
        
#         for result in results:
#             with st.expander(
#                 f"**#{result['rank']} - {result['code']}**: {result['title']} "
#                 f"({'%.3f' % result['score']})",
#                 expanded=(result['rank'] <= 3)
#             ):
#                 col1, col2 = st.columns([3, 1])
                
#                 with col1:
#                     st.markdown(f"**Code:** `{result['code']}`")
#                     st.markdown(f"**Title:** {result['title']}")
#                     if result['description']:
#                         st.markdown(f"**Description:** {result['description']}")
                
#                 with col2:
#                     confidence_html = render_confidence_badge(result['confidence'])
#                     st.markdown(f"**Confidence:** {confidence_html}", unsafe_allow_html=True)
#                     st.markdown(f"**Score:** `{result['score']:.4f}`")
#                     st.markdown(f"**Rank:** `#{result['rank']}`")
        
#         # Export results
#         st.markdown("---")
        
#         # Convert to DataFrame for export
#         df_export = pd.DataFrame(results)
#         csv = df_export.to_csv(index=False)
        
#         st.download_button(
#             label="📥 Download Results (CSV)",
#             data=csv,
#             file_name="icd_predictions.csv",
#             mime="text/csv",
#             use_container_width=True
#         )
    
#     else:
#         st.info("👈 Enter patient information and click 'Retrieve ICD Codes' to see results")

# with tab3:
#     if st.session_state.results:
#         results = st.session_state.results
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Top codes chart
#             fig1 = plot_top_codes_chart(results)
#             if fig1:
#                 st.plotly_chart(fig1, use_container_width=True)
        
#         with col2:
#             # Confidence distribution
#             fig2 = plot_confidence_distribution(results)
#             if fig2:
#                 st.plotly_chart(fig2, use_container_width=True)
        
#         # Score distribution histogram
#         st.subheader("Score Distribution")
#         df = pd.DataFrame(results)
#         fig3 = px.histogram(
#             df,
#             x='score',
#             nbins=20,
#             title="Distribution of Confidence Scores",
#             labels={'score': 'Confidence Score', 'count': 'Frequency'}
#         )
#         st.plotly_chart(fig3, use_container_width=True)
        
#         # Top codes table
#         st.subheader("Top 10 Codes (Detailed)")
#         df_top = df.head(10)[['rank', 'code', 'title', 'score', 'confidence']]
#         st.dataframe(df_top, use_container_width=True, hide_index=True)
    
#     else:
#         st.info("👈 Run a prediction first to see analytics")

# # Footer
# st.markdown("---")
# st.markdown(
#     "<div style='text-align: center; color: #666;'>"
#     "Built with ❤️ using Streamlit, Flask, and PyTorch | "
#     "Two-Tower ICD Retrieval System"
#     "</div>",
#     unsafe_allow_html=True
# )






"""
Streamlit frontend for ICD code retrieval system (FastAPI backend).
"""

import streamlit as st
import requests
import pandas as pd
from typing import Dict

# -------------------------
# Config
# -------------------------

BACKEND_URL = "http://127.0.0.1:8000"  # FastAPI
LAB_KEYS = [
    "a1c","glucose","creatinine","egfr","ldl","hdl","triglycerides",
    "wbc","hgb","platelets","crp","troponin","bnp","alt","ast"
]

st.set_page_config(
    page_title="ICD Code Retrieval System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Styling
# -------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .icd-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Backend helpers
# -------------------------

def check_api_health() -> bool:
    """Check if FastAPI backend is alive."""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        j = r.json()
        return j.get("status") == "ok"
    except Exception:
        return False


def predict_icd_codes(
    note_text: str,
    age: int,
    sex: str,
    labs: Dict[str, float],
    top_k: int
):
    """
    Call FastAPI /predict endpoint.
    Backend request model:
      note_text: str
      age: float | null
      sex: str
      labs: {lab_name: float}
      top_k: int
    Response:
      { "codes": [ {code, title, description, score}, ... ] }
    """
    payload = {
        "note_text": note_text,
        "age": age,
        "sex": sex,
        "labs": labs,
        "top_k": top_k,
    }
    try:
        r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=60)
        if r.status_code != 200:
            return None, f"Backend error {r.status_code}: {r.text}"
        return r.json(), None
    except Exception as e:
        return None, str(e)

# -------------------------
# Header
# -------------------------

st.markdown('<div class="main-header">🏥 ICD Code Retrieval System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-powered ICD-10 prediction using a two-tower neural retriever</div>',
    unsafe_allow_html=True
)

# -------------------------
# Health check
# -------------------------

if not check_api_health():
    st.error("Backend (FastAPI) is not running or not reachable at http://127.0.0.1:8000")
    st.code("uvicorn api_service:app --reload --host 0.0.0.0 --port 8000", language="bash")
    st.stop()
else:
    st.success("Connected to FastAPI backend")

# -------------------------
# Sidebar config
# -------------------------

with st.sidebar:
    st.header("Configuration")
    top_k = st.slider("Number of ICD codes to retrieve", 5, 50, 10, 5)
    st.markdown("---")
    st.markdown("""
    ### About the model
    - Two-tower retriever
    - Patient tower: ClinicalBERT + labs + demographics
    - ICD tower: code titles, descriptions, synonyms
    - Trained with multi-positive contrastive loss
    """)
    st.markdown("---")
    st.markdown("Backend: FastAPI + PyTorch")

# -------------------------
# Main layout
# -------------------------

col_main, col_side = st.columns([2, 1])

with col_main:
    st.subheader("Clinical Note")
    note_text = st.text_area(
        "Enter clinical note / discharge summary",
        height=260,
        placeholder="65-year-old male with history of type 2 diabetes, elevated HbA1c..."
    )

with col_side:
    st.subheader("Patient Info")
    age = st.number_input("Age", min_value=0, max_value=120, value=65)
    sex = st.selectbox("Sex", options=["M", "F"], index=0)

st.markdown("---")
st.subheader("Laboratory Values (optional)")

labs = {}
lab_cols = st.columns(4)
for i, key in enumerate(LAB_KEYS):
    col = lab_cols[i % 4]
    with col:
        val_str = st.text_input(key.upper(), value="", key=f"lab_{key}")
        if val_str.strip() != "":
            try:
                labs[key] = float(val_str)
            except ValueError:
                st.warning(f"Ignoring invalid value for {key}: {val_str}")

st.markdown("---")

center_col = st.columns([1, 2, 1])[1]
with center_col:
    if st.button("Retrieve ICD Codes", use_container_width=True, type="primary"):
        if not note_text.strip():
            st.error("Please enter a clinical note.")
        else:
            with st.spinner("Running retrieval..."):
                data, err = predict_icd_codes(
                    note_text=note_text,
                    age=age,
                    sex=sex,
                    labs=labs,
                    top_k=top_k,
                )
            if err:
                st.error(f"Error calling backend: {err}")
            else:
                codes = data.get("codes", [])
                if not codes:
                    st.info("No ICD codes returned.")
                else:
                    st.success(f"Retrieved {len(codes)} ICD-10 candidates")
                    # Convert to DataFrame for display
                    df = pd.DataFrame([
                        {
                            "ICD Code": c["code"],
                            "Title": c["title"],
                            "Score": round(c["score"], 4),
                            "Description": c["description"],
                        }
                        for c in codes
                    ])
                    st.dataframe(df, use_container_width=True)

                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download results as CSV",
                        data=csv_bytes,
                        file_name="icd_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Built with Streamlit, FastAPI, and PyTorch · Two-Tower ICD Retrieval Demo"
    "</div>",
    unsafe_allow_html=True,
)
