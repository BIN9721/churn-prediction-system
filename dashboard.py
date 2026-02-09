import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Retention AI | Churn Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make it look "Pro"
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #0056b3;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1 {
        color: #2c3e50;
    }
    h3 {
        color: #34495e;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. SIDEBAR INPUTS (PROFESSIONAL BANKING TERMINOLOGY)
# ---------------------------------------------------------
def get_user_input():
    st.sidebar.image("https://img.icons8.com/color/96/000000/bank-building.png", width=80)
    st.sidebar.title("Customer Profile")
    st.sidebar.markdown("---")

    # Group 1: Demographics
    with st.sidebar.expander("👤 Demographics", expanded=True):
        age = st.number_input("Customer Age", 18, 100, 45)
        gender = st.selectbox("Gender", ["M", "F"])
        dependent = st.slider("Dependent Count", 0, 5, 2)
        edu = st.selectbox("Education Level", 
                           ["Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate", "Unknown"], 
                           index=3)
        marital = st.selectbox("Marital Status", 
                               ["Single", "Married", "Divorced", "Unknown"], 
                               index=1)
        income = st.selectbox("Income Category", 
                              ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"], 
                              index=2)

    # Group 2: Account Information
    with st.sidebar.expander("💳 Account Info", expanded=False):
        card = st.selectbox("Card Tier", ["Blue", "Silver", "Gold", "Platinum"])
        months_book = st.number_input("Tenure (Months on Book)", 6, 60, 36, help="How long the customer has been with the bank.")
        total_rel = st.slider("Total Products Held", 1, 6, 4, help="Number of products (Cards, Loans, Savings) the customer has.")
        months_inactive = st.slider("Inactive Months (Last 12)", 0, 12, 2)
        contacts = st.slider("Contacts Count (Last 12)", 0, 10, 3, help="Number of times customer contacted support.")

    # Group 3: Financials & Transactions (CRITICAL)
    with st.sidebar.expander("💰 Financials & Behavior", expanded=True):
        credit_limit = st.number_input("Credit Limit ($)", 1000.0, 50000.0, 8500.0)
        revolving_bal = st.number_input("Total Revolving Bal ($)", 0.0, 30000.0, 1500.0, help="Unpaid balance carried over.")
        
        st.markdown("---")
        st.caption("Transaction Velocity (Last 12 Months)")
        trans_amt = st.number_input("Total Trans. Amount ($)", 0, 25000, 4500)
        trans_ct = st.number_input("Total Trans. Count", 0, 150, 70)
        
        st.markdown("---")
        st.caption("Change Rate (Q4 vs Q1) - **Critical Indicators**")
        amt_chng = st.slider("Amt Change Ratio", 0.0, 3.0, 0.75, 0.05, help="Ratio < 0.6 indicates spending drop-off.")
        ct_chng = st.slider("Count Change Ratio", 0.0, 3.0, 0.65, 0.05, help="Ratio < 0.6 indicates usage drop-off.")

    # Data Dictionary mapped to API schema
    data = {
        "Customer_Age": age,
        "Gender": gender,
        "Dependent_count": dependent,
        "Education_Level": edu,
        "Marital_Status": marital,
        "Income_Category": income,
        "Card_Category": card,
        "Months_on_book": months_book,
        "Total_Relationship_Count": total_rel,
        "Months_Inactive_12_mon": months_inactive,
        "Contacts_Count_12_mon": contacts,
        "Credit_Limit": credit_limit,
        "Total_Revolving_Bal": revolving_bal,
        "Total_Amt_Chng_Q4_Q1": amt_chng,
        "Total_Trans_Amt": trans_amt,
        "Total_Trans_Ct": trans_ct,
        "Total_Ct_Chng_Q4_Q1": ct_chng
    }
    return data

# ---------------------------------------------------------
# 3. MAIN DASHBOARD LAYOUT
# ---------------------------------------------------------
def main():
    # Header
    st.markdown("<h1>🏦 Retention AI <span style='font-size:20px; color:gray'>| Intelligent Churn Forecast System</span></h1>", unsafe_allow_html=True)
    st.markdown("**Model Engine:** XGBoost Optimized v2.0 | **Threshold Strategy:** High-Recall (0.02)")
    st.markdown("---")

    # Load Input
    input_data = get_user_input()

    # Top KPI Cards (Summary)
    st.subheader("📊 Customer Snapshot")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric("Total Spending (12M)", f"${input_data['Total_Trans_Amt']:,.0f}", delta="Annual Volume")
    with kpi2:
        st.metric("Transaction Count", input_data['Total_Trans_Ct'], delta="Frequency")
    with kpi3:
        # Highlight logic for Change Ratio
        delta_color = "normal" if input_data['Total_Ct_Chng_Q4_Q1'] > 0.7 else "inverse"
        st.metric("Activity Change (Q4/Q1)", f"{input_data['Total_Ct_Chng_Q4_Q1']:.2f}x", delta="Momentum", delta_color=delta_color)
    with kpi4:
        st.metric("Revolving Debt", f"${input_data['Total_Revolving_Bal']:,.0f}", delta="Risk Exposure")

    st.markdown("---")

    # Prediction Button
    col_btn, col_blank = st.columns([1, 2])
    with col_btn:
        analyze_btn = st.button("🚀 ANALYZE RISK PROFILE", use_container_width=True)

    # ---------------------------------------------------------
    # 4. PREDICTION LOGIC & VISUALIZATION
    # ---------------------------------------------------------
    if analyze_btn:
        with st.spinner("🔄 Processing behavioral data & calculating risk score..."):
            try:
                # API Call
                API_URL = "http://127.0.0.1:8000/predict"
                response = requests.post(API_URL, json=input_data)
                
                if response.status_code == 200:
                    result = response.json()
                    prob = result['churn_probability']
                    action = result['recommended_action']
                    
                    st.success("Analysis Complete!")
                    st.markdown("### 🎯 Risk Assessment Results")
                    
                    # Layout: Gauge Chart (Left) vs Action Plan (Right)
                    res_col1, res_col2 = st.columns([1, 1.5])

                    with res_col1:
                        # Determine Color & Status
                        if prob < 0.02:
                            color, status_text = '#28a745', "SAFE (Low Risk)"
                        elif prob < 0.5:
                            color, status_text = '#ffc107', "WATCHLIST (Medium Risk)"
                        else:
                            color, status_text = '#dc3545', "CRITICAL (High Risk)"

                        # Clean Metric Display
                        st.markdown(f"""
                        <div style="text-align: center; padding: 10px; background-color: {color}20; border-radius: 10px; border: 2px solid {color}">
                            <h2 style="color: {color}; margin:0;">{prob*100:.2f}%</h2>
                            <p style="margin:0; font-weight:bold; color: {color}">Churn Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Minimalist Bar Chart
                        fig, ax = plt.subplots(figsize=(6, 1.5))
                        ax.barh([0], [prob], color=color, height=0.5, edgecolor='black')
                        ax.barh([0], [1-prob], left=[prob], color='#e9ecef', height=0.5)
                        
                        # Add Threshold Marker
                        ax.axvline(0.02, color='#333', linestyle='--', linewidth=2)
                        ax.text(0.02, 0.4, ' Optimal Threshold (2%)', ha='left', va='center', fontsize=9, fontweight='bold')
                        
                        ax.set_xlim(0, 1)
                        ax.axis('off')
                        st.pyplot(fig, use_container_width=True)

                    with res_col2:
                        st.info(f"**Customer Status:** {status_text}")
                        
                        # Dynamic Recommendation Box
                        if prob >= 0.02:
                            st.error("⚠️ **Retention Action Required**")
                            st.markdown(f"""
                            **AI Recommendation:**
                            > {action}
                            
                            **Why this action?**
                            * Probability ({prob:.2f}) exceeds the economic safety threshold (0.02).
                            * Early intervention saves roughly **$490** in LTV per customer.
                            """)
                        else:
                            st.success("✅ **No Action Needed**")
                            st.markdown("""
                            **AI Recommendation:**
                            > Continue standard service.
                            
                            **Analysis:**
                            * Customer exhibits strong loyalty signals.
                            * Transaction velocity and engagement are healthy.
                            """)

                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.code(response.text)

            except requests.exceptions.ConnectionError:
                st.error("❌ Connection Failed!")
                st.warning("Ensure the backend API is running: `uvicorn app:app --reload`")

if __name__ == "__main__":
    main()