import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# Page Configuration
st.set_page_config(page_title="ADA 442 - Bank Prediction System", page_icon="🏦", layout="centered")

# --- 1. CRITICAL: FeatureEngineer Class Definition ---
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['age_group'] = pd.cut(X['age'], bins=[0, 30, 40, 50, 60, 100], 
                                labels=['Young', 'Middle', 'Senior', 'PreRetired', 'Retired']).astype(str)
        X['campaign_intensity'] = pd.cut(X['campaign'], bins=[0, 1, 3, 5, 100], 
                                          labels=['Single', 'Low', 'Medium', 'High']).astype(str)
        X['was_contacted_before'] = (X['pdays'] != 999).astype(int)
        X['economic_pressure'] = X['emp.var.rate'] * X['euribor3m']
        X['is_cellular'] = (X['contact'] == 'cellular').astype(int)
        
        if 'duration' in X.columns:
            X = X.drop(columns=['duration'])
        return X

# --- 2. Load Model ---
@st.cache_resource
def load_model():
    return joblib.load("bank_marketing_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model. Ensure 'bank_marketing_model.pkl' exists. Error: {e}")

# --- 3. App UI ---
st.title("🏦 Bank Term Deposit Prediction")
st.markdown("""
This system uses **Neural Networks (MLP)** and **Advanced Feature Engineering** to predict the probability of a client subscribing to a term deposit.
""")

st.sidebar.header("Customer Input Panel")

def get_user_input():
    age = st.sidebar.slider("Age", 18, 95, 35)
    job = st.sidebar.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
    marital = st.sidebar.selectbox("Marital Status", ['divorced', 'married', 'single', 'unknown'])
    education = st.sidebar.selectbox("Education Level", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'])
    default = st.sidebar.selectbox("Has Credit in Default?", ['no', 'yes', 'unknown'])
    housing = st.sidebar.selectbox("Has Housing Loan?", ['no', 'yes', 'unknown'])
    loan = st.sidebar.selectbox("Has Personal Loan?", ['no', 'yes', 'unknown'])
    contact = st.sidebar.selectbox("Contact Communication Type", ['cellular', 'telephone'])
    month = st.sidebar.selectbox("Last Contact Month", ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'sep'])
    day_of_week = st.sidebar.selectbox("Last Contact Day", ['mon', 'tue', 'wed', 'thu', 'fri'])
    campaign = st.sidebar.number_input("Number of Contacts (This Campaign)", 1, 50, 1)
    pdays = st.sidebar.number_input("Days Since Last Campaign Contact (999=None)", 0, 999, 999)
    previous = st.sidebar.number_input("Number of Contacts (Previous Campaign)", 0, 10, 0)
    poutcome = st.sidebar.selectbox("Previous Campaign Outcome", ['nonexistent', 'failure', 'success'])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Macroeconomic Indicators")
    emp_var_rate = st.sidebar.slider("Employment Variation Rate", -4.0, 2.0, 1.1)
    cons_price_idx = st.sidebar.slider("Consumer Price Index", 92.0, 95.0, 93.9)
    cons_conf_idx = st.sidebar.slider("Consumer Confidence Index", -50.0, -20.0, -36.0)
    euribor3m = st.sidebar.slider("Euribor 3 Month Rate", 0.0, 6.0, 4.8)
    nr_employed = st.sidebar.slider("Number of Employees", 4900, 5300, 5228)

    user_data = {
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
        'month': month, 'day_of_week': day_of_week, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
    }
    return pd.DataFrame(user_data, index=[0])

input_df = get_user_input()

st.subheader("Entered Profile Summary")
st.dataframe(input_df)

# --- 4. Prediction ---
if st.button("🔮 Calculate Probability"):
    with st.spinner("Neural Network is analyzing..."):
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")
    
    if prediction[0] == 1:
        st.success(f"### ✅ LIKELY TO SUBSCRIBE")
        st.write(f"The probability of this client opening a term deposit is: **{probability*100:.1f}%**")
        st.balloons()
    else:
        st.error(f"### ❌ UNLIKELY TO SUBSCRIBE")
        st.write(f"The probability of this client opening a term deposit is: **{probability*100:.1f}%**")

st.markdown("---")
st.caption("ADA 442 Statistical Learning Final Project | Streamlit Cloud Deployment")