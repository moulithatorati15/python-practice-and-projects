import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #eef2ff,
        #f5f3ff,
        #fce7f3
    );
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Main title */
.main-title {
    text-align: center;
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed,
        #db2777
    );
    color: white;
    margin-bottom: 30px;
}

/* Prediction button */
div.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 15px;
    font-size: 20px;
    font-weight: bold;
    background: linear-gradient(
        90deg,
        #7c3aed,
        #2563eb
    );
    color: white;
    border: none;
}

/* Button hover */
div.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #db2777,
        #7c3aed
    );
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    """
    <div class="main-title">
        <h1>🏦 Loan Approval Prediction</h1>
        <p>
            Enter applicant details below to predict
            whether the loan is likely to be approved.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

model = joblib.load(
    "model/xgboost_model.pkl"
)

preprocessor = joblib.load(
    "model/preprocessor.pkl"
)

st.success("✅ XGBoost Model Loaded Successfully!")

# PREDICTION HISTORY
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# APPLICANT INFORMATION
st.header("👤 Applicant Information")

col1, col2 = st.columns(2)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )


with col2:

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

# FINANCIAL INFORMATION
st.header("💰 Financial Information")

col1, col2 = st.columns(2)


with col1:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000,
        step=100
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0,
        value=0,
        step=100
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=150,
        step=10
    )


with col2:

    loan_amount_term = st.number_input(
        "Loan Amount Term",
        min_value=0,
        value=360,
        step=10
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0]
    )

# PROPERTY INFORMATION

st.header("🏠 Property Information")

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Rural", "Semiurban"]
)

# CREATE INPUT DATAFRAME
input_data = pd.DataFrame({

    "Gender": [gender],

    "Married": [married],

    "Dependents": [dependents],

    "Education": [education],

    "Self_Employed": [self_employed],

    "ApplicantIncome": [applicant_income],

    "CoapplicantIncome": [coapplicant_income],

    "LoanAmount": [loan_amount],

    "Loan_Amount_Term": [loan_amount_term],

    "Credit_History": [credit_history],

    "Property_Area": [property_area]

})

# PREDICTION BUTTON

st.write("")

if st.button("🚀 Predict Loan Status",key="predict_button"):

    # PREPROCESS INPUT

    input_processed = preprocessor.transform(
        input_data
    )


    # MAKE PREDICTION

    prediction = model.predict(
        input_processed
    )[0]


    # GET PROBABILITY

    probability = float(
        model.predict_proba(
            input_processed
        )[0][1]
    )
    # SAVE PREDICTION HISTORY

    result = "Approved" if prediction == 1 else "Not Approved"

    st.session_state.prediction_history.append({

        "Gender": gender,

        "Education": education,

        "Property Area": property_area,

        "Applicant Income": applicant_income,

        "Loan Amount": loan_amount,

        "Probability": f"{probability * 100:.2f}%",

        "Result": result

    })
    # CLEAR PREDICTION HISTORY
if st.session_state.prediction_history:

    if st.button("🗑️ Clear Prediction History",key="clear_history_button"):

        st.session_state.prediction_history = []

        st.rerun()
    # DISPLAY RESULT

    st.subheader("✨ Prediction Result")


    if prediction == 1:

        st.success("✅ Loan Approved")

        st.metric(
            label="Approval Probability",
            value=f"{probability * 100:.2f}%"
        )

        # PROBABILITY GAUGE
        
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",

                value=probability * 100,

                number={
                    "suffix": "%"
                },

                title={
                    "text": "Loan Approval Probability"
                },

                gauge={
                    "axis": {
                        "range": [0, 100]
                    },

                    "bar": {
                        "color": "#7c3aed"
                    },

                    "steps": [

                        {
                            "range": [0, 50],
                            "color": "#fee2e2"
                        },

                        {
                            "range": [50, 75],
                            "color": "#fef3c7"
                        },

                        {
                            "range": [75, 100],
                            "color": "#dcfce7"
                        }

                    ]
                }
            )
        )


        st.plotly_chart(
            gauge,
            use_container_width=True
        )


        st.info(
            "The model predicts a high likelihood "
            "of loan approval."
        )


    else:

        st.error("❌ Loan Not Approved")

        st.metric(
            label="Approval Probability",
            value=f"{probability * 100:.2f}%"
        )

        # PROBABILITY GAUGE
        

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",

                value=probability * 100,

                number={
                    "suffix": "%"
                },

                title={
                    "text": "Loan Approval Probability"
                },

                gauge={
                    "axis": {
                        "range": [0, 100]
                    },

                    "bar": {
                        "color": "#7c3aed"
                    },

                    "steps": [

                        {
                            "range": [0, 50],
                            "color": "#fee2e2"
                        },

                        {
                            "range": [50, 75],
                            "color": "#fef3c7"
                        },

                        {
                            "range": [75, 100],
                            "color": "#dcfce7"
                        }

                    ]
                }
            )
        )


        st.plotly_chart(
            gauge,
            use_container_width=True
        )


        st.warning(
            "The model predicts a lower likelihood "
            "of loan approval."
        )

# PREDICTION HISTORY TABLE


if st.session_state.prediction_history:

    st.divider()

    st.header("📋 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

# CLEAR PREDICTION HISTORY

if st.session_state.prediction_history:

    if st.button("🗑️ Clear Prediction History"):

        st.session_state.prediction_history = []

        st.rerun()

# FOOTER
st.divider()

st.markdown(
    "### 🏦 Loan Approval Prediction System"
)

st.caption(
    "Powered by XGBoost & Streamlit"
)

st.caption(
    "🔒 Secure  •  ⚡ Fast  •  🎯 Machine Learning"
)