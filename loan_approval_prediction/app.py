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

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 20%,
                rgba(0, 200, 255, 0.18),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 80%,
                rgba(30, 80, 255, 0.20),
                transparent 30%
            ),
            radial-gradient(
                circle at 50% 50%,
                rgba(0, 120, 180, 0.08),
                transparent 40%
            ),
            linear-gradient(
                135deg,
                #020617,
                #061a33,
                #031525,
                #020617
            );

        background-size: 200% 200%;
        animation: backgroundAnimation 18s ease infinite;
        color: #ffffff !important;
    }

    @keyframes backgroundAnimation {
        0% {
            background-position: 0% 50%;
        }

        50% {
            background-position: 100% 50%;
        }

        100% {
            background-position: 0% 50%;
        }
    }

    p,
    label,
    span,
    .stMarkdown,
    .stText {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: #ffffff !important;
        font-weight: 900 !important;
    }

    .stSelectbox label,
    .stNumberInput label {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    input {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    [data-baseweb="select"] * {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        background-color: #00d9ff !important;
        border: 2px solid #00d9ff !important;
        border-radius: 12px !important;
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 18px !important;
    }

    .stButton > button p {
        color: #000000 !important;
        font-weight: 900 !important;
    }

    .stButton > button div {
        color: #000000 !important;
        font-weight: 900 !important;
    }

    .stButton > button:hover {
        background-color: #67e8f9 !important;
        border-color: #ffffff !important;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 900 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    hr {
        border-color: rgba(0, 217, 255, 0.35) !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

try:
    model = joblib.load("model/xgboost_model.pkl")
    preprocessor = joblib.load("model/preprocessor.pkl")
    model_loaded = True

except Exception as e:
    model_loaded = False
    st.error("❌ Unable to load the model files.")
    st.code(str(e))


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ============================================================
# HEADER
# ============================================================

st.title("🏦 Loan Approval Prediction")

st.markdown(
    "**Enter applicant details below to predict whether "
    "the loan is likely to be approved.**"
)

if model_loaded:
    st.success("✅ XGBoost Model Loaded Successfully!")


# ============================================================
# APPLICANT INFORMATION
# ============================================================

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


# ============================================================
# FINANCIAL INFORMATION
# ============================================================

st.header("💰 Financial Information")

col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=150.0,
        step=10.0
    )

with col2:
    loan_amount_term = st.number_input(
        "Loan Amount Term",
        min_value=0.0,
        value=360.0,
        step=10.0
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0]
    )


# ============================================================
# PROPERTY INFORMATION
# ============================================================

st.header("🏠 Property Information")

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Rural", "Semiurban"]
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")

predict_button = st.button(
    "🚀 Predict Loan Status",
    key="predict_button",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button and model_loaded:

    try:
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

        input_processed = preprocessor.transform(input_data)

        prediction = int(
            model.predict(input_processed)[0]
        )

        probability = float(
            model.predict_proba(input_processed)[0][1]
        )

        probability_percentage = probability * 100

        result = "Approved" if prediction == 1 else "Not Approved"


        # ====================================================
        # SAVE PREDICTION HISTORY
        # ====================================================

        history_record = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self Employed": self_employed,
            "Applicant Income": applicant_income,
            "Coapplicant Income": coapplicant_income,
            "Loan Amount": loan_amount,
            "Loan Term": loan_amount_term,
            "Credit History": credit_history,
            "Property Area": property_area,
            "Probability": f"{probability_percentage:.2f}%",
            "Result": result
        }

        st.session_state.prediction_history.append(history_record)


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()
        st.subheader("✨ Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.success("✅ Loan Approved")

                st.info(
                    "**APPROVED**\n\n"
                    "The model predicts a high likelihood "
                    "of loan approval."
                )
            else:
                st.error("❌ Loan Not Approved")

                st.warning(
                    "**NOT APPROVED**\n\n"
                    "The model predicts a lower likelihood "
                    "of loan approval."
                )


        # ====================================================
        # PROBABILITY GAUGE
        # ====================================================

        with result_col2:

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability_percentage,

                    number={
                        "suffix": "%",
                        "font": {
                            "size": 40,
                            "color": "#000000"
                        }
                    },

                    title={
                        "text": "Loan Approval Probability",
                        "font": {
                            "size": 20,
                            "color": "#000000"
                        }
                    },

                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickwidth": 1,
                            "tickcolor": "#000000"
                        },

                        "bar": {
                            "color": "#0891b2",
                            "thickness": 0.75
                        },

                        "bgcolor": "#ffffff",

                        "borderwidth": 2,

                        "bordercolor": "#000000",

                        "steps": [
                            {
                                "range": [0, 50],
                                "color": "#fecaca"
                            },
                            {
                                "range": [50, 75],
                                "color": "#fde68a"
                            },
                            {
                                "range": [75, 100],
                                "color": "#bbf7d0"
                            }
                        ]
                    }
                )
            )

            gauge.update_layout(
                height=320,
                margin=dict(
                    l=30,
                    r=30,
                    t=70,
                    b=20
                ),
                paper_bgcolor="#ffffff"
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )


        # ====================================================
        # PROBABILITY
        # ====================================================

        st.divider()
        st.subheader("📈 Approval Probability")

        probability_col1, probability_col2 = st.columns(2)

        with probability_col1:
            st.metric(
                "Approval Probability",
                f"{probability_percentage:.2f}%"
            )

        with probability_col2:

            if probability_percentage >= 75:
                st.success("High Probability")

            elif probability_percentage >= 50:
                st.warning("Medium Probability")

            else:
                st.error("Low Probability")


        # ====================================================
        # PREDICTION DETAILS
        # ====================================================

        st.divider()
        st.subheader("📋 Prediction Details")

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:

            st.write(f"**Result:** {result}")
            st.write(f"**Gender:** {gender}")
            st.write(f"**Married:** {married}")
            st.write(f"**Dependents:** {dependents}")
            st.write(f"**Education:** {education}")
            st.write(f"**Self Employed:** {self_employed}")

        with detail_col2:

            st.write(
                f"**Applicant Income:** "
                f"{applicant_income:,.0f}"
            )

            st.write(
                f"**Coapplicant Income:** "
                f"{coapplicant_income:,.0f}"
            )

            st.write(
                f"**Loan Amount:** "
                f"{loan_amount:,.0f}"
            )

            st.write(
                f"**Loan Term:** "
                f"{loan_amount_term:,.0f}"
            )

            st.write(
                f"**Credit History:** "
                f"{credit_history}"
            )

            st.write(
                f"**Property Area:** "
                f"{property_area}"
            )


    except Exception as e:

        st.error("❌ Prediction failed.")
        st.code(str(e))


# ============================================================
# PREDICTION HISTORY
# ============================================================

if st.session_state.prediction_history:

    st.divider()

    st.header("📊 Prediction History")

    st.write(
        f"**Total Predictions:** "
        f"{len(st.session_state.prediction_history)}"
    )

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    st.write("")

    if st.button(
        "🗑️ Clear Prediction History",
        key="clear_history_button",
        use_container_width=True
    ):

        st.session_state.prediction_history = []

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption("🏦 Loan Approval Prediction System")

st.caption("Powered by XGBoost • Streamlit")

st.caption("🔒 Secure • ⚡ Fast • 🎯 Machine Learning")
