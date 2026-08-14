import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Loan Approval Prediction - Moulitha",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# SESSION STATE

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# CUSTOM CSS

st.markdown(
    """
    <style>

    /* MAIN BACKGROUND */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 20%,
                rgba(0, 180, 255, 0.10),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 80%,
                rgba(40, 90, 255, 0.10),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #021326 0%,
                #041d38 45%,
                #03101f 100%
            );

        color: white;
    }


    /* HIDE STREAMLIT MENU */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* MAIN CONTAINER */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }


    /* STREAMLIT HEADINGS */

    h1 {
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        color: white !important;
    }

    h2 {
        color: white !important;
        font-weight: 750 !important;
    }

    h3 {
        color: white !important;
        font-weight: 700 !important;
    }


    /* NORMAL TEXT */

    p {
        color: #dcecff;
    }


    /* INPUT LABELS */

    label {
        color: white !important;
        font-weight: 650 !important;
    }


    /* INPUT BOXES */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] > div {
        background-color: rgba(245, 247, 252, 0.96) !important;
        border-radius: 14px !important;
    }


    div[data-baseweb="select"] *,
    div[data-baseweb="input"] input,
    div[data-testid="stNumberInput"] input {
        color: #000000 !important;
        font-weight: 600 !important;
    }


    /* MODEL SUCCESS BOX */

    .model-success {
        padding: 18px 22px;
        border-radius: 15px;
        background: rgba(0, 180, 170, 0.13);
        border: 1px solid rgba(0, 220, 200, 0.18);
        color: white;
        font-size: 20px;
        font-weight: 700;
        margin: 20px 0 30px 0;
    }


    /* PREDICTION CARDS */

    .approved-card {
        padding: 30px;
        border-radius: 20px;
        background: rgba(0, 180, 120, 0.14);
        border: 1px solid rgba(0, 220, 160, 0.30);
        text-align: center;
        margin-top: 25px;
    }


    .rejected-card {
        padding: 30px;
        border-radius: 20px;
        background: rgba(220, 60, 70, 0.14);
        border: 1px solid rgba(255, 80, 90, 0.30);
        text-align: center;
        margin-top: 25px;
    }


    .prediction-title {
        font-size: 30px;
        font-weight: 800;
        color: white;
    }


    .prediction-text {
        font-size: 18px;
        color: #dcecff;
        margin-top: 10px;
    }


    /* INFO SECTION */

    .info-card {
        padding: 25px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(120, 190, 255, 0.12);
        margin-top: 20px;
        line-height: 1.8;
    }


    /* FLOW BOX */

    .flow-box {
        padding: 20px;
        margin-top: 15px;
        border-radius: 16px;
        background: rgba(0, 140, 255, 0.08);
        border: 1px solid rgba(0, 170, 255, 0.16);
        text-align: center;
        color: white;
        font-size: 18px;
        font-weight: 650;
        line-height: 2;
    }


    /* PREDICTION SUMMARY */

    .summary-card {
        padding: 22px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(120, 190, 255, 0.14);
        margin-top: 20px;
    }


    .summary-title {
        font-size: 22px;
        font-weight: 800;
        color: white;
        margin-bottom: 15px;
    }


    .summary-item {
        color: #dcecff;
        font-size: 16px;
        line-height: 1.9;
    }


    /* HISTORY CARD */

    .history-card {
        padding: 20px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(120, 190, 255, 0.14);
        margin-top: 15px;
    }


    /* PREDICT LOAN APPROVAL BUTTON */

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        padding: 14px;

        font-size: 18px;
        font-weight: 800 !important;

        color: #ffffff !important;
        background-color: #071a2f !important;

        border: 1px solid #16456d !important;

        box-shadow:
            0 6px 18px rgba(0, 0, 0, 0.35);

        transition: all 0.25s ease;
    }


    .stButton > button:hover {
        color: #ffffff !important;

        background-color: #0b2948 !important;

        border: 1px solid #19cfff !important;

        box-shadow:
            0 0 18px rgba(25, 207, 255, 0.25);

        transform: translateY(-1px);
    }


    .stButton > button:active {
        background-color: #041221 !important;
        color: #ffffff !important;
    }


    /* FLOATING PARTICLES */

    .particle {
        position: fixed;
        width: 4px;
        height: 4px;
        background: rgba(100, 210, 255, 0.65);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        animation: floatParticle linear infinite;
    }


    .p1 {
        left: 8%;
        top: 85%;
        animation-duration: 18s;
    }

    .p2 {
        left: 20%;
        top: 70%;
        animation-duration: 22s;
        animation-delay: 2s;
    }

    .p3 {
        left: 35%;
        top: 90%;
        animation-duration: 20s;
        animation-delay: 4s;
    }

    .p4 {
        left: 50%;
        top: 75%;
        animation-duration: 25s;
        animation-delay: 1s;
    }

    .p5 {
        left: 65%;
        top: 88%;
        animation-duration: 19s;
        animation-delay: 3s;
    }

    .p6 {
        left: 78%;
        top: 65%;
        animation-duration: 23s;
        animation-delay: 5s;
    }

    .p7 {
        left: 90%;
        top: 82%;
        animation-duration: 21s;
        animation-delay: 2s;
    }


    @keyframes floatParticle {

        0% {
            transform: translateY(0px) translateX(0px);
            opacity: 0;
        }

        15% {
            opacity: 0.7;
        }

        50% {
            transform: translateY(-350px) translateX(45px);
            opacity: 0.45;
        }

        85% {
            opacity: 0.6;
        }

        100% {
            transform: translateY(-700px) translateX(-35px);
            opacity: 0;
        }
    }


    /* GRID OVERLAY */

    .grid-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;

        background-image:
            linear-gradient(
                rgba(70, 170, 255, 0.035) 1px,
                transparent 1px
            ),
            linear-gradient(
                90deg,
                rgba(70, 170, 255, 0.035) 1px,
                transparent 1px
            );

        background-size: 55px 55px;
    }

    </style>

    <div class="grid-overlay"></div>

    <div class="particle p1"></div>
    <div class="particle p2"></div>
    <div class="particle p3"></div>
    <div class="particle p4"></div>
    <div class="particle p5"></div>
    <div class="particle p6"></div>
    <div class="particle p7"></div>

    """,
    unsafe_allow_html=True
)

# FILE PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "xgboost_model.pkl"
)

PREPROCESSOR_FILE = os.path.join(
    BASE_DIR,
    "preprocessor.pkl"
)

# LOAD MODEL AND PREPROCESSOR

model = None
preprocessor = None
model_loaded = False

try:

    model = joblib.load(MODEL_FILE)

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    model_loaded = True

except Exception:

    model_loaded = False

# TOP HEADING

st.title(
    "🏦 Loan Approval Prediction — Moulitha"
)

st.write(
    "Enter applicant details below to predict whether "
    "the loan is likely to be approved."
)

# MODEL STATUS

if model_loaded:

    st.markdown(
        """
        <div class="model-success">
        ✅ XGBoost Model Loaded Successfully!
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.error(
        "❌ Unable to load the model files. "
        "Please check that xgboost_model.pkl and "
        "preprocessor.pkl are in the same folder as app.py."
    )

# APPLICANT INFORMATION

st.header(
    "👤 Applicant Information"
)


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

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

# FINANCIAL INFORMATION

st.header(
    "💰 Financial Information"
)


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
        step=12.0
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0],
        format_func=lambda x:
        "Positive (1)"
        if x == 1.0
        else "Negative (0)"
    )

# PREDICTION BUTTON

st.markdown("---")


predict_button = st.button(
    "🔮 Predict Loan Approval"
)

# PREDICTION

if predict_button:

    if not model_loaded:

        st.error(
            "Model files could not be loaded. "
            "Please check the model files."
        )

    else:

        input_data = pd.DataFrame(
            {
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
            }
        )


        try:

            
            # PREPROCESS INPUT

            processed_data = preprocessor.transform(
                input_data
            )

            # MODEL PREDICTION

            prediction = model.predict(
                processed_data
            )[0]

            # PREDICTION PROBABILITY

            probability = None

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    processed_data
                )[0]


                if hasattr(
                    model,
                    "classes_"
                ):

                    class_list = list(
                        model.classes_
                    )

                    try:

                        prediction_index = (
                            class_list.index(
                                prediction
                            )
                        )

                    except ValueError:

                        prediction_index = int(
                            prediction
                        )

                else:

                    prediction_index = int(
                        prediction
                    )


                probability = (
                    probabilities[
                        prediction_index
                    ] * 100
                )


            # ------------------------------------------------
            # PREDICTION RESULT
            # ------------------------------------------------

            is_approved = (
                str(prediction) == "1"
                or str(prediction).upper() == "Y"
            )


            if is_approved:

                result_text = "Loan Approved"

                result_icon = "✅"

                result_description = (
                    "Based on the information provided, "
                    "the model predicts that the loan is "
                    "likely to be approved."
                )

            else:

                result_text = "Loan Not Approved"

                result_icon = "❌"

                result_description = (
                    "Based on the information provided, "
                    "the model predicts that the loan is "
                    "unlikely to be approved."
                )

            # SAVE PREDICTION HISTORY

            history_record = {

                "Result": result_text,

                "Probability": (
                    round(
                        probability,
                        2
                    )
                    if probability is not None
                    else None
                ),

                "Education": education,

                "Credit History": (
                    "Positive"
                    if credit_history == 1.0
                    else "Negative"
                ),

                "Property Area": property_area,

                "Loan Amount": loan_amount

            }


            st.session_state.prediction_history.append(
                history_record
            )


            # ------------------------------------------------
            # DISPLAY RESULT
            # ------------------------------------------------

            if is_approved:

                st.markdown(
                    f"""
                    <div class="approved-card">

                    <div class="prediction-title">
                    {result_icon} {result_text}
                    </div>

                    <div class="prediction-text">
                    {result_description}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="rejected-card">

                    <div class="prediction-title">
                    {result_icon} {result_text}
                    </div>

                    <div class="prediction-text">
                    {result_description}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # PROBABILITY GAUGE

            if probability is not None:

                st.subheader(
                    "📊 Prediction Probability"
                )


                gauge = go.Figure(
                    go.Indicator(

                        mode="gauge+number",

                        value=probability,

                        number={
                            "suffix": "%",
                            "font": {
                                "size": 38,
                                "color": "#000000"
                            }
                        },

                        title={
                            "text": "Model Confidence",
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
                                "color": "#00B8D9"
                            },

                            "bgcolor":
                                "rgba(255,255,255,0.80)",

                            "borderwidth": 1,

                            "bordercolor": "#FFFFFF",

                            "steps": [

                                {
                                    "range": [0, 40],
                                    "color": "#FFD6D6"
                                },

                                {
                                    "range": [40, 70],
                                    "color": "#FFF1C7"
                                },

                                {
                                    "range": [70, 100],
                                    "color": "#D7F5E8"
                                }

                            ]

                        }

                    )
                )


                gauge.update_layout(

                    height=300,

                    margin={
                        "l": 30,
                        "r": 30,
                        "t": 70,
                        "b": 20
                    },

                    paper_bgcolor=
                        "rgba(255,255,255,0.75)",

                    plot_bgcolor=
                        "rgba(255,255,255,0.75)",

                    font={
                        "color": "#000000"
                    }

                )


                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

            # APPLICANT SUMMARY

            st.subheader(
                "📋 Applicant Summary"
            )


            st.markdown(
                f"""
                <div class="summary-card">

                <div class="summary-title">
                Applicant Details Used for Prediction
                </div>

                <div class="summary-item">

                <b>Gender:</b> {gender}<br>

                <b>Married:</b> {married}<br>

                <b>Dependents:</b> {dependents}<br>

                <b>Education:</b> {education}<br>

                <b>Self Employed:</b> {self_employed}<br>

                <b>Applicant Income:</b>
                {applicant_income:,.2f}<br>

                <b>Coapplicant Income:</b>
                {coapplicant_income:,.2f}<br>

                <b>Loan Amount:</b>
                {loan_amount:,.2f}<br>

                <b>Loan Amount Term:</b>
                {loan_amount_term:,.0f}<br>

                <b>Credit History:</b>
                {
                    "Positive"
                    if credit_history == 1.0
                    else "Negative"
                }<br>

                <b>Property Area:</b>
                {property_area}

                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        except Exception as e:

            st.error(
                "An error occurred while making the prediction."
            )

            st.code(
                str(e)
            )

# PREDICTION HISTORY

st.markdown("---")


st.header(
    "📋 Prediction History"
)


if len(
    st.session_state.prediction_history
) == 0:

    st.info(
        "No predictions have been made yet. "
        "Enter applicant details and click "
        "'Predict Loan Approval'."
    )

else:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )


    history_df.index = range(
        1,
        len(history_df) + 1
    )


    history_df.index.name = (
        "Prediction #"
    )


    st.dataframe(
        history_df,
        use_container_width=True
    )


    st.write(
        f"Total predictions: "
        f"**{len(st.session_state.prediction_history)}**"
    )


    clear_history = st.button(
        "🗑️ Clear Prediction History"
    )


    if clear_history:

        st.session_state.prediction_history = []

        st.rerun()


# HOW LOAN PREDICTION WORKS

st.markdown("---")


st.header(
    "🔍 How Loan Prediction Works"
)


st.markdown(
    """
    <div class="info-card">

    <b>Loan Approval Prediction</b> is a machine learning
    application that predicts whether a loan application is
    likely to be <b>Approved or Not Approved</b>.

    <br><br>

    I used a dataset containing <b>614 previous loan
    applications</b>. The dataset includes information such as
    <b>Gender, Married status, Dependents, Education,
    Self Employment, Applicant Income, Coapplicant Income,
    Loan Amount, Loan Amount Term, Credit History,
    and Property Area</b>.

    <br><br>

    I removed the <b>Loan ID</b> because it is only an identifier
    and does not help the model make a prediction.

    <br><br>

    I used <b>Loan Status</b> as the target variable. In the
    dataset, <b>Y</b> represents an approved loan and
    <b>N</b> represents a rejected loan.

    <br><br>

    I divided the dataset into <b>80% training data and
    20% testing data</b>. I used the training data to teach the
    <b>XGBoost classifier</b> to identify patterns between the
    applicant's details and previous loan decisions.

    <br><br>

    When I enter a new applicant's details in this Streamlit
    application, the model uses these details as input and
    predicts whether the loan is likely to be
    <b>Approved or Not Approved</b>.

    <br><br>

    <b>Important:</b> The <b>Loan Amount is not predicted by
    my model</b>. It is entered by the applicant as an input.
    The model uses the entered Loan Amount along with the
    applicant's other details to predict the
    <b>Loan Status</b>.

    </div>
    """,
    unsafe_allow_html=True
)

# EXPLANATION

with st.expander(
    "🔍 How does the prediction work?"
):

    st.markdown(
        """
        1. I enter the applicant's details.
        2. The details are passed through the same preprocessing
           steps used during model training.
        3. The processed data is given to the trained XGBoost model.
        4. XGBoost uses patterns learned from previous loan
           applications.
        5. The model predicts **Approved (Y)** or
           **Not Approved (N)**.
        6. The application also displays the model's estimated
           approval probability.

        **Input → Preprocessing → XGBoost → Loan Status**
        """
    )


# PREDICTION FLOW

st.subheader(
    "🔄 Prediction Flow"
)


st.markdown(
    """
    <div class="flow-box">

    Applicant Details
    <br>↓<br>

    Data Preprocessing
    <br>↓<br>

    Trained XGBoost Model
    <br>↓<br>

    Learned Patterns
    <br>↓<br>

    Loan Status Prediction
    <br>↓<br>

    ✅ Approved &nbsp;&nbsp; / &nbsp;&nbsp; ❌ Not Approved

    </div>
    """,
    unsafe_allow_html=True
)

# IMPORTANT NOTE

st.info(
    "💡 Note: Loan Amount is an input given by the applicant. "
    "The XGBoost model uses Loan Amount along with the other "
    "applicant details to predict Loan Status. "
    "The model does not predict the Loan Amount."
)

# FOOTER

st.markdown("---")


st.caption(
    "🏦 Loan Approval Prediction | "
    "Machine Learning with XGBoost | "
    "Streamlit | "
    "Developed by Moulitha | "
    "© 2026 Moulitha. All Rights Reserved."
)