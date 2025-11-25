import streamlit as st
import pandas as pd

# --------------------------------------------------------
# Modern Page Config
# --------------------------------------------------------
st.set_page_config(
    page_title="Naive Bayes – Play Tennis",
    page_icon="🎾",
    layout="wide"
)

# --------------------------------------------------------
# Custom CSS for Styling
# --------------------------------------------------------
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #e2ebf0, #ffffff);
        }
        .title-style {
            text-align: center;
            color: #2c3e50;
            font-size: 38px;
            font-weight: 900;
            padding-bottom: 10px;
        }
        .subtitle-style {
            text-align: center;
            color: #34495e;
            font-size: 18px;
            margin-top: -10px;
        }
        .card {
            padding: 25px;
            border-radius: 18px;
            background-color: white;
            box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
        }
        .predict-box {
            padding: 20px;
            border-radius: 12px;
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            color: white;
            background: #27ae60;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
        }
        .footer {
            text-align:center;
            font-size:14px;
            color:#888;
            margin-top:40px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------
# Load Dataset
# --------------------------------------------------------
df = pd.read_csv("PlayTennis.csv")

features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
target = 'Play Tennis'

priors = df[target].value_counts(normalize=True).to_dict()

conditional_prob = {}
for f in features:
    conditional_prob[f] = (
        pd.crosstab(df[f], df[target], normalize='columns')
        .stack()
        .to_dict()
    )

# --------------------------------------------------------
# Naive Bayes Function
# --------------------------------------------------------
def predict(sample):
    classes = df[target].unique()
    posteriors = {}
    for c in classes:
        posterior = priors[c]
        for f in features:
            posterior *= conditional_prob[f].get((sample[f], c), 1e-6)
        posteriors[c] = posterior
    return max(posteriors, key=posteriors.get), posteriors


# --------------------------------------------------------
# UI Title Section
# --------------------------------------------------------
st.markdown("<h1 class='title-style'>🎯 Naive Bayes Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-style'>Play Tennis Prediction Based on Weather Conditions</p>", unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------
# Input Card
# --------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 🌦️ Select Weather Conditions")

col1, col2 = st.columns(2)

with col1:
    outlook = st.selectbox("☀️ Outlook", ["Sunny", "Overcast", "Rain"])
    temperature = st.selectbox("🌡️ Temperature", ["Hot", "Mild", "Cool"])

with col2:
    humidity = st.selectbox("💧 Humidity", ["High", "Normal"])
    wind = st.selectbox("🌬️ Wind", ["Weak", "Strong"])

st.markdown("</div>", unsafe_allow_html=True)

sample = {
    "Outlook": outlook,
    "Temperature": temperature,
    "Humidity": humidity,
    "Wind": wind
}

st.write("")
st.write("")

# --------------------------------------------------------
# Prediction Button
# --------------------------------------------------------
center = st.columns([3, 1, 3])[1]
with center:
    predict_btn = st.button("🔍 Predict", use_container_width=True)


# --------------------------------------------------------
# Output Section
# --------------------------------------------------------
if predict_btn:
    prediction, probs = predict(sample)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='predict-box'>Prediction: {prediction}</div>", unsafe_allow_html=True)
    st.write("### 📊 Posterior Probabilities")
    st.json(probs)
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# Dataset Preview Section
# --------------------------------------------------------
st.write("")
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.write("### 📄 Dataset Preview")
st.dataframe(df, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# Footer
# --------------------------------------------------------
st.markdown("<p class='footer'>Developed by Kamran Khan Orakzai | Streamlit Machine Learning App</p>", unsafe_allow_html=True)
