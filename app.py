import streamlit as st
import pandas as pd

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("PlayTennis.csv")

features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
target = 'Play Tennis'

# -----------------------------
# Compute Priors
# -----------------------------
priors = df[target].value_counts(normalize=True).to_dict()

# -----------------------------
# Conditional Probabilities
# -----------------------------
conditional_prob = {}

for f in features:
    conditional_prob[f] = (
        pd.crosstab(df[f], df[target], normalize='columns')
        .stack()
        .to_dict()
    )

# -----------------------------
# Naive Bayes Prediction
# -----------------------------
def predict(sample):
    classes = df[target].unique()
    posteriors = {}

    for c in classes:
        posterior = priors[c]

        for f in features:
            key = (sample[f], c)
            posterior *= conditional_prob[f].get(key, 1e-6)

        posteriors[c] = posterior

    return max(posteriors, key=posteriors.get), posteriors


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Naive Bayes Classifier – Play Tennis Prediction")

st.write("### Select Weather Conditions")

col1, col2 = st.columns(2)

with col1:
    outlook = st.selectbox("Outlook", ["Sunny", "Overcast", "Rain"])
    temperature = st.selectbox("Temperature", ["Hot", "Mild", "Cool"])

with col2:
    humidity = st.selectbox("Humidity", ["High", "Normal"])
    wind = st.selectbox("Wind", ["Weak", "Strong"])

# Make sample
sample = {
    "Outlook": outlook,
    "Temperature": temperature,
    "Humidity": humidity,
    "Wind": wind
}

if st.button("Predict"):
    prediction, probs = predict(sample)

    st.write("## 🎯 Prediction:", prediction)

    st.write("### Posterior Probabilities")
    st.json(probs)

    st.success(f"Model Prediction: **{prediction}**")


st.write("---")
st.write("Dataset Preview:")
st.dataframe(df)
