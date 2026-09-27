import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(
    page_title="Anemia Prediction App", page_icon="🩸", layout="centered"
)


# Train a dummy model on start so the app runs standalone
@st.cache_resource
def train_model():
    # Synthetic dataset matching typical CBC ranges
    # Features: [Gender (1: Male, 0: Female), Hemoglobin, MCH, MCHC, MCV]
    np.random.seed(42)
    n_samples = 200

    gender = np.random.choice([0, 1], size=n_samples)
    hemoglobin = np.random.uniform(6.0, 18.0, size=n_samples)
    mch = np.random.uniform(15.0, 35.0, size=n_samples)
    mchc = np.random.uniform(28.0, 38.0, size=n_samples)
    mcv = np.random.uniform(60.0, 105.0, size=n_samples)

    # Anemia label rule: Hb < 12.0 for females, Hb < 13.0 for males
    target = []
    for g, hb in zip(gender, hemoglobin):
        if (g == 0 and hb < 12.0) or (g == 1 and hb < 13.0):
            target.append(1)  # Anemic
        else:
            target.append(0)  # Non-Anemic

    X = pd.DataFrame(
        {
            "Gender": gender,
            "Hemoglobin": hemoglobin,
            "MCH": mch,
            "MCHC": mchc,
            "MCV": mcv,
        }
    )
    y = np.array(target)

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    return clf


# Main App Function
def main():
    st.title("🩸 Anemia Risk Prediction System")
    st.write(
        "Enter the patient's complete blood count (CBC) values below to analyze anemia risk."
    )
    st.divider()

    # Load Model
    model = train_model()

    # User Input Form
    st.subheader("📋 Patient Information & Lab Results")

    col1, col2 = st.columns(2)

    with col1:
        gender_option = st.selectbox("Gender", ["Female", "Male"])
        gender = 1 if gender_option == "Male" else 0

        hemoglobin = st.number_input(
            "Hemoglobin (g/dL)",
            min_value=3.0,
            max_value=20.0,
            value=12.5,
            step=0.1,
        )

        mch = st.number_input(
            "MCH (pg)", min_value=10.0, max_value=50.0, value=27.0, step=0.1
        )

    with col2:
        mchc = st.number_input(
            "MCHC (g/dL)", min_value=20.0, max_value=45.0, value=33.0, step=0.1
        )

        mcv = st.number_input(
            "MCV (fL)", min_value=50.0, max_value=120.0, value=85.0, step=0.1
        )

    st.divider()

    # Prediction Action
    if st.button("Predict Anemia Risk", type="primary"):
        input_data = pd.DataFrame(
            [[gender, hemoglobin, mch, mchc, mcv]],
            columns=["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"],
        )

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        st.subheader("🔍 Prediction Results")

        if prediction == 1:
            st.error(
                f"**High Risk of Anemia Detected** (Confidence: {probability[1] * 100:.1f}%)"
            )
            st.warning(
                "Please consult a healthcare professional for clinical diagnostics and treatment."
            )
        else:
            st.success(
                f"**Normal - Low Risk of Anemia** (Confidence: {probability[0] * 100:.1f}%)"
            )
            st.info("Blood parameters appear within standard ranges.")


if __name__ == "__main__":
    main()
