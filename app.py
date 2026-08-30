import streamlit as st
import pandas as pd
import joblib


model = joblib.load("breast_cancer_pipeline.pkl")


st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Classification")

st.write(
    "Enter the tumor measurements below to predict "
    "whether the tumor is benign or malignant."
)
#hey

st.warning(
    "⚠️ This application is for educational purposes only "
    "and is not a medical diagnostic tool."
)


st.header("📊 Mean Features")

col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input(
        "Radius Mean",
        min_value=7.691,
        max_value=28.110,
        value=14.0
    )

    perimeter_mean = st.number_input(
        "Perimeter Mean",
        min_value=47.920,
        max_value=188.500,
        value=90.0
    )

    smoothness_mean = st.number_input(
        "Smoothness Mean",
        min_value=0.052630,
        max_value=0.163400,
        value=0.096
    )

    concavity_mean = st.number_input(
        "Concavity Mean",
        min_value=0.0,
        max_value=0.426800,
        value=0.08
    )

    symmetry_mean = st.number_input(
        "Symmetry Mean",
        min_value=0.116700,
        max_value=0.304000,
        value=0.18
    )


with col2:
    texture_mean = st.number_input(
        "Texture Mean",
        min_value=9.710,
        max_value=39.280,
        value=19.0
    )

    area_mean = st.number_input(
        "Area Mean",
        min_value=170.4,
        max_value=2501.0,
        value=650.0
    )

    compactness_mean = st.number_input(
        "Compactness Mean",
        min_value=0.019380,
        max_value=0.314000,
        value=0.10
    )

    concave_points_mean = st.number_input(
        "Concave Points Mean",
        min_value=0.0,
        max_value=0.202100,
        value=0.05
    )

    fractal_dimension_mean = st.number_input(
        "Fractal Dimension Mean",
        min_value=0.049960,
        max_value=0.097440,
        value=0.06
    )

st.header("📐 Standard Error Features")

col1, col2 = st.columns(2)

with col1:
    radius_se = st.number_input(
        "Radius SE",
        min_value=0.111500,
        max_value=2.873000,
        value=0.40
    )

    perimeter_se = st.number_input(
        "Perimeter SE",
        min_value=0.757000,
        max_value=21.980000,
        value=2.5
    )

    smoothness_se = st.number_input(
        "Smoothness SE",
        min_value=0.001713,
        max_value=0.031130,
        value=0.007
    )

    concavity_se = st.number_input(
        "Concavity SE",
        min_value=0.0,
        max_value=0.396000,
        value=0.02
    )

    symmetry_se = st.number_input(
        "Symmetry SE",
        min_value=0.007882,
        max_value=0.061460,
        value=0.02
    )


with col2:
    texture_se = st.number_input(
        "Texture SE",
        min_value=0.360200,
        max_value=4.885000,
        value=1.2
    )

    area_se = st.number_input(
        "Area SE",
        min_value=6.802000,
        max_value=542.200000,
        value=40.0
    )

    compactness_se = st.number_input(
        "Compactness SE",
        min_value=0.002252,
        max_value=0.135400,
        value=0.03
    )

    concave_points_se = st.number_input(
        "Concave Points SE",
        min_value=0.0,
        max_value=0.052790,
        value=0.01
    )

    fractal_dimension_se = st.number_input(
        "Fractal Dimension SE",
        min_value=0.000895,
        max_value=0.029840,
        value=0.003
    )


st.header("🔬 Worst Features")

col1, col2 = st.columns(2)

with col1:
    radius_worst = st.number_input(
        "Radius Worst",
        min_value=8.678000,
        max_value=36.040000,
        value=16.0
    )

    perimeter_worst = st.number_input(
        "Perimeter Worst",
        min_value=54.490000,
        max_value=251.200000,
        value=105.0
    )

    smoothness_worst = st.number_input(
        "Smoothness Worst",
        min_value=0.071170,
        max_value=0.218400,
        value=0.13
    )

    concavity_worst = st.number_input(
        "Concavity Worst",
        min_value=0.0,
        max_value=1.252000,
        value=0.20
    )

    symmetry_worst = st.number_input(
        "Symmetry Worst",
        min_value=0.156500,
        max_value=0.663800,
        value=0.29
    )


with col2:
    texture_worst = st.number_input(
        "Texture Worst",
        min_value=12.020000,
        max_value=49.540000,
        value=25.0
    )

    area_worst = st.number_input(
        "Area Worst",
        min_value=223.600000,
        max_value=4254.000000,
        value=850.0
    )

    compactness_worst = st.number_input(
        "Compactness Worst",
        min_value=0.027290,
        max_value=0.937900,
        value=0.25
    )

    concave_points_worst = st.number_input(
        "Concave Points Worst",
        min_value=0.0,
        max_value=0.291000,
        value=0.10
    )

    fractal_dimension_worst = st.number_input(
        "Fractal Dimension Worst",
        min_value=0.055040,
        max_value=0.173000,
        value=0.08
    )

input_data = pd.DataFrame([{
    "radius_mean": radius_mean,
    "texture_mean": texture_mean,
    "perimeter_mean": perimeter_mean,
    "area_mean": area_mean,
    "smoothness_mean": smoothness_mean,
    "compactness_mean": compactness_mean,
    "concavity_mean": concavity_mean,
    "concave points_mean": concave_points_mean,
    "symmetry_mean": symmetry_mean,
    "fractal_dimension_mean": fractal_dimension_mean,

    "radius_se": radius_se,
    "texture_se": texture_se,
    "perimeter_se": perimeter_se,
    "area_se": area_se,
    "smoothness_se": smoothness_se,
    "compactness_se": compactness_se,
    "concavity_se": concavity_se,
    "concave points_se": concave_points_se,
    "symmetry_se": symmetry_se,
    "fractal_dimension_se": fractal_dimension_se,

    "radius_worst": radius_worst,
    "texture_worst": texture_worst,
    "perimeter_worst": perimeter_worst,
    "area_worst": area_worst,
    "smoothness_worst": smoothness_worst,
    "compactness_worst": compactness_worst,
    "concavity_worst": concavity_worst,
    "concave points_worst": concave_points_worst,
    "symmetry_worst": symmetry_worst,
    "fractal_dimension_worst": fractal_dimension_worst
}])

st.divider()

if st.button("🔍 Predict", use_container_width=True):

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    benign_probability = probabilities[0]
    malignant_probability = probabilities[1]

    st.subheader("Prediction")

    if prediction == "M":
        st.error("🔴 Malignant")
    else:
        st.success("🟢 Benign")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Benign Probability",
            f"{benign_probability:.2%}"
        )

    with col2:
        st.metric(
            "Malignant Probability",
            f"{malignant_probability:.2%}"
        )