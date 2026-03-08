import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Suicide Risk Dashboard",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🌍 Global Suicide Risk Prediction System")
st.caption("Data Mining Project | Machine Learning Dashboard")

# -----------------------------
# Load model and data
# -----------------------------
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("suicide_complete_2000_2016.csv")

# Encode country
le_country = LabelEncoder()
df["country_encoded"] = le_country.fit_transform(df["country"])

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Input Parameters")
st.sidebar.markdown("### Prediction Inputs")
st.sidebar.caption("Select demographic and country data to estimate suicide risk.")

countries = sorted(df["country"].unique())
years = sorted(df["year"].unique())

country = st.sidebar.selectbox("Country", countries)
year = st.sidebar.selectbox("Year", years)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.selectbox(
    "Age Group",
    [
        "5-14 years",
        "15-24 years",
        "25-34 years",
        "35-54 years",
        "55-74 years",
        "75+ years",
    ],
)

sex_db = sex.lower()

# -----------------------------
# Filter dataset for dropdowns
# -----------------------------
filtered_df = df[
    (df["country"] == country)
    & (df["year"] == year)
    & (df["sex"] == sex_db)
    & (df["age"] == age)
]

population_options = sorted(
    filtered_df["population"].dropna().astype(int).unique().tolist()
)

if len(population_options) == 0:
    st.sidebar.warning("No population data found for this selection.")
    population = None
else:
    population = st.sidebar.selectbox("Population", population_options)

suicides_options = sorted(
    filtered_df["suicides_no"].dropna().astype(int).unique().tolist()
)

if len(suicides_options) == 0:
    suicides_no = st.sidebar.number_input("Number of Suicides", min_value=0)
else:
    suicides_no = st.sidebar.selectbox("Number of Suicides", suicides_options)

predict_clicked = st.sidebar.button("Predict Risk")

# -----------------------------
# Prediction dashboard
# -----------------------------
st.header("🎯 Prediction Dashboard")

if predict_clicked:
    if population is None:
        st.error("Invalid input combination. Please choose another selection.")
    else:
        sex_encoded = 1 if sex == "Male" else 0

        age_map = {
            "5-14 years": 1,
            "15-24 years": 2,
            "25-34 years": 3,
            "35-54 years": 4,
            "55-74 years": 5,
            "75+ years": 6,
        }

        age_encoded = age_map[age]
        country_encoded = le_country.transform([country])[0]

        X = [[year, suicides_no, population, sex_encoded, age_encoded, country_encoded]]
        X_scaled = scaler.transform(X)

        prob = rf_model.predict_proba(X_scaled)[0][1]
        label = "🔴 High Suicide Risk" if prob > 0.5 else "🟢 Low Suicide Risk"

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Level", label)
        col2.metric("High Risk Probability", f"{prob:.2%}")
        col3.metric("Model Used", "Random Forest")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Suicide Risk Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 40], "color": "green"},
                        {"range": [40, 70], "color": "yellow"},
                        {"range": [70, 100], "color": "red"},
                    ],
                },
            )
        )

        st.plotly_chart(gauge, use_container_width=True)

# -----------------------------
# Country trend chart
# -----------------------------
st.divider()
st.header("📈 Country Suicide Rate Trend")

country_data = df[df["country"] == country]
yearly = country_data.groupby("year")["suicide_rate"].mean().reset_index()

trend_chart = px.line(
    yearly,
    x="year",
    y="suicide_rate",
    markers=True,
    title=f"Suicide Rate Trend in {country}",
)

st.plotly_chart(trend_chart, use_container_width=True)

# -----------------------------
# Model comparison section
# -----------------------------
st.divider()
st.header("📊 Model Performance Comparison")

models = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "SVM"],
    "Accuracy": [0.948775, 0.977728, 0.933185],
    "Precision": [0.969466, 0.978873, 0.926471],
    "Recall": [0.869863, 0.952055, 0.863014],
    "F1 Score": [0.916968, 0.965278, 0.893617],
    "AUC": [0.928331, 0.971077, 0.915005]
})

st.subheader("Model Evaluation Metrics")

st.dataframe(
    models.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1 Score": "{:.3f}",
        "AUC": "{:.3f}"
    }),
    use_container_width=True
)

metrics_long = models.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

col1, col2 = st.columns(2)

with col1:
    bar = px.bar(
        metrics_long,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Bar Chart: Model Performance Comparison"
    )
    st.plotly_chart(bar, use_container_width=True)

with col2:
    selected_metric = st.selectbox(
        "Select metric for pie chart",
        ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    )

    pie = px.pie(
        models,
        names="Model",
        values=selected_metric,
        title=f"Pie Chart: {selected_metric} Comparison"
    )
    st.plotly_chart(pie, use_container_width=True) 
