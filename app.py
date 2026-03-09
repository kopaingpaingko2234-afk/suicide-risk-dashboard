import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

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
st.caption("Data Mining Project | 4-Level Random Forest Classification Dashboard")

# -----------------------------
# Load model and data
# -----------------------------
@st.cache_resource
def load_artifacts():
    rf_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    country_encoder = joblib.load("country_encoder.pkl")
    return rf_model, scaler, country_encoder

@st.cache_data
def load_data():
    return pd.read_csv("suicide_complete_2000_2016.csv")

try:
    rf_model, scaler, country_encoder = load_artifacts()
    df = load_data()
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.info("Make sure these files are in the same folder as app.py: random_forest_model.pkl, scaler.pkl, country_encoder.pkl, suicide_complete_2000_2016.csv")
    st.stop()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# -----------------------------
# Normalize text columns
# -----------------------------
df["sex"] = df["sex"].astype(str).str.lower().str.strip()
df["age"] = df["age"].astype(str).str.strip()
df["country"] = df["country"].astype(str).str.strip()

# -----------------------------
# Mappings
# -----------------------------
age_map = {
    "5-14 years": 1,
    "15-24 years": 2,
    "25-34 years": 3,
    "35-54 years": 4,
    "55-74 years": 5,
    "75+ years": 6,
}

risk_labels = {
    0: "🟢 Low Suicide Risk",
    1: "🟡 Medium Suicide Risk",
    2: "🟠 High Suicide Risk",
    3: "🔴 Very High Suicide Risk",
}

risk_plain_labels = {
    0: "Low Suicide Risk",
    1: "Medium Suicide Risk",
    2: "High Suicide Risk",
    3: "Very High Suicide Risk",
}

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Input Parameters")
st.sidebar.markdown("### Prediction Inputs")
st.sidebar.caption("Select demographic and country data to estimate suicide risk.")

countries = sorted(df["country"].dropna().unique().tolist())
years = sorted(df["year"].dropna().astype(int).unique().tolist())

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
    & (df["year"].astype(int) == int(year))
    & (df["sex"] == sex_db)
    & (df["age"] == age)
]

population_options = sorted(
    filtered_df["population"].dropna().astype(int).unique().tolist()
)

if len(population_options) == 0:
    st.sidebar.warning("No population data found for this selection. Enter manually.")
    population = st.sidebar.number_input("Population", min_value=1, value=100000)
else:
    population = st.sidebar.selectbox("Population", population_options)

suicides_options = sorted(
    filtered_df["suicides_no"].dropna().astype(int).unique().tolist()
)

if len(suicides_options) == 0:
    st.sidebar.warning("No suicides number found for this selection. Enter manually.")
    suicides_no = st.sidebar.number_input("Number of Suicides", min_value=0, value=0)
else:
    suicides_no = st.sidebar.selectbox("Number of Suicides", suicides_options)

predict_clicked = st.sidebar.button("Predict Risk")

# -----------------------------
# Prediction dashboard
# -----------------------------
st.header("🎯 Prediction Dashboard")

if predict_clicked:
    try:
        sex_encoded = 1 if sex == "Male" else 0
        age_encoded = age_map[age]
        country_encoded = country_encoder.transform([country])[0]

        # Must match training feature order exactly:
        # ['year', 'suicides_no', 'population', 'sex_encoded', 'age_encoded', 'country_encoded']
        X_input = pd.DataFrame([{
            "year": int(year),
            "suicides_no": int(suicides_no),
            "population": int(population),
            "sex_encoded": int(sex_encoded),
            "age_encoded": int(age_encoded),
            "country_encoded": int(country_encoded)
        }])

        X_scaled = scaler.transform(X_input)

        pred_class = int(rf_model.predict(X_scaled)[0])
        probs = rf_model.predict_proba(X_scaled)[0]

        label = risk_labels[pred_class]
        pred_prob = float(probs[pred_class])

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Risk Level", label)
        col2.metric("Prediction Confidence", f"{pred_prob:.2%}")
        col3.metric("Model Used", "Random Forest (4 Classes)")

        prob_df = pd.DataFrame({
            "Risk Level": [
                "Low Suicide Risk",
                "Medium Suicide Risk",
                "High Suicide Risk",
                "Very High Suicide Risk"
            ],
            "Probability": probs
        })

        st.subheader("Prediction Probabilities")
        st.dataframe(
            prob_df.style.format({"Probability": "{:.2%}"}),
            use_container_width=True
        )

        prob_bar = px.bar(
            prob_df,
            x="Risk Level",
            y="Probability",
            title="Probability for Each Risk Level",
            text="Probability"
        )
        prob_bar.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        prob_bar.update_yaxes(tickformat=".0%")
        st.plotly_chart(prob_bar, use_container_width=True)

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pred_prob * 100,
                title={"text": f"Confidence in {risk_plain_labels[pred_class]}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 25], "color": "#2ecc71"},
                        {"range": [25, 50], "color": "#f1c40f"},
                        {"range": [50, 75], "color": "#e67e22"},
                        {"range": [75, 100], "color": "#e74c3c"},
                    ],
                },
            )
        )
        st.plotly_chart(gauge, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")

# -----------------------------
# Country trend chart
# -----------------------------
st.divider()
st.header("📈 Country Suicide Rate Trend")

country_data = df[df["country"] == country].copy()
yearly = country_data.groupby("year", as_index=False)["suicide_rate"].mean()

trend_chart = px.line(
    yearly,
    x="year",
    y="suicide_rate",
    markers=True,
    title=f"Average Suicide Rate Trend in {country}",
    labels={"year": "Year", "suicide_rate": "Average Suicide Rate"}
)
st.plotly_chart(trend_chart, use_container_width=True)

# -----------------------------
# Risk distribution for selected country
# -----------------------------
st.divider()
st.header("📌 Risk Distribution Snapshot")

country_year_data = df[df["country"] == country].copy()

if "suicide_rate" in country_year_data.columns and not country_year_data.empty:
    q1 = df["suicide_rate"].quantile(0.25)
    q2 = df["suicide_rate"].quantile(0.50)
    q3 = df["suicide_rate"].quantile(0.75)

    def risk_level(rate):
        if rate <= q1:
            return "Low Suicide Risk"
        elif rate <= q2:
            return "Medium Suicide Risk"
        elif rate <= q3:
            return "High Suicide Risk"
        else:
            return "Very High Suicide Risk"

    country_year_data["risk_level"] = country_year_data["suicide_rate"].apply(risk_level)

    risk_counts = country_year_data["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]

    risk_pie = px.pie(
        risk_counts,
        names="Risk Level",
        values="Count",
        title=f"Historical Risk Distribution in {country}"
    )
    st.plotly_chart(risk_pie, use_container_width=True)

# -----------------------------
# Model comparison section
# -----------------------------
st.divider()
st.header("📊 Model Performance Comparison")

models = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "SVM"],
    "Accuracy": [0.815145, 0.926503, 0.810690],
    "Precision": [0.818877, 0.926920, 0.822191],
    "Recall": [0.815145, 0.926503, 0.810690],
    "F1 Score": [0.815477, 0.926449, 0.813763],
    "AUC": [0.967296, 0.989835, 0.959837]
})

st.subheader("Model Evaluation Metrics")
st.dataframe(
    models.style.format({
        "Accuracy": "{:.6f}",
        "Precision": "{:.6f}",
        "Recall": "{:.6f}",
        "F1 Score": "{:.6f}",
        "AUC": "{:.6f}"
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
