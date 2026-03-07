import streamlit as st
from modules.data_loader import load_data
from modules.data_analyzer import analyze_data
from modules.data_cleaner import clean_data
from modules.eda import (
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_feature_distribution
)
from modules.model_trainer import train_models
import seaborn as sns
import matplotlib.pyplot as plt

from modules.feature_importance import get_feature_importance
from llm.insight_generator import generate_insights
from llm.dataframe_agent import create_dataframe_agent
from modules.predictor import make_prediction
from modules.report_generator import generate_report


@st.cache_resource
def load_agent(df):
    return create_dataframe_agent(df)


st.title("AI Autonomous Data Scientist")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Dataset",
    "EDA",
    "Model Training",
    "AI Insights",
    "Dataset Chat",
    "Prediction",
    "Report"
])

st.sidebar.title("Workflow")

st.sidebar.write("""
1 Upload Dataset  
2 Run Data Cleaning  
3 Train Models  
4 Generate Insights  
5 Chat with Dataset  
6 Make Predictions  
7 Download Report
""")

with tab1:

    uploaded_file = st.file_uploader("Upload Dataset", type=["csv","xlsx"])

    if uploaded_file:

        df = load_data(uploaded_file)
        st.session_state["df"] = df

        st.subheader("Dataset Preview")
        st.write(df.head())

        summary = analyze_data(df)

        st.subheader("Dataset Summary")

        st.write("Rows:", summary["rows"])
        st.write("Columns:", summary["columns"])

        st.write("Numerical Columns:", summary["numerical_columns"])
        st.write("Categorical Columns:", summary["categorical_columns"])

        st.write("Missing Values:", summary["missing_values"])

        target_column = st.selectbox("Select Target Column", df.columns)

        if st.button("Prepare Data"):

            X, y = clean_data(df, target_column)

            st.session_state["X"] = X
            st.session_state["y"] = y

            st.success("Data prepared successfully!")


with tab2:

    if "X" in st.session_state and "y" in st.session_state:

        st.subheader("Target Distribution")
        st.pyplot(plot_target_distribution(st.session_state["y"]))

        st.subheader("Correlation Heatmap")
        st.pyplot(plot_correlation_heatmap(st.session_state["X"]))

        for fig in plot_feature_distribution(st.session_state["X"]):
            st.pyplot(fig)



with tab3:

    if "X" in st.session_state and "y" in st.session_state:

        if st.button("Train Models"):

            problem_type, results, best_model_name, best_model, conf_matrix = train_models(
                st.session_state["X"],
                st.session_state["y"]
            )

            st.session_state["results"] = results
            st.session_state["best_model"] = best_model
            st.session_state["best_model_name"] = best_model_name
            st.session_state["conf_matrix"] = conf_matrix
            

        if "results" in st.session_state:

            st.subheader("Model Performance")

            for model, score in st.session_state["results"].items():
                st.write(f"{model}: {score}")

            st.subheader("Best Model")
            st.write(st.session_state["best_model_name"])



with tab4:

    if "results" in st.session_state:

        if st.button("Generate Insights"):

            feature_importance = get_feature_importance(
                st.session_state["best_model"],
                st.session_state["X"]
            )

            insights = generate_insights(feature_importance, st.session_state["results"])

            st.session_state["insights"] = insights

        if "insights" in st.session_state:
            st.write(st.session_state["insights"])



with tab5:

    if "df" in st.session_state:

        question = st.text_input("Ask a question about your dataset")

        if question:

            agent = create_dataframe_agent(st.session_state["df"])

            response = agent.run(question)

            st.write(response)


with tab6:

    if "X" in st.session_state:

        input_data = {}

        for col in st.session_state["X"].columns:
            input_data[col] = st.number_input(col, value=0.0)

        if st.button("Predict"):

            prediction = make_prediction(input_data)

            st.success(f"Prediction: {prediction}")


with tab7:

    if "results" in st.session_state and "insights" in st.session_state:

        plot_target_distribution(st.session_state["y"], save=True)

        plot_correlation_heatmap(st.session_state["X"], save=True)

        plot_feature_distribution(st.session_state["X"], save=True)

        if st.button("Generate Report"):
            
            report_path = generate_report(
                st.session_state["insights"],
                st.session_state["results"]
            )

            with open(report_path, "rb") as f:

                st.download_button(
                    "Download Report",
                    data=f,
                    file_name="ai_data_scientist_report.pdf",
                    mime="application/pdf"
                )

