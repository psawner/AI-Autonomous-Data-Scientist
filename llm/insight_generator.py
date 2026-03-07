from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    "groq:meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.3
)


def generate_insights(feature_importance, model_results):

    top_features = feature_importance.head(5).to_string()
    results = str(model_results)

    prompt = f"""
You are an expert senior data scientist creating insights for a business report.

Analyze the machine learning results and feature importance.

MODEL PERFORMANCE
{results}

TOP IMPORTANT FEATURES
{top_features}

Write clear business insights using this structure:

KEY DRIVERS
Explain which features most strongly influence predictions and why.

RISK FACTORS
Identify variables that may negatively impact outcomes.

BUSINESS INTERPRETATION
Explain what the results mean in simple business language.

RECOMMENDATIONS
Give 2 practical actions a company should take based on the model.

IMPORTANT:
- Write professionally
- Avoid technical jargon
- Make insights concise
- Format with headings exactly as shown above
"""

    response = model.invoke(prompt)

    return response.content