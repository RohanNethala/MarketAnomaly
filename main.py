import pandas as pd
import os
from openai import OpenAI
import streamlit as st

data = pd.read_csv("Rohan FinancialMarketData.csv")
from sklearn.ensemble import IsolationForest

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY"))


def suggest_strategy(
    anomaly_score,
    investment_strategy,
):
    is_likely_to_be_an_anomaly = ""
    if anomaly_score < 40:
        is_likely_to_be_an_anomaly = "not"
    elif anomaly_score >= 40 and anomaly_score < 70:
        is_likely_to_be_an_anomaly = "somewhat"
    else:
        is_likely_to_be_an_anomaly = "very"

    prompt = f'''You are an investor looking for advice on investment strategies based on the current state of the stock market. You are worried about whether there could be an economic downturn in the near future.

    You decided to train your machine learning model on the historical data of the stock market. You did this with the aim of trying to see how likely the stock market is to enter an anomaly in the near future. Then, based on the results from this model, you want to come up with a suitable investment strategy to minimize your losses.

    Your machine learning model has predicted that the stock market is 
    {is_likely_to_be_an_anomaly} likely to enter an anomaly in the near future.

based on whatever scenario this is, here is the strategy to pick:

    if the stock market is likely to enter an anomaly in the near future, then you want to suggest the following investment strategy of investing conservatively, mostly investing in bonds.

    if the stock market is not likely to enter an anomaly in the near future, then you want to suggest the following investment strategy of mostly investing in high-risk assets, like equities and stocks.

    if the stock market is somewhat likely to enter an anomaly in the near future, then you want to suggest the following investment strategy of investing in equities and bonds with diversification and having some cash.
    '''
    print("EXPLANATION PROMPT", prompt)

    raw_response = client.chat.completions.create(
        model='llama-3.2-3b-preview',
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )
    return raw_response.choices[0].message.content


st.title("Market Anomaly Predictor")

model_recent_data = IsolationForest(n_estimators=100,
                                    contamination=0.05,
                                    random_state=42)
model_recent_data.fit(data[[
    'XAU BGNL', 'BDIY', 'CRY', 'DXY', 'JPY', 'GBP', 'Cl1', 'VIX', 'USGG30YR',
    'GT10', 'USGG2YR', 'USGG3M', 'US0001M', 'GTDEM30Y', 'GTDEM10Y', 'GTDEM2Y',
    'EONIA', 'LUMSTRUU', 'LUACTRUU', 'LG30TRUU', 'LP01TREU', 'EMUSTRUU', 'MXUS'
]])

data['anomaly_score_recent'] = model_recent_data.decision_function(data[[
    'XAU BGNL', 'BDIY', 'CRY', 'DXY', 'JPY', 'GBP', 'Cl1', 'VIX', 'USGG30YR',
    'GT10', 'USGG2YR', 'USGG3M', 'US0001M', 'GTDEM30Y', 'GTDEM10Y', 'GTDEM2Y',
    'EONIA', 'LUMSTRUU', 'LUACTRUU', 'LG30TRUU', 'LP01TREU', 'EMUSTRUU', 'MXUS'
]])

data['anomaly_label_recent'] = model_recent_data.predict(data[[
    'XAU BGNL', 'BDIY', 'CRY', 'DXY', 'JPY', 'GBP', 'Cl1', 'VIX', 'USGG30YR',
    'GT10', 'USGG2YR', 'USGG3M', 'US0001M', 'GTDEM30Y', 'GTDEM10Y', 'GTDEM2Y',
    'EONIA', 'LUMSTRUU', 'LUACTRUU', 'LG30TRUU', 'LP01TREU', 'EMUSTRUU', 'MXUS'
]])

data = pd.DataFrame(data)

percentage_anomaly = (len(data.loc[data['anomaly_label_recent'] == -1]) /
                      len(data)) * 100

investment_strategy = {"equities": 0, "bonds": 0, "cash": 0}

if percentage_anomaly < 40:
    # Normal market conditions: go for high-risk allocation
    investment_strategy = {"equities": 70, "bonds": 20, "cash": 10}

elif percentage_anomaly < 70 and percentage_anomaly >= 40:
    # Moderate risk: You should hedge part of the portfolio
    investment_strategy = {"equities": 50, "bonds": 40, "cash": 10}

else:
    # Anomaly detected: You should shift to conservative assets
    investment_strategy = {"equities": 20, "bonds": 70, "cash": 10}

explanation = suggest_strategy(percentage_anomaly, investment_strategy)

st.markdown("---")

st.markdown("Investment Strategy Suggestion")

st.markdown(explanation)
