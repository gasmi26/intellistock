from flask import Flask, jsonify, request
from flask_cors import CORS

import sqlite3
import torch
import pulp
import datetime
import json
import traceback
import requests

from bs4 import BeautifulSoup
from groq import Groq
from chronos import ChronosPipeline

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================
# Flask App
# ==========================================
app = Flask(__name__)
CORS(app)

# ==========================================
# Groq Client
# ==========================================
client = Groq(
    api_key="YOUR_API_KEY"
)

# ==========================================
# Load Chronos Forecasting Model
# ==========================================
print("\nLoading Chronos forecasting model...")

pipe = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    dtype=torch.float32
)

print("Chronos model loaded!")

# ==========================================
# Static Supply Chain Docs
# ==========================================
docs = [
    "Copper prices forecast to rise 12% in Q3 due to Chile mine strike.",
    "Supplier A lead time increased from 3 to 6 weeks updated May 2025.",
    "Sea freight rates from Shanghai up 30% use air for urgent orders.",
    "Supplier B offers 8% volume discount above 500 units.",
    "Steel shortage expected in Q4 due to port strikes in South Korea.",
    "Supplier C has been blacklisted in winter months due to delays.",
    "Cotton prices stable no major disruptions expected this quarter.",
    "Warehouse storage costs rising 15% recommend reducing buffer stock."
]

# ==========================================
# Live Web Scraping
# ==========================================
print("\nScraping live market intelligence...")

scraped_docs = []

try:

    url = "https://news.ycombinator.com/"

    response = requests.get(url)

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    titles = soup.find_all(
        "span",
        class_="titleline"
    )

    for t in titles[:15]:

        text = t.get_text(strip=True)

        scraped_docs.append(text)

    print("\nLive Scraped Headlines:\n")

    for doc in scraped_docs:
        print("-", doc)

except Exception as e:

    print("Scraping failed:", e)

# ==========================================
# Combine Static + Scraped Docs
# ==========================================
all_docs = docs + scraped_docs

print(f"\nTotal documents loaded: {len(all_docs)}")

# ==========================================
# Embedding Model
# ==========================================
print("\nLoading embedding model...")

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# Build FAISS Vector Database
# ==========================================
print("\nBuilding FAISS vector database...")

vectorstore = FAISS.from_texts(
    all_docs,
    embedder
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("FAISS vector database ready!")

# ==========================================
# Get Historical Sales Data
# ==========================================
def get_context(sku, n=52):

    conn = sqlite3.connect("intellistock.db")

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT sales
        FROM sales
        WHERE sku=?
        ORDER BY date DESC
        LIMIT ?
        """,
        (sku, n)
    )

    rows = cursor.fetchall()

    conn.close()

    series = [row[0] for row in reversed(rows)]

    # Fallback if no sales data exists
    if len(series) == 0:

        print(f"No sales data found for {sku}")

        series = [100] * 52

    return torch.tensor(
        series,
        dtype=torch.float32
    )

# ==========================================
# Demand Forecasting
# ==========================================
def predict_demand(sku, horizon=4):

    context = get_context(sku)

    print(f"\nForecasting demand for {sku}")

    forecast = pipe.predict(
        inputs=[context],
        prediction_length=horizon,
        num_samples=20
    )

    prediction = round(
        float(
            forecast[0]
            .median(dim=0)
            .values
            .sum()
        ),
        2
    )

    print(f"Predicted Demand: {prediction}")

    return prediction

# ==========================================
# Inventory Optimization
# ==========================================
def optimize_inventory(
    predicted_demand,
    unit_cost=50,
    storage_cost=2,
    budget=30000,
    storage_capacity=1000,
    min_order=100
):

    model = pulp.LpProblem(
        "Inventory_Optimization",
        pulp.LpMinimize
    )

    # Decision Variable
    order_qty = pulp.LpVariable(
        "order_quantity",
        lowBound=min_order,
        cat="Integer"
    )

    # Shortage Variable
    shortage = pulp.LpVariable(
        "shortage",
        lowBound=0,
        cat="Integer"
    )

    # Objective Function
    model += (
        unit_cost * order_qty
        + storage_cost * order_qty
        + 1000 * shortage
    )

    # Demand Constraint
    model += (
        order_qty + shortage
        >= predicted_demand
    )

    # Budget Constraint
    model += (
        unit_cost * order_qty
        <= budget
    )

    # Storage Constraint
    model += (
        order_qty
        <= storage_capacity
    )

    # Solve
    model.solve(
        pulp.PULP_CBC_CMD(msg=0)
    )

    return {

        "status":
            pulp.LpStatus[model.status],

        "recommended_order":
            int(order_qty.varValue or 0),

        "shortage":
            int(shortage.varValue or 0),

        "total_cost":
            round(
                pulp.value(model.objective) or 0,
                2
            )
    }

# ==========================================
# LLM Critic
# ==========================================
def llm_critic(
    sku,
    predicted_demand,
    order_qty,
    total_cost,
    market_context
):

    prompt = f"""
You are a procurement manager reviewing an inventory order.

SKU: {sku}

Predicted Demand:
{predicted_demand}

Recommended Order:
{order_qty}

Total Cost:
{total_cost}

Market Context:
{market_context}

Business Rules:
- Never order more than 600 units
- Total cost must not exceed 27000
- Flag blacklisted suppliers
- Flag if order exceeds 2x demand

Respond EXACTLY in this format:

VERDICT: PASS or REVISE
REASON: short explanation
WARNING: specific risk or NONE
"""

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = response.choices[0].message.content

    print("\nLLM Critic Result:\n")
    print(result)

    return result

# ==========================================
# AI Report Generator
# ==========================================
def llm_generate_report(
    sku,
    predicted_demand,
    order_qty,
    total_cost,
    market_context,
    verdict
):

    prompt = f"""
You are a supply chain analyst.

Write a professional procurement report
in EXACTLY 3 bullet points.

SKU: {sku}

Predicted Demand:
{predicted_demand}

Recommended Order:
{order_qty}

Total Cost:
{total_cost}

Market Intelligence:
{market_context}

Critic Verdict:
{verdict}

Format:
- Demand insight: ...
- Procurement recommendation: ...
- Risk alert: ...
"""

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = response.choices[0].message.content

    print("\nAI Report:\n")
    print(result)

    return result

# ==========================================
# Home Route
# ==========================================
@app.route("/")
def home():

    return jsonify({
        "message": "IntelliStock Backend Running"
    })

# ==========================================
# Health Check
# ==========================================
@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "message": "IntelliStock API running"
    })

# ==========================================
# Get Available SKUs
# ==========================================
@app.route("/api/skus", methods=["GET"])
def get_skus():

    conn = sqlite3.connect("intellistock.db")

    cursor = conn.cursor()

    cursor.execute(
        "SELECT DISTINCT sku FROM sales"
    )

    skus = [
        row[0]
        for row in cursor.fetchall()
    ]

    conn.close()

    return jsonify({
        "skus": skus
    })

# ==========================================
# Main AI Pipeline
# ==========================================
@app.route("/api/run", methods=["POST"])
def run_pipeline():

    try:

        print("\n===================================")
        print("NEW PIPELINE RUN")
        print("===================================")

        data = request.json

        sku = data.get(
            "sku",
            "ITEM_001"
        )

        print(f"\nRunning pipeline for: {sku}")

        # ==================================
        # Phase 1 - Forecast Demand
        # ==================================
        predicted_demand = predict_demand(sku)

        # ==================================
        # Phase 2 - RAG Retrieval
        # ==================================
        query = f"""
        SKU: {sku}

        Find:
        - supplier risks
        - freight delays
        - price changes
        - inventory risks
        - procurement alerts
        """

        rag_docs = retriever.invoke(query)

        print("\nRetrieved RAG Documents:\n")

        for doc in rag_docs:
            print("-", doc.page_content)

        market_context = "\n".join([
            d.page_content
            for d in rag_docs
        ])

        market_docs = [
            d.page_content
            for d in rag_docs
        ]

        # ==================================
        # Phase 3 - Optimization
        # ==================================
        optimization_result = optimize_inventory(
            predicted_demand
        )

        print("\nOptimization Result:\n")
        print(optimization_result)

        order_qty = optimization_result[
            "recommended_order"
        ]

        total_cost = optimization_result[
            "total_cost"
        ]

        shortage = optimization_result[
            "shortage"
        ]

        # ==================================
        # Phase 4 - LLM Critic
        # ==================================
        critic_result = llm_critic(
            sku=sku,
            predicted_demand=predicted_demand,
            order_qty=order_qty,
            total_cost=total_cost,
            market_context=market_context
        )

        verdict = (
            "PASS"
            if "VERDICT: PASS" in critic_result
            else "REVISE"
        )

        # ==================================
        # Phase 5 - AI Report
        # ==================================
        ai_report = llm_generate_report(
            sku=sku,
            predicted_demand=predicted_demand,
            order_qty=order_qty,
            total_cost=total_cost,
            market_context=market_context,
            verdict=critic_result
        )

        # ==================================
        # Final Report
        # ==================================
        report = {

            "run_date":
                datetime.date.today().isoformat(),

            "sku":
                sku,

            "predicted_demand_4_weeks":
                predicted_demand,

            "recommended_order_qty":
                order_qty,

            "shortage":
                shortage,

            "total_cost":
                total_cost,

            "optimization_status":
                optimization_result["status"],

            "critic_verdict":
                verdict,

            "critic_reasoning":
                critic_result,

            "ai_report":
                ai_report,

            "market_context":
                market_docs
        }

        # ==================================
        # Save Report
        # ==================================
        with open(
            f"report_{sku}.json",
            "w"
        ) as f:

            json.dump(
                report,
                f,
               indent=2
            )

        print("\nPipeline completed successfully!")

        return jsonify({
            "success": True,
            "report": report
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ==========================================
# Run Flask Server
# ==========================================
if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )
