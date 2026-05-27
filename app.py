from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import pandas as pd
import torch
import pulp
from chronos import ChronosPipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import datetime
import json

app = Flask(__name__)
CORS(app)

# ── Load models once at startup ──
print("Loading Chronos model...")
pipe = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    dtype=torch.float32
)

print("Loading FAISS vector store...")
docs = [
    "Copper prices forecast to rise 12% in Q3 due to Chile mine strike.",
    "Supplier A lead time increased from 3 to 6 weeks updated May 2025.",
    "Sea freight rates from Shanghai up 30% use air for urgent orders.",
    "Supplier B offers 8% volume discount above 500 units.",
    "Steel shortage expected in Q4 due to port strikes in South Korea.",
    "Supplier C has been blacklisted in winter months due to delays.",
    "Cotton prices stable no major disruptions expected this quarter.",
    "Warehouse storage costs rising 15% recommend reducing buffer stock.",
]
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_texts(docs, embedder)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("All models ready!")

# ── Helper functions ──
def get_context(sku, n=52):
    conn = sqlite3.connect("intellistock.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT sales FROM sales WHERE sku=? ORDER BY date DESC LIMIT ?",
        (sku, n)
    )
    rows = cursor.fetchall()
    conn.close()
    series = [row[0] for row in reversed(rows)]
    return torch.tensor(series, dtype=torch.float32)

def predict_demand(sku, horizon=4):
    context = get_context(sku, n=52)

    print(f"\nSKU: {sku}")
    print("Last 10 sales values:")
    print(context[-10:])

    forecast = pipe.predict(
        inputs=[context],
        prediction_length=horizon,
        num_samples=20
    )

    prediction = round(
        float(forecast[0].median(dim=0).values.sum()),
        2
    )

    print("Prediction:", prediction)

    return prediction

def optimize_inventory(predicted_demand, unit_cost=50,
                       storage_cost=2, budget=30000,
                       storage_capacity=1000, min_order=100):
    model = pulp.LpProblem("Inventory_Optimization", pulp.LpMinimize)
    order_qty = pulp.LpVariable("order_quantity",
                                lowBound=min_order, cat="Integer")
    model += (unit_cost * order_qty + storage_cost * order_qty)
    model += order_qty >= predicted_demand
    model += unit_cost * order_qty <= budget
    model += order_qty <= storage_capacity
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    return {
        "status": pulp.LpStatus[model.status],
        "recommended_order": int(order_qty.varValue),
        "total_cost": round(pulp.value(model.objective), 2)
    }

def run_critic(optimization_result, market_context):
    warnings = []
    order_qty = optimization_result["recommended_order"]
    total_cost = optimization_result["total_cost"]
    if order_qty > 550:
        warnings.append("Order quantity exceeds safe storage threshold.")
    if total_cost > 27000:
        warnings.append("Total cost approaching budget ceiling.")
    if "blacklisted" in market_context.lower():
        warnings.append("Blacklisted supplier detected in market context.")
    verdict = "REVISE" if len([w for w in warnings
                               if "blacklisted" not in w]) > 0 else "PASS"
    return {"verdict": verdict, "warnings": warnings}

# ── API Routes ──
@app.route("/")
def home():
    return jsonify({
        "message": "IntelliStock Backend Running"
    })
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "IntelliStock API running"})

@app.route("/api/skus", methods=["GET"])
def get_skus():
    conn = sqlite3.connect("intellistock.db")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT sku FROM sales")
    skus = [row[0] for row in cursor.fetchall()]
    conn.close()
    return jsonify({"skus": skus})

@app.route("/api/run", methods=["POST"])
def run_pipeline():
    data = request.json
    sku = data.get("sku", "ITEM_001")

    try:
        # Phase 2 - Predict
        predicted_demand = predict_demand(sku)

        # Phase 3 - RAG
        rag_docs = retriever.invoke(
            "supplier delays shipping issues price changes"
        )
        market_context = "\n".join([d.page_content for d in rag_docs])
        market_docs = [d.page_content for d in rag_docs]

        # Phase 4 - Optimize
        optimization_result = optimize_inventory(predicted_demand)

        # Phase 5 - Critic
        critic_result = run_critic(optimization_result, market_context)

        # Phase 6 - Report
        report = {
            "run_date": datetime.date.today().isoformat(),
            "sku": sku,
            "predicted_demand_4_weeks": predicted_demand,
            "recommended_order_qty": optimization_result["recommended_order"],
            "total_cost": optimization_result["total_cost"],
            "optimization_status": optimization_result["status"],
            "critic_verdict": critic_result["verdict"],
            "warnings": critic_result["warnings"],
            "market_context": market_docs,
        }

        # Save report
        with open(f"report_{sku}.json", "w") as f:
            json.dump(report, f, indent=2)

        return jsonify({"success": True, "report": report})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
