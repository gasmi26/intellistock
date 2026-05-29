🧠 What IntelliStock Is

An AI-powered autonomous procurement system that takes a product SKU as input and automatically decides how much to order, when to flag risks, and generates an executive report — all without human intervention.

🔄 FULL DATA FLOW

SKU Input (React)
      ↓
Flask /api/run
      ↓
SQLite → Chronos-T5 → predicted_demand
      ↓
FAISS RAG → market_context (fetched once)
      ↓
PuLP LP Solver → order_qty + total_cost
      ↓
Llama3 Critic → verdict + reasoning
      ↓
Llama3 Reporter → ai_report
      ↓
JSON report saved + returned to React frontend


🏗️ TECH STACK

<img width="1108" height="534" alt="image" src="https://github.com/user-attachments/assets/5439f425-d5d8-4ac5-8ee9-cef9efbb8c25" />
