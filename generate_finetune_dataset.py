"""
=============================================================================
  JSONL Dataset Generator for Qwen2.5 Fine-Tuning
  ─────────────────────────────────────────────────
  Project : Federated Learning Supply Chain 5.0 — Milk
  Purpose : Convert 3 CSV client datasets into a high-quality ChatML
            instruction-following JSONL file for SFT.
  Output  : milk_supply_chain_qwen.jsonl
=============================================================================
"""

import pandas as pd
import numpy as np
import json
import os
import random
from pathlib import Path

random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATASETS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent / "DATASETS"

clients = {
    "Amul_Gujarat": {
        "file": BASE_DIR / "client_1_amul_gujarat.csv",
        "brand": "Amul",
        "region": "Gujarat",
        "client_id": 1,
    },
    "MotherDairy_Delhi": {
        "file": BASE_DIR / "client_2_mother_dairy_delhi.csv",
        "brand": "Mother Dairy",
        "region": "Delhi",
        "client_id": 2,
    },
    "Sudha_Bihar": {
        "file": BASE_DIR / "client_3_sudha_bihar.csv",
        "brand": "Sudha",
        "region": "Bihar",
        "client_id": 3,
    },
}

dataframes = {}
for name, meta in clients.items():
    df = pd.read_csv(meta["file"])
    df["date"] = pd.to_datetime(df["date"])
    dataframes[name] = df
    print(f"✅ Loaded {name}: {len(df)} rows")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are SupplyChainGPT, a senior supply chain strategist specializing in "
    "perishable dairy logistics across India. You have expertise in demand "
    "forecasting, cold-chain management, inventory optimization, sustainability "
    "metrics, and financial analysis. You provide data-driven, professional, "
    "and actionable recommendations. Always structure your response with clear "
    "headings, bullet points, and quantified insights wherever possible."
)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  TEMPLATE GENERATORS
# ──────────────────────────────────────────────────────────────────────────────

def fmt(v, places=2):
    """Format a number nicely."""
    if isinstance(v, float):
        return f"{v:,.{places}f}"
    return f"{v:,}"


def gen_weekly_analysis(row, brand, region):
    """Generate a weekly snapshot analysis Q&A."""
    date_str = row["date"].strftime("%B %d, %Y")
    week = int(row["week"])

    user = (
        f"Analyze the supply chain performance for {brand} ({region}) for the "
        f"week of {date_str} (Week {week}). The data is as follows:\n"
        f"- Demand: {fmt(row['demand'], 0)} units\n"
        f"- Supply: {fmt(row['supply'], 0)} units\n"
        f"- Quantity Sold: {fmt(row['quantity_sold'], 0)} units\n"
        f"- Production Volume: {fmt(row['production_volume'], 0)} units\n"
        f"- Price per Unit: ₹{fmt(row['price_per_unit'])}\n"
        f"- Cost per Unit: ₹{fmt(row['cost_per_unit'])}\n"
        f"- Total Revenue: ₹{fmt(row['total_revenue'])}\n"
        f"- Profit Margin: {fmt(row['profit_margin'] * 100, 1)}%\n"
        f"- Inventory Level: {fmt(row['inventory_level'], 0)} units\n"
        f"- Min Stock Threshold: {fmt(row['min_stock_threshold'])}\n"
        f"- Reorder Quantity: {fmt(row['reorder_quantity'])}\n"
        f"- Disruption Probability: {fmt(row['disruption_prob'] * 100, 1)}%\n"
        f"- Emission Factor: {fmt(row['emission_factor'])}\n"
        f"- Shelf Life: {int(row['shelf_life_days'])} days\n"
        f"- Storage: {row['storage_condition']}\n\n"
        f"Provide a comprehensive performance summary and strategic recommendations."
    )

    # Build intelligent response based on data
    demand = row["demand"]
    supply = row["supply"]
    sold = row["quantity_sold"]
    inv = row["inventory_level"]
    margin = row["profit_margin"]
    disruption = row["disruption_prob"]
    emission = row["emission_factor"]
    shelf = int(row["shelf_life_days"])
    gap = demand - supply
    fill_rate = (sold / demand * 100) if demand > 0 else 0
    wastage_risk = "HIGH" if shelf <= 3 and inv > demand * 0.5 else "MODERATE" if shelf <= 5 else "LOW"
    supply_status = "SURPLUS" if supply >= demand * 1.1 else "BALANCED" if supply >= demand * 0.9 else "DEFICIT"
    risk_level = "HIGH" if disruption > 0.3 else "MODERATE" if disruption > 0.15 else "LOW"

    assistant = (
        f"## Weekly Performance Report — {brand} ({region})\n"
        f"**Period:** Week {week} ({date_str})\n\n"
        f"### 📊 Key Performance Indicators\n"
        f"| Metric | Value | Status |\n"
        f"|--------|-------|--------|\n"
        f"| Demand-Supply Gap | {fmt(gap, 0)} units | {supply_status} |\n"
        f"| Order Fill Rate | {fmt(fill_rate, 1)}% | {'✅ Healthy' if fill_rate > 90 else '⚠️ Below Target'} |\n"
        f"| Profit Margin | {fmt(margin * 100, 1)}% | {'✅ Strong' if margin > 0.42 else '⚠️ Needs Attention'} |\n"
        f"| Inventory Health | {fmt(inv, 0)} units | {'⚠️ Overstocked' if inv > demand * 1.5 else '✅ Optimal' if inv > row['min_stock_threshold'] else '🔴 Critical'} |\n\n"
        f"### 🔍 Analysis\n\n"
        f"**Supply Chain Status:** The {supply_status.lower()} condition indicates that "
    )

    if supply_status == "SURPLUS":
        assistant += (
            f"supply exceeded demand by {fmt(supply - demand, 0)} units. While this provides buffer stock, "
            f"it increases holding costs and wastage risk, especially given the {shelf}-day shelf life "
            f"under {row['storage_condition']} conditions.\n\n"
        )
    elif supply_status == "DEFICIT":
        assistant += (
            f"there is a shortfall of {fmt(demand - supply, 0)} units. This risks lost sales and customer "
            f"dissatisfaction. Immediate procurement action is recommended to bridge the gap.\n\n"
        )
    else:
        assistant += (
            f"supply and demand are well-aligned, minimizing both stockout risk and holding costs. "
            f"Continue monitoring to maintain this balance.\n\n"
        )

    assistant += (
        f"**Risk Assessment:** Disruption probability is at {fmt(disruption * 100, 1)}% ({risk_level} risk). "
    )
    if risk_level == "HIGH":
        assistant += (
            f"This is significantly elevated. Recommend activating contingency procurement channels "
            f"and increasing safety stock by 15-20%.\n\n"
        )
    else:
        assistant += f"This is within acceptable operational bounds.\n\n"

    assistant += (
        f"**Sustainability:** Emission factor of {fmt(emission)} is "
        f"{'above the target threshold — consider route optimization and consolidating shipments' if emission > 1.4 else 'within the green corridor — maintain current logistics practices'}.\n\n"
        f"**Spoilage Risk:** {wastage_risk}. With {shelf}-day shelf life in {row['storage_condition']} storage, "
    )
    if wastage_risk == "HIGH":
        assistant += "implement FIFO strictly and consider flash sales or B2B redistribution for near-expiry stock.\n\n"
    else:
        assistant += "current inventory turnover is manageable.\n\n"

    assistant += (
        f"### 💡 Strategic Recommendations\n"
        f"1. **Order Quantity:** Target reorder of ~{fmt(row['reorder_quantity'], 0)} units when inventory hits {fmt(row['min_stock_threshold'])} units.\n"
        f"2. **Pricing:** At ₹{fmt(row['price_per_unit'])} with ₹{fmt(row['cost_per_unit'])} cost, the margin of {fmt(margin * 100, 1)}% is {'healthy — maintain current pricing strategy' if margin > 0.42 else 'thin — explore cost reduction or premium SKU variants'}.\n"
        f"3. **Production:** Align production volume (~{fmt(row['production_volume'], 0)} units) more closely with forecasted demand to reduce overproduction waste.\n"
        f"4. **Cold Chain:** {'Upgrade to refrigerated transport for longer shelf life' if row['storage_condition'] == 'Polythene Packet' else 'Current cold chain setup is adequate'}."
    )

    return user, assistant


def gen_trend_analysis(df_slice, brand, region):
    """Generate a multi-week trend analysis Q&A."""
    weeks = df_slice["week"].values
    start_date = df_slice["date"].iloc[0].strftime("%B %Y")
    end_date = df_slice["date"].iloc[-1].strftime("%B %Y")

    avg_demand = df_slice["demand"].mean()
    avg_supply = df_slice["supply"].mean()
    avg_margin = df_slice["profit_margin"].mean()
    avg_disruption = df_slice["disruption_prob"].mean()
    total_revenue = df_slice["total_revenue"].sum()
    demand_trend = "increasing" if df_slice["demand"].iloc[-1] > df_slice["demand"].iloc[0] else "decreasing"
    volatility = df_slice["demand"].std()

    user = (
        f"Provide a trend analysis for {brand} ({region}) milk supply chain from "
        f"{start_date} to {end_date} (Weeks {int(weeks[0])}-{int(weeks[-1])}).\n"
        f"Key statistics:\n"
        f"- Average weekly demand: {fmt(avg_demand, 0)} units\n"
        f"- Average weekly supply: {fmt(avg_supply, 0)} units\n"
        f"- Total revenue (period): ₹{fmt(total_revenue)}\n"
        f"- Average profit margin: {fmt(avg_margin * 100, 1)}%\n"
        f"- Average disruption probability: {fmt(avg_disruption * 100, 1)}%\n"
        f"- Demand volatility (σ): {fmt(volatility, 1)} units\n"
        f"- Demand trend: {demand_trend}\n"
    )

    assistant = (
        f"## Trend Analysis — {brand} ({region})\n"
        f"**Period:** {start_date} to {end_date}\n\n"
        f"### 📈 Demand Dynamics\n"
        f"Demand has been **{demand_trend}** over this period with a standard deviation of "
        f"{fmt(volatility, 1)} units. "
    )
    if volatility > 100:
        assistant += (
            "This high volatility suggests significant demand fluctuations, likely driven by "
            "seasonal patterns, promotional activity, or market disruptions. A more responsive "
            "forecasting model (e.g., LSTM-based federated prediction) is recommended.\n\n"
        )
    else:
        assistant += (
            "This relatively stable demand pattern allows for more predictable production planning "
            "and inventory management.\n\n"
        )

    gap_pct = ((avg_supply - avg_demand) / avg_demand * 100) if avg_demand > 0 else 0
    assistant += (
        f"### 📦 Supply-Demand Alignment\n"
        f"Average supply gap is **{'+' if gap_pct > 0 else ''}{fmt(gap_pct, 1)}%** relative to demand. "
        f"{'This surplus provides a comfortable buffer but may increase wastage costs for perishables.' if gap_pct > 5 else 'The supply chain is well-calibrated to demand levels.' if abs(gap_pct) < 5 else 'A persistent deficit indicates capacity constraints requiring attention.'}\n\n"
        f"### 💰 Financial Health\n"
        f"Total revenue of **₹{fmt(total_revenue)}** with an average margin of **{fmt(avg_margin * 100, 1)}%** "
        f"indicates {'strong profitability — focus on scaling operations' if avg_margin > 0.42 else 'moderate profitability — explore cost efficiencies'}.\n\n"
        f"### ⚠️ Risk Profile\n"
        f"Average disruption probability of **{fmt(avg_disruption * 100, 1)}%** is "
        f"{'elevated — diversify supplier base and increase safety stock' if avg_disruption > 0.25 else 'manageable — maintain current risk mitigation strategies'}.\n\n"
        f"### 🎯 Action Items\n"
        f"1. {'Implement demand smoothing strategies to reduce volatility' if volatility > 100 else 'Maintain current demand management approach'}\n"
        f"2. {'Reduce buffer stock to minimize wastage' if gap_pct > 10 else 'Consider increasing safety stock' if gap_pct < -5 else 'Current inventory policy is optimal'}\n"
        f"3. Target margin improvement to 45%+ through procurement negotiation\n"
        f"4. Deploy predictive models for proactive disruption management"
    )

    return user, assistant


def gen_comparison(rows_dict):
    """Generate a cross-client comparison Q&A."""
    user = "Compare the supply chain performance across all three milk brands:\n\n"
    for name, (row, brand, region) in rows_dict.items():
        user += (
            f"**{brand} ({region}):**\n"
            f"  Demand: {fmt(row['demand'], 0)} | Supply: {fmt(row['supply'], 0)} | "
            f"Margin: {fmt(row['profit_margin'] * 100, 1)}% | "
            f"Disruption: {fmt(row['disruption_prob'] * 100, 1)}%\n"
        )
    user += "\nProvide a comparative analysis with strategic insights."

    # Build comparison
    items = list(rows_dict.values())
    margins = [(row["profit_margin"], brand) for row, brand, region in items]
    best_margin = max(margins, key=lambda x: x[0])
    worst_margin = min(margins, key=lambda x: x[0])

    assistant = (
        "## Cross-Regional Performance Comparison\n\n"
        "### 📊 Comparative Dashboard\n"
        "| Metric | Amul (Gujarat) | Mother Dairy (Delhi) | Sudha (Bihar) |\n"
        "|--------|---------------|---------------------|---------------|\n"
    )

    for metric, key, fmt_fn in [
        ("Demand", "demand", lambda x: fmt(x, 0)),
        ("Supply", "supply", lambda x: fmt(x, 0)),
        ("Margin", "profit_margin", lambda x: fmt(x * 100, 1) + "%"),
        ("Disruption", "disruption_prob", lambda x: fmt(x * 100, 1) + "%"),
    ]:
        vals = [fmt_fn(row[key]) for row, _, _ in items]
        assistant += f"| {metric} | {vals[0]} | {vals[1]} | {vals[2]} |\n"

    assistant += (
        f"\n### 🏆 Key Findings\n"
        f"- **Highest Margin:** {best_margin[1]} at {fmt(best_margin[0] * 100, 1)}% — indicates superior cost control or premium pricing power.\n"
        f"- **Lowest Margin:** {worst_margin[1]} at {fmt(worst_margin[0] * 100, 1)}% — requires cost structure review.\n"
        f"- **Scale Advantage:** Mother Dairy (Delhi) operates at the highest volume, leveraging urban demand density.\n"
        f"- **Regional Risk:** Bihar operations face higher supply chain disruption probabilities, reflecting infrastructure challenges.\n\n"
        f"### 💡 Strategic Recommendations\n"
        f"1. **Gujarat (Amul):** Leverage cooperative network for cost-efficient scaling.\n"
        f"2. **Delhi (Mother Dairy):** Focus on last-mile optimization in urban distribution.\n"
        f"3. **Bihar (Sudha):** Invest in cold chain infrastructure and supplier diversification to de-risk operations."
    )

    return user, assistant


def gen_inventory_advisory(row, brand, region):
    """Generate an inventory management advisory Q&A."""
    inv = row["inventory_level"]
    min_stock = row["min_stock_threshold"]
    demand = row["demand"]
    shelf = int(row["shelf_life_days"])
    storage = row["storage_condition"]

    user = (
        f"As inventory manager for {brand} ({region}), I need guidance on the current stock position:\n"
        f"- Current Inventory: {fmt(inv, 0)} units\n"
        f"- Minimum Stock Threshold: {fmt(min_stock)} units\n"
        f"- Current Demand: {fmt(demand, 0)} units/week\n"
        f"- Shelf Life: {shelf} days ({storage})\n"
        f"- Reorder Quantity: {fmt(row['reorder_quantity'])} units\n\n"
        f"Should I place a reorder? What is the optimal restocking strategy?"
    )

    days_of_stock = (inv / (demand / 7)) if demand > 0 else 999
    should_reorder = inv <= min_stock * 1.2 or days_of_stock < shelf

    assistant = (
        f"## Inventory Advisory — {brand} ({region})\n\n"
        f"### 📦 Current Stock Assessment\n"
        f"- **Days of Stock Remaining:** ~{fmt(days_of_stock, 1)} days\n"
        f"- **Stock vs. Threshold:** {'🔴 BELOW minimum' if inv < min_stock else '🟡 NEAR minimum' if inv < min_stock * 1.2 else '✅ ABOVE minimum'}\n"
        f"- **Spoilage Window:** {shelf} days shelf life\n\n"
        f"### 🎯 Recommendation: {'**REORDER NOW**' if should_reorder else '**HOLD — No immediate reorder needed**'}\n\n"
    )

    if should_reorder:
        safety_factor = 1.2 if row["disruption_prob"] > 0.2 else 1.1
        optimal_qty = int(row["reorder_quantity"] * safety_factor)
        assistant += (
            f"**Optimal Order Quantity:** {fmt(optimal_qty, 0)} units "
            f"(base: {fmt(row['reorder_quantity'], 0)} × {safety_factor} safety factor)\n\n"
            f"**Rationale:**\n"
            f"- Current stock of {fmt(inv, 0)} units covers only ~{fmt(days_of_stock, 1)} days\n"
            f"- With {shelf}-day shelf life, timing is critical to avoid stockouts\n"
            f"- Disruption probability of {fmt(row['disruption_prob'] * 100, 1)}% warrants {'enhanced' if row['disruption_prob'] > 0.2 else 'standard'} safety buffer\n\n"
            f"**Urgency:** {'🔴 HIGH — Place order immediately' if inv < min_stock else '🟡 MEDIUM — Order within 24 hours'}\n"
        )
    else:
        assistant += (
            f"**Rationale:**\n"
            f"- Stock of {fmt(inv, 0)} units provides ~{fmt(days_of_stock, 1)} days coverage\n"
            f"- Well above the minimum threshold of {fmt(min_stock)} units\n"
            f"- Monitor daily consumption rate and reorder when stock drops to {fmt(min_stock * 1.1, 0)} units\n"
        )

    return user, assistant


def gen_sustainability_insight(row, brand, region):
    """Generate a sustainability-focused Q&A."""
    user = (
        f"Evaluate the sustainability metrics for {brand} ({region}) operations:\n"
        f"- Emission Factor: {fmt(row['emission_factor'])}\n"
        f"- Disruption Probability: {fmt(row['disruption_prob'] * 100, 1)}%\n"
        f"- Storage Condition: {row['storage_condition']}\n"
        f"- Shelf Life: {int(row['shelf_life_days'])} days\n"
        f"- Production Volume: {fmt(row['production_volume'], 0)} units\n"
        f"- Demand: {fmt(row['demand'], 0)} units\n\n"
        f"How can we reduce environmental impact while maintaining supply chain efficiency?"
    )

    overproduction = row["production_volume"] - row["demand"]
    emission = row["emission_factor"]
    carbon_grade = "A" if emission < 1.15 else "B" if emission < 1.35 else "C" if emission < 1.5 else "D"

    assistant = (
        f"## Sustainability Assessment — {brand} ({region})\n\n"
        f"### 🌍 Environmental Scorecard\n"
        f"| Metric | Value | Grade |\n"
        f"|--------|-------|-------|\n"
        f"| Carbon Intensity | {fmt(emission)} CO₂e/unit | {carbon_grade} |\n"
        f"| Overproduction | {fmt(overproduction, 0)} units | {'🔴 High Waste Risk' if overproduction > 200 else '✅ Controlled'} |\n"
        f"| Cold Chain | {row['storage_condition']} | {'✅ Optimal' if row['storage_condition'] == 'Refrigerated' else '⚠️ Can Improve'} |\n\n"
        f"### 🔧 Decarbonization Roadmap\n\n"
        f"**1. Production Optimization**\n"
        f"- Current overproduction of {fmt(overproduction, 0)} units leads to potential spoilage waste\n"
        f"- Align production with LSTM-based demand forecasts to achieve <5% overproduction\n"
        f"- Estimated waste reduction: {fmt(abs(overproduction) * 0.3, 0)} units/week\n\n"
        f"**2. Logistics & Transport**\n"
        f"- Emission factor of {fmt(emission)} {'exceeds the 1.3 target — implement route optimization using federated fleet data' if emission > 1.3 else 'is within target — maintain current logistics efficiency'}\n"
        f"- Consider consolidating shipments during {'low-demand seasons (monsoon/summer)' if region == 'Bihar' else 'off-peak hours for urban distribution' if region == 'Delhi' else 'inter-state transit corridors'}\n\n"
        f"**3. Packaging & Storage**\n"
        f"- {'Transition from Polythene Packet to recyclable Tetra Pack to extend shelf life and reduce plastic waste' if row['storage_condition'] == 'Polythene Packet' else 'Current packaging is aligned with sustainability goals'}\n"
        f"- {'Refrigerated storage extends shelf life and reduces spoilage-related emissions' if row['storage_condition'] != 'Refrigerated' else 'Refrigerated cold chain minimizes waste effectively'}\n\n"
        f"**4. Target Metrics**\n"
        f"- Carbon intensity: Reduce to <1.2 CO₂e/unit within 6 months\n"
        f"- Spoilage rate: Target <3% of production volume\n"
        f"- Renewable energy: 30% of cold chain operations by next fiscal year"
    )

    return user, assistant


def gen_forecast_explanation(df_slice, forecast_val, brand, region):
    """Generate a forecast explanation Q&A."""
    recent_demands = df_slice["demand"].values[-5:]
    avg_recent = np.mean(recent_demands)

    user = (
        f"The Federated LSTM model predicts a demand of {forecast_val} units for {brand} "
        f"({region}) next week. Recent 5-week demands were: {', '.join(str(int(d)) for d in recent_demands)}. "
        f"Explain this forecast and recommend an appropriate order quantity."
    )

    trend = "upward" if forecast_val > avg_recent else "downward" if forecast_val < avg_recent else "stable"
    diff_pct = ((forecast_val - avg_recent) / avg_recent * 100) if avg_recent > 0 else 0

    assistant = (
        f"## Demand Forecast Analysis — {brand} ({region})\n\n"
        f"### 📉 Forecast Breakdown\n"
        f"- **Predicted Demand:** {fmt(forecast_val, 0)} units\n"
        f"- **5-Week Average:** {fmt(avg_recent, 0)} units\n"
        f"- **Trend Direction:** {trend.capitalize()} ({'+' if diff_pct > 0 else ''}{fmt(diff_pct, 1)}%)\n\n"
        f"### 🧠 Model Interpretation\n"
        f"The Federated LSTM model has identified a **{trend} trend** in demand. "
        f"The forecast of {fmt(forecast_val, 0)} units represents a "
        f"{fmt(abs(diff_pct), 1)}% {'increase' if diff_pct > 0 else 'decrease'} from the rolling average. "
    )
    if abs(diff_pct) > 15:
        assistant += (
            "This significant deviation may be driven by seasonal factors, market events, or "
            "competitor activity. Cross-validate with external market intelligence.\n\n"
        )
    else:
        assistant += "This is within normal demand variation bounds.\n\n"

    safety_stock = int(np.std(recent_demands) * 1.65)  # 95% service level
    recommended_order = int(forecast_val + safety_stock)

    assistant += (
        f"### 📦 Recommended Order\n"
        f"- **Base Order:** {fmt(forecast_val, 0)} units (forecast)\n"
        f"- **Safety Stock:** {fmt(safety_stock, 0)} units (σ × 1.65 for 95% service level)\n"
        f"- **Total Recommended:** {fmt(recommended_order, 0)} units\n\n"
        f"### ⚡ Confidence\n"
        f"- Model trained across {3} federated clients with differential privacy (ε-bounded)\n"
        f"- Historical accuracy within ±8% for similar demand patterns\n"
        f"- Recommendation: Proceed with {fmt(recommended_order, 0)} units and review mid-week"
    )

    return user, assistant


# ──────────────────────────────────────────────────────────────────────────────
# 4.  GENERATE ALL SAMPLES
# ──────────────────────────────────────────────────────────────────────────────

samples = []

for name, meta in clients.items():
    df = dataframes[name]
    brand = meta["brand"]
    region = meta["region"]

    # --- Weekly Analysis (sample ~50% of the dataset) ---
    indices = sorted(random.sample(range(len(df)), min(250, len(df))))
    for idx in indices:
        row = df.iloc[idx]
        u, a = gen_weekly_analysis(row, brand, region)
        samples.append({"system": SYSTEM_PROMPT, "user": u, "assistant": a})

    # --- Trend Analysis (multi-week windows, overlapping) ---
    for start in range(0, len(df) - 12, 5):
        window = df.iloc[start : start + 12]
        u, a = gen_trend_analysis(window, brand, region)
        samples.append({"system": SYSTEM_PROMPT, "user": u, "assistant": a})

    # --- Inventory Advisory (sample ~30% of rows) ---
    inv_indices = sorted(random.sample(range(len(df)), min(150, len(df))))
    for idx in inv_indices:
        row = df.iloc[idx]
        u, a = gen_inventory_advisory(row, brand, region)
        samples.append({"system": SYSTEM_PROMPT, "user": u, "assistant": a})

    # --- Sustainability (sample ~20% of rows) ---
    sus_indices = sorted(random.sample(range(len(df)), min(100, len(df))))
    for idx in sus_indices:
        row = df.iloc[idx]
        u, a = gen_sustainability_insight(row, brand, region)
        samples.append({"system": SYSTEM_PROMPT, "user": u, "assistant": a})

    # --- Forecast Explanations (sample 100 windows) ---
    for _ in range(100):
        start = random.randint(10, len(df) - 6)
        window = df.iloc[start : start + 5]
        forecast = int(df.iloc[start + 5]["demand"] * random.uniform(0.9, 1.1))
        u, a = gen_forecast_explanation(window, forecast, brand, region)
        samples.append({"system": SYSTEM_PROMPT, "user": u, "assistant": a})


# --- Cross-client comparisons (sample ~100 matching weeks) ---
min_len = min(len(df) for df in dataframes.values())
comp_indices = sorted(random.sample(range(min_len), min(100, min_len)))
names_list = list(clients.keys())
for idx in comp_indices:
    rows_dict = {}
    for name in names_list:
        meta = clients[name]
        rows_dict[name] = (dataframes[name].iloc[idx], meta["brand"], meta["region"])
    u, a = gen_comparison(rows_dict)
    samples.append({"system": SYSTEM_PROMPT, "user": u, "assistant": a})


# ──────────────────────────────────────────────────────────────────────────────
# 5.  CONVERT TO ChatML JSONL
# ──────────────────────────────────────────────────────────────────────────────

random.shuffle(samples)

output_path = Path(__file__).parent / "milk_supply_chain_qwen.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for sample in samples:
        entry = {
            "messages": [
                {"role": "system", "content": sample["system"]},
                {"role": "user", "content": sample["user"]},
                {"role": "assistant", "content": sample["assistant"]},
            ]
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"\n{'='*60}")
print(f"✅ Dataset generated successfully!")
print(f"   Total samples : {len(samples)}")
print(f"   Output file   : {output_path}")
print(f"   Format        : ChatML messages JSONL")
print(f"{'='*60}")
