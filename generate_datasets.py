"""
generate_datasets.py
====================
Generates 3 CSV datasets for Federated Learning Milk Supply Chain.
Each CSV represents a different dairy cooperative (FL client) with
non-IID data distributions calibrated from real Kaggle + Government data.

Clients:
  1. Amul (Gujarat)       - Large cooperative, high volume, stable
  2. Mother Dairy (Delhi)  - Urban distributor, high demand, price-sensitive
  3. Sudha (Bihar)         - Regional cooperative, seasonal, rural

Output: DATASETS/client_1_amul_gujarat.csv
        DATASETS/client_2_mother_dairy_delhi.csv
        DATASETS/client_3_sudha_bihar.csv

Each: 500 rows (weeks) × 20 columns
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

np.random.seed(42)

# =====================================================
# Client Profiles (Calibrated from real data)
# =====================================================
# Government data reference:
#   Gujarat price: ₹53-54, production: 12,784 MT (2016-17)
#   Delhi price: ₹55-56, production: 279 MT (urban, imports heavily)
#   Bihar price: ₹49-51, production: 8,711 MT (2016-17)
#
# Kaggle data reference:
#   Milk avg price: ₹54.93, shelf life: 1-30 days
#   Amul: 1053 rows, Mother Dairy: 1010 rows, Sudha: 648 rows

CLIENT_PROFILES = {
    1: {
        "brand": "Amul",
        "region": "Gujarat",
        # Demand/Supply
        "demand_base": 1100,        # High volume cooperative
        "demand_noise_std": 80,
        "demand_trend_per_week": 0.4,  # Steady growth
        "demand_season_amplitude": 120,  # Moderate seasonality
        "supply_surplus_ratio": 1.12,   # Usually surplus (well-organized)
        "supply_noise_std": 60,
        "production_base": 1400,
        # Pricing (from Gov data: Gujarat ₹53-54)
        "price_base": 53.0,
        "price_variation": 1.5,
        "cost_ratio": 0.58,          # Cost is ~58% of price (efficient)
        # Inventory
        "inventory_base": 300,
        "inventory_noise_std": 80,
        "min_stock_base": 60,
        "reorder_qty_base": 150,
        # Risk & Sustainability
        "disruption_base": 0.06,
        "disruption_noise_std": 0.03,
        "disruption_monsoon_boost": 0.08,
        "emission_base": 1.3,
        "emission_noise_std": 0.12,
        # Quality & Logistics
        "shelf_life_choices": [3, 5, 7],  # Modern cold chain
        "shelf_life_weights": [0.2, 0.5, 0.3],
        "storage_choices": ["Tetra Pack", "Refrigerated", "Polythene Packet"],
        "storage_weights": [0.55, 0.30, 0.15],
        # Festival boost (Navratri, Diwali in Gujarat)
        "festival_demand_boost": 0.22,
        # Summer boost
        "summer_demand_boost": 0.12,
    },
    2: {
        "brand": "Mother Dairy",
        "region": "Delhi",
        # Demand/Supply
        "demand_base": 1450,        # Very high urban demand
        "demand_noise_std": 110,
        "demand_trend_per_week": 0.6,  # Fast urban growth
        "demand_season_amplitude": 180,  # High seasonality (summer spikes)
        "supply_surplus_ratio": 0.95,   # Sometimes short (imports from Punjab/Haryana)
        "supply_noise_std": 90,
        "production_base": 600,      # Delhi produces little, imports bulk
        # Pricing (from Gov data: Delhi ₹55-56)
        "price_base": 55.0,
        "price_variation": 2.0,
        "cost_ratio": 0.62,          # Slightly higher cost (logistics)
        # Inventory
        "inventory_base": 250,
        "inventory_noise_std": 100,
        "min_stock_base": 70,
        "reorder_qty_base": 180,
        # Risk & Sustainability
        "disruption_base": 0.12,
        "disruption_noise_std": 0.05,
        "disruption_monsoon_boost": 0.15,  # Floods affect supply routes
        "emission_base": 1.8,
        "emission_noise_std": 0.18,
        # Quality & Logistics
        "shelf_life_choices": [1, 2, 3],  # Urban freshness demand
        "shelf_life_weights": [0.35, 0.45, 0.2],
        "storage_choices": ["Tetra Pack", "Refrigerated", "Polythene Packet"],
        "storage_weights": [0.70, 0.20, 0.10],
        # Festival boost (Diwali, Holi in North India)
        "festival_demand_boost": 0.28,
        # Summer boost (Delhi extreme heat)
        "summer_demand_boost": 0.20,
    },
    3: {
        "brand": "Sudha",
        "region": "Bihar",
        # Demand/Supply
        "demand_base": 600,          # Moderate volume
        "demand_noise_std": 65,
        "demand_trend_per_week": 0.2,  # Slower growth
        "demand_season_amplitude": 100,  # High seasonality (agriculture-dependent)
        "supply_surplus_ratio": 1.05,   # Slight surplus
        "supply_noise_std": 70,
        "production_base": 950,
        # Pricing (from Gov data: Bihar ₹49-51)
        "price_base": 49.0,
        "price_variation": 1.0,
        "cost_ratio": 0.55,          # Lower cost (local sourcing)
        # Inventory
        "inventory_base": 180,
        "inventory_noise_std": 60,
        "min_stock_base": 40,
        "reorder_qty_base": 100,
        # Risk & Sustainability
        "disruption_base": 0.18,
        "disruption_noise_std": 0.06,
        "disruption_monsoon_boost": 0.22,  # Bihar floods heavily
        "emission_base": 2.2,
        "emission_noise_std": 0.25,
        # Quality & Logistics
        "shelf_life_choices": [1, 2],  # Limited cold chain
        "shelf_life_weights": [0.60, 0.40],
        "storage_choices": ["Polythene Packet", "Refrigerated", "Tetra Pack"],
        "storage_weights": [0.65, 0.25, 0.10],
        # Festival boost (Chhath Puja in Bihar)
        "festival_demand_boost": 0.30,
        # Summer boost
        "summer_demand_boost": 0.10,
    }
}


def get_season_factor(week_of_year, profile):
    """
    Returns seasonal multiplier based on week of year.
    Accounts for: monsoon (Jul-Sep), festival (Oct-Nov), summer (Apr-Jun), winter (Dec-Feb).
    """
    factor = 0.0

    # Base sinusoidal seasonality
    factor += np.sin(2 * np.pi * week_of_year / 52)

    # Summer spike (weeks 14-26, April-June)
    if 14 <= week_of_year <= 26:
        factor += profile["summer_demand_boost"]

    # Monsoon disruption dip (weeks 27-39, July-September)
    if 27 <= week_of_year <= 39:
        factor -= 0.08  # Slight demand reduction during heavy rains

    # Festival season spike (weeks 40-48, October-November)
    if 40 <= week_of_year <= 48:
        factor += profile["festival_demand_boost"]

    # Winter slight dip (weeks 49-52 + 1-5)
    if week_of_year >= 49 or week_of_year <= 5:
        factor -= 0.05

    return factor


def get_disruption_factor(week_of_year, profile):
    """
    Returns disruption probability accounting for monsoon season.
    """
    base = profile["disruption_base"]
    noise = np.abs(np.random.normal(0, profile["disruption_noise_std"]))

    # Monsoon boost (July-September)
    if 27 <= week_of_year <= 39:
        base += profile["disruption_monsoon_boost"]

    # Random extreme events (1% chance per week)
    if np.random.random() < 0.01:
        base += 0.25  # Strike, flood, etc.

    return np.clip(base + noise, 0.0, 0.95)


def generate_client_csv(client_id, profile, num_weeks=500):
    """
    Generates a complete CSV for one federated learning client.
    """
    start_date = datetime(2015, 1, 5)  # Start from first Monday of 2015

    rows = []
    prev_inventory = profile["inventory_base"]

    for w in range(num_weeks):
        current_date = start_date + timedelta(weeks=w)
        week_of_year = current_date.isocalendar()[1]

        # --- Season & Disruption ---
        season = get_season_factor(week_of_year, profile)
        disruption = get_disruption_factor(week_of_year, profile)

        # --- Demand (with trend + seasonality + noise) ---
        trend = profile["demand_base"] + (w * profile["demand_trend_per_week"])
        seasonal_component = profile["demand_season_amplitude"] * season
        noise = np.random.normal(0, profile["demand_noise_std"])
        demand = max(50, int(trend + seasonal_component + noise))

        # --- Supply (correlated with demand, affected by disruption) ---
        supply_base = demand * profile["supply_surplus_ratio"]
        supply_disruption = supply_base * (1 - disruption * 0.3)  # Disruption reduces supply
        supply_noise = np.random.normal(0, profile["supply_noise_std"])
        supply = max(30, int(supply_disruption + supply_noise))

        # --- Production Volume ---
        production = max(50, int(
            profile["production_base"]
            + (w * 0.3)
            + np.random.normal(0, profile["supply_noise_std"] * 0.8)
            - (disruption * profile["production_base"] * 0.15)
        ))

        # --- Quantity Sold (can't sell more than min(demand, supply + inventory)) ---
        available = supply + prev_inventory
        quantity_sold = max(0, min(demand, available))
        # Add some realistic variance (not everything sells perfectly)
        sell_efficiency = np.random.uniform(0.88, 1.0)
        quantity_sold = int(quantity_sold * sell_efficiency)

        # --- Pricing ---
        # Seasonal price variation (higher in summer/festival, lower in flush season)
        price_seasonal = profile["price_variation"] * np.sin(2 * np.pi * week_of_year / 52)
        price_per_unit = round(profile["price_base"] + price_seasonal + np.random.normal(0, 0.5), 2)
        price_per_unit = max(20.0, price_per_unit)

        cost_per_unit = round(price_per_unit * profile["cost_ratio"] + np.random.normal(0, 0.3), 2)
        cost_per_unit = max(15.0, min(cost_per_unit, price_per_unit * 0.85))

        # --- Revenue & Profit ---
        total_revenue = round(quantity_sold * price_per_unit, 2)
        profit_margin = round((price_per_unit - cost_per_unit) / price_per_unit, 4)
        profit_margin = np.clip(profit_margin, 0.05, 0.55)

        # --- Inventory (rolling) ---
        inventory_level = max(0, int(
            prev_inventory + supply - quantity_sold
            + np.random.normal(0, profile["inventory_noise_std"] * 0.3)
        ))
        # Cap inventory to prevent unrealistic buildup
        inventory_level = min(inventory_level, profile["inventory_base"] * 4)
        prev_inventory = inventory_level

        min_stock_threshold = round(
            profile["min_stock_base"] + np.random.uniform(-5, 5), 2
        )
        reorder_quantity = round(
            profile["reorder_qty_base"] + np.random.uniform(-15, 15), 2
        )

        # --- Emission Factor ---
        emission = round(
            profile["emission_base"]
            + np.random.normal(0, profile["emission_noise_std"])
            + (disruption * 0.2),  # Disruption increases emissions (rerouting)
            3
        )
        emission = max(0.5, emission)

        # --- Quality & Logistics ---
        shelf_life = np.random.choice(
            profile["shelf_life_choices"],
            p=profile["shelf_life_weights"]
        )
        storage_condition = np.random.choice(
            profile["storage_choices"],
            p=profile["storage_weights"]
        )

        # --- Build Row ---
        rows.append({
            # Group A: Identity & Time
            "week": w,
            "date": current_date.strftime("%Y-%m-%d"),
            "client_id": client_id,
            "brand": profile["brand"],
            "region": profile["region"],
            # Group B: Demand & Supply
            "demand": demand,
            "supply": supply,
            "quantity_sold": quantity_sold,
            "production_volume": production,
            # Group C: Pricing & Financials
            "price_per_unit": price_per_unit,
            "cost_per_unit": cost_per_unit,
            "total_revenue": total_revenue,
            "profit_margin": profit_margin,
            # Group D: Inventory
            "inventory_level": inventory_level,
            "min_stock_threshold": min_stock_threshold,
            "reorder_quantity": reorder_quantity,
            # Group E: Risk & Sustainability
            "disruption_prob": round(disruption, 4),
            "emission_factor": emission,
            # Group F: Quality & Logistics
            "shelf_life_days": shelf_life,
            "storage_condition": storage_condition,
        })

    return pd.DataFrame(rows)


def validate_dataframe(df, client_name):
    """Validates the generated dataframe."""
    errors = []

    if len(df) != 500:
        errors.append(f"Expected 500 rows, got {len(df)}")
    if len(df.columns) != 20:
        errors.append(f"Expected 20 columns, got {len(df.columns)}")
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        errors.append(f"Found NaN values in columns: {null_cols}")
    if (df["demand"] <= 0).any():
        errors.append("Found non-positive demand values")
    if (df["supply"] <= 0).any():
        errors.append("Found non-positive supply values")
    if (df["price_per_unit"] <= 0).any():
        errors.append("Found non-positive prices")
    if (df["disruption_prob"] < 0).any() or (df["disruption_prob"] > 1).any():
        errors.append("disruption_prob out of [0, 1] range")
    if (df["emission_factor"] <= 0).any():
        errors.append("Found non-positive emission factors")
    if (df["profit_margin"] < 0).any() or (df["profit_margin"] > 1).any():
        errors.append("profit_margin out of [0, 1] range")

    if errors:
        print(f"  ❌ {client_name} VALIDATION FAILED:")
        for e in errors:
            print(f"     - {e}")
        return False
    else:
        print(f"  ✅ {client_name} passed all validation checks")
        return True


def print_summary(df, name):
    """Prints a summary of key statistics for a client."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Date range: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    print(f"  Demand:  mean={df['demand'].mean():.0f}, min={df['demand'].min()}, max={df['demand'].max()}")
    print(f"  Supply:  mean={df['supply'].mean():.0f}, min={df['supply'].min()}, max={df['supply'].max()}")
    print(f"  Price:   mean=₹{df['price_per_unit'].mean():.2f}, range=[₹{df['price_per_unit'].min():.2f}, ₹{df['price_per_unit'].max():.2f}]")
    print(f"  Cost:    mean=₹{df['cost_per_unit'].mean():.2f}")
    print(f"  Revenue: mean=₹{df['total_revenue'].mean():.0f}")
    print(f"  Margin:  mean={df['profit_margin'].mean():.2%}")
    print(f"  Inventory: mean={df['inventory_level'].mean():.0f}")
    print(f"  Disruption: mean={df['disruption_prob'].mean():.3f}")
    print(f"  Emission: mean={df['emission_factor'].mean():.3f}")
    print(f"  Shelf Life: {dict(df['shelf_life_days'].value_counts())}")
    print(f"  Storage: {dict(df['storage_condition'].value_counts())}")


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "DATASETS")
    os.makedirs(output_dir, exist_ok=True)

    filenames = {
        1: "client_1_amul_gujarat.csv",
        2: "client_2_mother_dairy_delhi.csv",
        3: "client_3_sudha_bihar.csv",
    }

    print("🥛 Federated Learning - Milk Supply Chain Dataset Generator")
    print("=" * 60)

    all_valid = True

    for client_id, profile in CLIENT_PROFILES.items():
        print(f"\n⏳ Generating Client {client_id}: {profile['brand']} ({profile['region']})...")
        df = generate_client_csv(client_id, profile)

        # Validate
        valid = validate_dataframe(df, f"Client {client_id} ({profile['brand']})")
        if not valid:
            all_valid = False

        # Print summary
        print_summary(df, f"Client {client_id}: {profile['brand']} ({profile['region']})")

        # Save
        filepath = os.path.join(output_dir, filenames[client_id])
        df.to_csv(filepath, index=False)
        print(f"  💾 Saved: {filepath}")

    print(f"\n{'='*60}")
    if all_valid:
        print("✅ ALL 3 CSVs GENERATED SUCCESSFULLY!")
    else:
        print("⚠️  Some validations failed — check above for details.")

    # Cross-client comparison
    print(f"\n{'='*60}")
    print("📊 NON-IID COMPARISON (Proving data heterogeneity)")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Amul (GJ)':<18} {'Mother Dairy (DL)':<20} {'Sudha (BR)':<18}")
    print("-" * 81)

    dfs = {}
    for cid, fname in filenames.items():
        dfs[cid] = pd.read_csv(os.path.join(output_dir, fname))

    for metric, fmt in [
        ("demand", ".0f"), ("supply", ".0f"), ("price_per_unit", ".2f"),
        ("cost_per_unit", ".2f"), ("disruption_prob", ".4f"),
        ("emission_factor", ".3f"), ("inventory_level", ".0f"),
        ("profit_margin", ".4f"),
    ]:
        vals = [f"{dfs[cid][metric].mean():{fmt}}" for cid in [1, 2, 3]]
        print(f"  {metric:<23} {vals[0]:<18} {vals[1]:<20} {vals[2]:<18}")


if __name__ == "__main__":
    main()
