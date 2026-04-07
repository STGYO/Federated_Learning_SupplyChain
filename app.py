import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*")

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import json
import time
import matplotlib.pyplot as plt
import matplotlib

# Import logic from main.py
from main import (
    load_model, 
    SCConfig, 
    SupplyChainDataManager, 
    FedSim,
    LSTMModel, 
    optimize, 
    llm_generate, 
    to_serializable,
    get_device
)


st.set_page_config(
    page_title="FedSim · Supply Chain 5.0",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Global Dark Theme ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #141e30 50%, #0d1117 100%);
        color: #e2e8f0;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] span {
        color: #c9d1d9 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #a78bfa !important;
    }

    /* ── Headers ── */
    h1 { color: #e2e8f0 !important; font-weight: 800 !important; letter-spacing: -0.5px; }
    h2, h3 { color: #c9d1d9 !important; font-weight: 700 !important; }
    .stMarkdown p, .stMarkdown li, .stCaption, label, span { color: #a8b2c1 !important; }

    /* ── Animated Gradient Border Top ── */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #a78bfa, #06b6d4, #10b981, #6366f1);
        background-size: 300% 100%;
        animation: gradient-slide 4s linear infinite;
        z-index: 999;
    }
    @keyframes gradient-slide {
        0% { background-position: 0% 50%; }
        100% { background-position: 300% 50%; }
    }

    /* ── Glassmorphism Cards ── */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.15);
        transform: translateY(-2px);
    }
    div[data-testid="stMetric"] label { color: #8b95a5 !important; font-size: 13px !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-weight: 700 !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5) !important;
    }
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
    }
    .stFormSubmitButton > button:hover {
        box-shadow: 0 6px 25px rgba(16, 185, 129, 0.5) !important;
    }

    /* ── Inputs ── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stTextArea > div > div > textarea {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 8px !important;
    }
    
    /* ── Expanders / Forms ── */
    .stForm {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 10px;
    }
    
    /* ── Info / Warning / Error Boxes ── */
    .stAlert {
        border-radius: 10px !important;
    }
    div[data-testid="stNotification"] {
        border-radius: 10px;
    }

    /* ── Chat Messages ── */
    .stChatMessage {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(99, 102, 241, 0.1) !important;
        border-radius: 12px !important;
    }

    /* ── Dividers ── */
    hr { border-color: rgba(99, 102, 241, 0.15) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 10px; }

    /* ── Custom Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(6,182,212,0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 14px;
        padding: 18px 28px;
        margin-bottom: 24px;
    }
    .hero-banner h4 { margin: 0; color: #a78bfa !important; font-size: 14px; letter-spacing: 2px; text-transform: uppercase; }
    .hero-banner .stats { display: flex; gap: 30px; margin-top: 8px; }
    .hero-banner .stat-item { color: #8b95a5; font-size: 13px; }
    .hero-banner .stat-item strong { color: #06b6d4; }

    /* ── Pulse Dot ── */
    .pulse-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #10b981;
        animation: pulse 1.5s ease infinite;
        margin-right: 6px;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
        50% { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
    }

    /* ── Progress Bar ── */
    .stProgress > div > div > div { background-color: #6366f1 !important; }
    
    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 8px;
        color: #8b95a5 !important;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.15) !important;
        border-color: #6366f1 !important;
        color: #a78bfa !important;
    }
</style>
""", unsafe_allow_html=True)



matplotlib.rcParams.update({
    'figure.facecolor': '#0d111700',
    'axes.facecolor': '#161b2200',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#8b95a5',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b95a5',
    'ytick.color': '#8b95a5',
    'grid.color': '#21262d',
    'grid.alpha': 0.5,
})

def plot_financial_pie(financials):
    sizes = [
        max(0, financials['net_profit']), 
        financials['order_cost'], 
        financials['waste_cost']
    ]
    labels = ['Net Profit', 'Order Cost', 'Waste Risk']
    colors = ['#10b981', '#f59e0b', '#ef4444']
    explode = (0.06, 0, 0)

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=False, startangle=90,
        textprops={'color': '#c9d1d9', 'weight': 'bold', 'size': 10}
    )
    for t in texts:
        t.set_color('#8b95a5')
    ax.axis('equal')
    return fig

def plot_training_curves(metrics):
    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_alpha(0)
    ax1.set_facecolor('none')
    
    rounds = metrics["rounds"]
    ax1.plot(rounds, metrics["loss"], color='#6366f1', linewidth=2, marker='o', markersize=5, label='Loss')
    ax1.fill_between(rounds, metrics["loss"], alpha=0.1, color='#6366f1')
    ax1.set_xlabel("Round", fontsize=11)
    ax1.set_ylabel("Loss (MSE)", color='#6366f1', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#6366f1')
    ax1.grid(True, alpha=0.2)
    
    ax2 = ax1.twinx()
    ax2.plot(rounds, metrics["mae"], color='#06b6d4', linewidth=2, marker='s', markersize=5, label='MAE')
    ax2.set_ylabel("MAE", color='#06b6d4', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#06b6d4')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', 
               facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    
    fig.tight_layout()
    return fig

def plot_demand_history(history_df, forecast):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    
    weeks = history_df["week"].values
    demand = history_df["demand"].values
    
    ax.plot(weeks, demand, color='#a78bfa', linewidth=2, marker='o', markersize=4, label='Actual Demand')
    ax.fill_between(weeks, demand, alpha=0.08, color='#a78bfa')
    
    # Forecast point
    next_week = weeks[-1] + 1
    ax.scatter([next_week], [forecast], color='#f59e0b', s=80, zorder=5, marker='D', label=f'Forecast: {forecast}')
    ax.axvline(x=next_week - 0.5, color='#f59e0b', linestyle='--', alpha=0.3)
    
    ax.set_xlabel("Week", fontsize=11)
    ax.set_ylabel("Demand (Units)", fontsize=11)
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


with st.sidebar:
    st.markdown("## 🔮 FedSim")
    st.caption("Supply Chain 5.0 · Privacy-Preserving AI")
    st.markdown("---")
    
    st.markdown("#### ⚙️ Configuration")
    st.info(f"**Product:** {SCConfig.PRODUCT_NAME}")
    
    num_clients = st.slider("Network Clients", 1, 10, SCConfig.NUM_CLIENTS)
    num_rounds = st.slider("FL Rounds", 1, 10, SCConfig.NUM_ROUNDS)
    carbon_cap = st.number_input("Carbon Cap (CO₂e)", value=SCConfig.CARBON_CAP)
    
    st.markdown("#### 🛡️ Privacy & Security")
    dp_epsilon = st.slider("Privacy Budget (ε)", 0.1, 20.0, SCConfig.DP_EPSILON, help="Lower ε = Stronger Privacy")
    clip_norm = st.slider("Gradient Clipping", 1.0, 10.0, SCConfig.CLIP_NORM, help="Bound sensitivity before adding noise.")
    poison_thresh = st.slider("Poisoning Threshold", 2.0, 20.0, SCConfig.POISON_THRESHOLD, help="Reject clients with loss N times higher than median.")
    
    log_dir = st.text_input("Log Directory", SCConfig.LOG_DIR)
    
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; padding: 8px; opacity: 0.5; font-size: 11px;">
        <span class="pulse-dot"></span> System Active<br>
        v1.0.0 · Federated Learning
    </div>
    """, unsafe_allow_html=True)

# Update Config
SCConfig.NUM_CLIENTS = num_clients
SCConfig.NUM_ROUNDS = num_rounds
SCConfig.CARBON_CAP = carbon_cap
SCConfig.LOG_DIR = log_dir
SCConfig.DP_EPSILON = dp_epsilon
SCConfig.CLIP_NORM = clip_norm
SCConfig.POISON_THRESHOLD = poison_thresh

# Initialize Session State
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "simulation_done" not in st.session_state:
    st.session_state.simulation_done = False
if "opt_result" not in st.session_state:
    st.session_state.opt_result = None
if "forecast" not in st.session_state:
    st.session_state.forecast = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_manager" not in st.session_state:
    st.session_state.data_manager = None
if "fed_sim" not in st.session_state:
    st.session_state.fed_sim = None
if "inventory" not in st.session_state:
    st.session_state.inventory = 50.0


st.markdown(f"""
<div class="hero-banner">
    <h4>🌐 Federated Supply Chain Optimization</h4>
    <div class="stats">
        <div class="stat-item">Device: <strong>{get_device()}</strong></div>
        <div class="stat-item">Model: <strong>{SCConfig.MODEL_NAME}</strong></div>
        <div class="stat-item">Objective: <strong>Profit + Sustainability</strong></div>
        <div class="stat-item">Privacy: <strong>ε = {SCConfig.DP_EPSILON}</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)


main_col, chat_col = st.columns([7, 3])

with main_col:
    # ── Model Loading ──
    if not st.session_state.model:
        st.markdown("### 🚀 Initialize AI Engine")
        st.markdown("Load the fine-tuned Qwen model and launch the federated core.")
        if st.button("⚡ Boot System"):
            with st.spinner("Loading Qwen 0.5B & Federated Core..."):
                try:
                    tokenizer, model = load_model()
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model = model
                    st.success("✅ System Online!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Initialization Failed: {e}")
    else:
        # ── Tabs ──
        tab1, tab2, tab3 = st.tabs(["⚡ Simulation", "📊 Analytics", "📋 Decision"])


        with tab1:
            st.markdown("### Federated Simulation")
            st.markdown("Run a privacy-preserving training round across distributed supply chain nodes.")
            
            if st.button("▶ Run Simulation Round"):
                st.session_state.run_simulation = True
                
            if st.session_state.get("run_simulation", False):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 1. Initialize Data
                if st.session_state.data_manager is None:
                    status_text.text("⏳ Generating distributed synthetic data...")
                    data_manager = SupplyChainDataManager(SCConfig.NUM_CLIENTS)
                    st.session_state.data_manager = data_manager
                else:
                    status_text.text("⏳ Advancing dataset by 1 week...")
                    data_manager = st.session_state.data_manager
                    data_manager.advance_one_week()
                    
                    # Process pending inventory updates
                    if "pending_order" in st.session_state:
                        new_actual_demand = data_manager.get_client_data("0")["demand"].iloc[-1]
                        st.session_state.inventory = max(0.0, st.session_state.inventory + st.session_state.pending_order - new_actual_demand)
                        del st.session_state.pending_order
                        
                progress_bar.progress(20)
                
                # 2. Run FedSim
                status_text.text(f"⏳ Training across {SCConfig.NUM_CLIENTS} clients ({SCConfig.NUM_ROUNDS} rounds)...")
                if st.session_state.fed_sim is None:
                    st.session_state.fed_sim = FedSim(data_manager)
                    
                fed = st.session_state.fed_sim
                lstm_model, metrics = fed.run(st.session_state.tokenizer, st.session_state.model, epsilon=SCConfig.DP_EPSILON)
                progress_bar.progress(80)
                
                # 3. Optimization using LSTM Forecast
                status_text.text("⏳ Running Multi-Objective Optimization...")
                client0_df = data_manager.get_client_data("0")
                data = client0_df["demand"].values.astype(np.float32)
                max_val = 300.0
                last_seq = data[-5:] / max_val
                inp = torch.tensor(last_seq).unsqueeze(0).unsqueeze(-1)
                
                lstm_model.eval()
                with torch.no_grad():
                    pred_norm = lstm_model(inp).item()
                
                forecast = int(pred_norm * max_val)
                client0 = client0_df.iloc[-1]
                
                opt = optimize(
                    forecast=forecast,
                    inventory=st.session_state.inventory,
                    emission_factor=float(client0["emission_factor"]),
                    risk=float(client0["disruption_prob"])
                )
                
                st.session_state.forecast = forecast
                st.session_state.opt_result = opt
                st.session_state.metrics = metrics
                st.session_state.simulation_done = True
                
                progress_bar.progress(100)
                status_text.text("✅ Simulation Complete!")
                st.session_state.run_simulation = False


        with tab2:
            if st.session_state.simulation_done and st.session_state.opt_result:
                opt = st.session_state.opt_result
                forecast = st.session_state.forecast
                metrics = st.session_state.metrics
                fin = opt['financials']
                
                # ── Top Metrics Row ──
                m0, m1, m2, m3, m4 = st.columns(5)
                m0.metric("Current Inv.", f"{st.session_state.inventory:.0f}")
                m1.metric("Forecast", f"{forecast}")
                m2.metric("Order Qty", f"{opt['optimized_qty']}")
                m3.metric("Emissions", f"{opt['emissions']:.2f} CO₂e")
                m4.metric("Feasible?", "✅ Yes" if opt['feasible'] else "❌ No")
                
                st.markdown("")
                
                # ── Training Curves ──
                st.markdown("#### 📈 Model Convergence")
                st.pyplot(plot_training_curves(metrics), use_container_width=True)
                
                st.markdown("")
                
                # ── Demand History ──
                st.markdown("#### 📉 Demand History & Forecast")
                client0_df = st.session_state.data_manager.get_client_data("0")
                history_df = client0_df.tail(20).copy()
                st.pyplot(plot_demand_history(history_df, forecast), use_container_width=True)
                
                st.markdown("")
                
                # ── Financials ──
                st.markdown("#### 💰 Financial Projection")
                f_left, f_right = st.columns([2, 3])
                
                with f_left:
                    st.metric("Revenue", f"${fin['revenue']:.2f}")
                    st.metric("Order Cost", f"${fin['order_cost']:.2f}")
                    st.metric("Waste Cost", f"${fin['waste_cost']:.2f}")
                    st.metric("Net Profit", f"${fin['net_profit']:.2f}")
                
                with f_right:
                    st.pyplot(plot_financial_pie(fin), use_container_width=True)
                    
            else:
                st.info("💡 Run a simulation round in the **Simulation** tab to see analytics here.")


        with tab3:
            if st.session_state.simulation_done and st.session_state.opt_result:
                opt = st.session_state.opt_result
                forecast = st.session_state.forecast
                
                # AI Insight
                st.markdown("#### 🤖 AI Strategic Insight")
                if st.button("✨ Generate Explanation"):
                    with st.spinner("Consulting Fine-Tuned AI..."):
                        system_msg = {
                            "role": "system", 
                            "content": f"You are a supply chain expert. Product: {SCConfig.PRODUCT_NAME}. Forecast: {forecast}. Emissions: {opt['emissions']:.2f}."
                        }
                        user_msg = {
                            "role": "user", 
                            "content": f"The recommended order quantity is {opt['optimized_qty']}. Provide a brief strategic recommendation."
                        }
                        
                        insight = llm_generate(
                            [system_msg, user_msg],
                            st.session_state.tokenizer,
                            st.session_state.model,
                            max_tokens=150
                        )
                        st.markdown(f"""
                        <div style="background: rgba(99,102,241,0.1); border-left: 4px solid #6366f1; padding: 16px; border-radius: 8px; color: #c9d1d9;">
                            <strong style="color: #a78bfa;">AI Analysis:</strong><br><br>{insight}
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("")
                st.markdown("---")
                
                # Approval Form
                st.markdown("#### 📝 Managerial Override & Approval")
                with st.form("override_form"):
                    new_qty = st.number_input("Final Order Quantity", value=int(opt['optimized_qty']))
                    submitted = st.form_submit_button("✅ Approve & Run Next Round")
                    
                    if submitted:
                        if new_qty != opt['optimized_qty']:
                            log_entry = {"event": "override", "new": int(new_qty), "original": opt['optimized_qty'], "product": SCConfig.PRODUCT_NAME}
                            st.warning(f"⚠️ Order quantity overridden to {new_qty}")
                        else:
                            log_entry = {"event": "approved", "qty": opt['optimized_qty'], "product": SCConfig.PRODUCT_NAME}
                            st.success("✅ AI Recommendation Approved")
                            
                        # Set pending order for the next round
                        st.session_state.pending_order = int(new_qty)
                        
                        os.makedirs(SCConfig.LOG_DIR, exist_ok=True)
                        with open(os.path.join(SCConfig.LOG_DIR, "decision_log.json"), "a") as f:
                            f.write(json.dumps(to_serializable(log_entry)) + "\n")
                        st.toast("Decision saved! Starting next round...")
                        
                        time.sleep(1)
                        st.session_state.run_simulation = True
                        st.rerun()
            else:
                st.info("💡 No active decision pending. Run a simulation first.")



with chat_col:
    st.markdown("### 💬 AI Assistant")
    st.caption(f"Context: {SCConfig.PRODUCT_NAME}")
    
    chat_container = st.container(height=550)
    
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center; padding: 40px 10px; opacity: 0.5;">
                <div style="font-size: 40px; margin-bottom: 10px;">🤖</div>
                <div style="font-size: 13px; color: #8b95a5;">Ask about inventory, sustainability, or the latest forecast.</div>
            </div>
            """, unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
    if prompt := st.chat_input(f"Ask about {SCConfig.PRODUCT_NAME}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        context_str = ""
        if st.session_state.simulation_done and st.session_state.opt_result:
            opt = st.session_state.opt_result
            forecast = st.session_state.forecast
            context_str = (
                f"Context: Managing supply chain for '{SCConfig.PRODUCT_NAME}'. "
                f"Forecast: {forecast} units. "
                f"Recommended Order: {opt['optimized_qty']} units. "
                f"Emissions: {opt['emissions']:.2f}. "
            )
        
        messages = [
            {"role": "system", "content": f"You are a helpful Supply Chain Assistant. {context_str}"},
            {"role": "user", "content": prompt}
        ]
        
        if st.session_state.model:
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = llm_generate(
                            messages, 
                            st.session_state.tokenizer,
                            st.session_state.model,
                            max_tokens=200
                        )
                        st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("⚠️ Please initialize the AI system first.")
