"""
Supply Chain 5.0
Local VS Code Version
Real AI using TinyLlama 1.1B (4-bit)
Federated Simulation + Optimization + Human Override + AI Impact
"""

import os
import gc
import json
import re
import time
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig



@dataclass
class SCConfig:
    MODEL_NAME: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    PRODUCT_NAME: str = "Milk"  # Fixed Product
    NUM_CLIENTS: int = 3
    NUM_ROUNDS: int = 2
    CARBON_CAP: float = 500.0  # Adjusted for Milk (e.g. per batch)
    LOG_DIR: str = "sc50_logs"
    DP_EPSILON: float = 5.0  # Privacy Budget (Lower = More Privacy/Noise)
    
    # Financials (Per Unit)
    SELLING_PRICE: float = 4.0
    COST_PRICE: float = 1.5
    WASTE_COST: float = 0.5  # Cost of disposal/spoilage
    
    # Security Features
    CLIP_NORM: float = 5.0 # Gradient clipping threshold
    POISON_THRESHOLD: float = 10.0 # Multiplier over avg loss to consider it poisoned


# Create logs folder if missing
os.makedirs(SCConfig.LOG_DIR, exist_ok=True)



def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    return obj



def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model():
    device = get_device()
    log(f"Loading TinyLlama 1.1B on {device}...")

    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            SCConfig.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        # CPU or MPS (Apple Silicon) - 4-bit quantization usually requires CUDA
        # We load in float32 (default) or float16 if supported to save memory
        torch_dtype = torch.float32 
        if device == "mps":
             torch_dtype = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            SCConfig.MODEL_NAME,
            device_map=device,
            torch_dtype=torch_dtype
        )

    tokenizer = AutoTokenizer.from_pretrained(SCConfig.MODEL_NAME)
    
    log("Model Loaded Successfully.")
    return tokenizer, model


def llm_generate(prompt, tokenizer, model, max_tokens=200, temperature=0.7):
    # Support both raw string prompts and chat messages list
    if isinstance(prompt, str):
        messages = [
            {"role": "system", "content": f"You are a helpful Supply Chain Assistant optimized for {SCConfig.PRODUCT_NAME}."},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = prompt

    # Apply Chat Template (handles <|system|>, <|user|>, etc. for TinyLlama)
    input_ids = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,      
            do_sample=True,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1 # Prevent repetition
        )

    # Decode only the new tokens (response)
    # outputs contains [input_ids + new_tokens]
    response_ids = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)



class DifferentialPrivacy:
    @staticmethod
    def add_noise(tensor_group: list[torch.Tensor], epsilon: float, sensitivity: float = 1.0):
        """Adds Laplacian noise across an entire group of aggregated tensors for Secure Aggregation."""
        if epsilon <= 0: return tensor_group
        beta = sensitivity / epsilon
        return [t + torch.tensor(np.random.laplace(0, beta, t.shape)).float() for t in tensor_group]

    @staticmethod
    def clip_gradients(tensor: torch.Tensor, clip_norm: float = SCConfig.CLIP_NORM) -> torch.Tensor:
        """Clips the tensor (e.g., weights) to bound sensitivity before aggregation."""
        norm = torch.norm(tensor)
        if norm > clip_norm:
            return tensor * (clip_norm / (norm + 1e-6))
        return tensor

class SecureTransmission:
    @staticmethod
    def generate_masks(trusted_indices: list, model_template: dict) -> dict:
        """
        Simulates the cryptographic key-exchange phase of Secure Aggregation.
        Generates random large tensors (masks) for each trusted client such that their sum across the network is exactly zero.
        This ensures the central server cannot read individual local weights, only the aggregated sum.
        """
        masks = {cid: {k: torch.zeros_like(v) for k, v in model_template.items()} for cid in trusted_indices}
        
        # Pairwise Masking Simulation (analogous to Diffie-Hellman Key Exchange in SecAgg)
        for i in range(len(trusted_indices)):
            for j in range(i + 1, len(trusted_indices)):
                cid_i = trusted_indices[i]
                cid_j = trusted_indices[j]
                
                for k in model_template.keys():
                    # Generate a random cryptographic mask for the client pair
                    pair_mask = torch.randn_like(model_template[k]) * 1000.0 # Extremely large random noise to hide real weights
                    
                    # Client i adds the mask
                    masks[cid_i][k] += pair_mask
                    # Client j subtracts the mask
                    masks[cid_j][k] -= pair_mask
                    
        return masks



class SupplyChainDataManager:
    def __init__(self, num_clients: int, weeks: int = 52):
        self.num_clients = num_clients
        self.weeks = weeks
        self.client_data = {}
        self.firm_shifts = {}
        self.generate_data()

    def generate_data(self):
        np.random.seed(42)
        for cid in range(self.num_clients):
            t = np.arange(self.weeks)
            trend = 100 + (t * 0.5)
            seasonality = 20 * np.sin(2 * np.pi * t / 12)
            noise = np.random.normal(0, 5, self.weeks)
            firm_shift = np.random.randint(-10, 20)
            self.firm_shifts[str(cid)] = firm_shift

            demand = trend + seasonality + noise + firm_shift
            disruption_prob = np.clip(np.random.beta(2, 10, self.weeks), 0, 1)
            emission_factor = np.full(self.weeks, 1.5 + (np.random.rand() * 0.5))

            self.client_data[str(cid)] = pd.DataFrame({
                "week": t,
                "demand": demand.astype(int),
                "disruption_prob": disruption_prob,
                "emission_factor": emission_factor
            })

    def get_client_data(self, cid: str):
        return self.client_data[str(cid)]

    def advance_one_week(self):
        self.weeks += 1
        for cid in range(self.num_clients):
            df = self.client_data[str(cid)]
            next_week = int(df['week'].max() + 1)
            
            trend = 100 + (next_week * 0.5)
            seasonality = 20 * np.sin(2 * np.pi * next_week / 12)
            noise = np.random.normal(0, 5)
            firm_shift = self.firm_shifts.get(str(cid), 0)
            
            demand = int(trend + seasonality + noise + firm_shift)
            disruption_prob = np.clip(np.random.beta(2, 10), 0, 1)
            emission_factor = 1.5 + (np.random.rand() * 0.5)
            
            new_row = pd.DataFrame({
                "week": [next_week],
                "demand": [demand],
                "disruption_prob": [disruption_prob],
                "emission_factor": [emission_factor]
            })
            self.client_data[str(cid)] = pd.concat([df, new_row], ignore_index=True)



class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def secure_federated_average(models_state_dict, epsilon=SCConfig.DP_EPSILON, sensitivity=SCConfig.CLIP_NORM):
    """Secure Aggregation: Averages weights and adds global noise instead of per-client noise."""
    if not models_state_dict:
        return {}
        
    global_dict = models_state_dict[0].copy()
    keys = list(global_dict.keys())
    
    for k in keys:
        # Sum up all clipped tensors
        for i in range(1, len(models_state_dict)):
            global_dict[k] += models_state_dict[i][k]
        
        # Average
        global_dict[k] = torch.div(global_dict[k], len(models_state_dict))
        
        # Add DP noise at the aggregation level (Secure Aggregation property)
        if epsilon > 0:
            beta = sensitivity / epsilon
            noise = torch.tensor(np.random.laplace(0, beta, global_dict[k].shape)).float()
            global_dict[k] += noise
            
    return global_dict



class FedSim:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.input_size = 1
        self.sequence_length = 5
        self.max_val = 300.0  # Approximate max demand
        # Initialize Global Model
        self.global_model = LSTMModel(input_size=self.input_size)
        self.metrics = {"rounds": [], "mae": [], "rmse": [], "loss": []}
        self.total_rounds_completed = 0

    def train_client(self, cid, global_weights, epochs=5, lr=0.01):
        """Trains a local model on client data."""
        # Load local model with global weights
        local_model = LSTMModel(input_size=self.input_size)
        local_model.load_state_dict(global_weights)
        local_model.train()
        
        optimizer = optim.Adam(local_model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Prepare Data
        df = self.data_manager.get_client_data(str(cid))
        data = df["demand"].values.astype(np.float32)
        
        # Normalize Data (Simple MinMax for stability, ideally learned globally but approximating here)
        data_norm = data / self.max_val

        # Create Sequences
        X, y = [], []
        for i in range(len(data_norm) - self.sequence_length):
            X.append(data_norm[i:i+self.sequence_length])
            y.append(data_norm[i+self.sequence_length])
            
        # Convert lists to numpy arrays first to avoid UserWarning
        X = np.array(X)
        y = np.array(y)
        
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) # (Batch, Seq, Feature)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1) # (Batch, 1)
        
        # Local Training Loop
        epoch_loss = 0
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = local_model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        return local_model.state_dict(), epoch_loss / epochs

    def run(self, tokenizer=None, model=None, epsilon=SCConfig.DP_EPSILON):
        # NOTE: tokenizer/model args kept for compatibility but not used for LSTM training
        log("Starting Federated LSTM Simulation")
        
        for r in range(SCConfig.NUM_ROUNDS):
            current_round = self.total_rounds_completed + 1
            log(f"--- Round {current_round} ---")
            local_weights = []
            client_losses = []
            round_loss = 0
            
            # Broadcast Global Weights
            global_weights = self.global_model.state_dict()
            
            for cid in range(SCConfig.NUM_CLIENTS):
                # Train Client
                w, loss = self.train_client(cid, global_weights)
                # Client-Side Defenses: Gradient Clipping 
                for k in w.keys():
                    w[k] = DifferentialPrivacy.clip_gradients(w[k], SCConfig.CLIP_NORM)
                
                local_weights.append(w)
                round_loss += loss
                client_losses.append(loss)
                
            # Poisoning Protection (Anomaly Detection)
            # Exclude clients whose loss is extremely different from the median
            median_loss = np.median(client_losses)
            trusted_indices = []
            trusted_weights = []
            for i, c_loss in enumerate(client_losses):
                # If loss is N times higher than median, it's considered poisoned/anomalous
                if c_loss <= median_loss * SCConfig.POISON_THRESHOLD:
                    trusted_indices.append(i)
                    trusted_weights.append(local_weights[i])
                else:
                    log(f"[WARNING] Poisoning Detected: Client {i} rejected (Loss: {c_loss:.4f} vs Median: {median_loss:.4f})")
            
            # --- TRANSMISSION SECURITY (Cryptographic Masking) ---
            if trusted_indices:
                # 1. Clients generate masking keys via Secure Transmission protocol
                # (In a real system, server never sees 'masks' dictionary, only the clients do via peer-to-peer DHKE)
                masks = SecureTransmission.generate_masks(trusted_indices, global_weights)
                
                encrypted_weights = []
                # 2. Clients encrypt their weights with their local masks BEFORE sending to the server
                for idx, cid in enumerate(trusted_indices):
                    w_encrypted = {k: trusted_weights[idx][k] + masks[cid][k] for k in global_weights.keys()}
                    encrypted_weights.append(w_encrypted)
                    
                log(f"[SECURE] {len(trusted_indices)} clients sent encrypted and masked weights to server.")
                
                # Secure Aggregation & DP (Global)
                # 3. Server receives only the strongly-masked encrypted_weights. 
                # Mathematical property: Sum of masks = 0, so the average reveals the true aggregated update.
                new_global_weights = secure_federated_average(encrypted_weights, epsilon, SCConfig.CLIP_NORM)
                self.global_model.load_state_dict(new_global_weights)
            else:
                log("[WARNING] All clients rejected this round!")
            
            # Validation (Metrics on all clients)
            # Use the new global model to predict last known data point
            total_mae = 0
            total_rmse = 0
            
            self.global_model.eval()
            with torch.no_grad():
                for cid in range(SCConfig.NUM_CLIENTS):
                    df = self.data_manager.get_client_data(str(cid))
                    data = df["demand"].values.astype(np.float32)
                    
                    # Predict last week using previous sequence
                    last_seq = data[-self.sequence_length-1:-1] / self.max_val
                    true_val = data[-1]
                    
                    inp = torch.tensor(last_seq).unsqueeze(0).unsqueeze(-1)
                    pred_norm = self.global_model(inp).item()
                    pred = int(pred_norm * self.max_val)
                    
                    err = abs(true_val - pred)
                    total_mae += err
                    total_rmse += err**2
                    
            avg_loss = round_loss / SCConfig.NUM_CLIENTS
            mae = total_mae / SCConfig.NUM_CLIENTS
            rmse = np.sqrt(total_rmse / SCConfig.NUM_CLIENTS)
            
            self.metrics["rounds"].append(current_round)
            self.metrics["mae"].append(mae)
            self.metrics["rmse"].append(rmse)
            self.metrics["loss"].append(avg_loss)
            self.total_rounds_completed += 1

            log(f"Round {current_round} | Loss: {avg_loss:.4f} | MAE: {mae:.2f}")

        return self.global_model, self.metrics



def optimize(forecast, inventory, emission_factor, risk):
    # Safety Stock includes risk buffer
    safety_stock = int(forecast * (0.1 + risk))
    
    # Order Qty logic
    qty = max(0, forecast + safety_stock - inventory)

    # Emissions
    emissions = float(qty * emission_factor)
    feasible = emissions <= SCConfig.CARBON_CAP
    
    # Financials (Projected)
    # Scenario: We sell everything we forecast (up to available stock)
    # Available for sale = Inventory + Qty
    available_stock = inventory + qty
    projected_sales = min(forecast, available_stock)
    unsold_stock = max(0, available_stock - projected_sales)
    
    revenue = projected_sales * SCConfig.SELLING_PRICE
    cost = qty * SCConfig.COST_PRICE # Cost of new order
    # Note: Logic for "Profit" usually includes Cost of Goods Sold (COGS). 
    # Here we simplify: Project Cost = Cost of New Order + Holding/Waste of Unsold.
    
    # Assuming unsold milk spoils (Waste Cost)
    waste_cost = unsold_stock * SCConfig.WASTE_COST
    
    net_profit = revenue - cost - waste_cost

    return {
        "optimized_qty": qty,
        "emissions": emissions,
        "feasible": feasible,
        "safety_stock": safety_stock,
        "financials": {
            "revenue": revenue,
            "order_cost": cost,
            "waste_cost": waste_cost,
            "net_profit": net_profit
        }
    }



def main():
    device = get_device()
    log(f"Running on {device}")

    # Load LLM for Explanation only
    tokenizer, model = load_model()

    data_manager = SupplyChainDataManager(SCConfig.NUM_CLIENTS)

    # Federated LSTM Training
    fed = FedSim(data_manager)
    lstm_model, metrics = fed.run(tokenizer, model)

    # FINAL FORECAST (Using trained LSTM)
    client0_df = data_manager.get_client_data("0")
    data = client0_df["demand"].values.astype(np.float32)
    max_val = 300.0 # Same normalization constant as in training
    
    # Get last 5 weeks
    last_seq = data[-5:] / max_val
    inp = torch.tensor(last_seq).unsqueeze(0).unsqueeze(-1)
    
    lstm_model.eval()
    with torch.no_grad():
        pred_norm = lstm_model(inp).item()
        
    forecast = int(pred_norm * max_val)
    
    # Get last known emission/risk factors
    last_week_data = client0_df.iloc[-1]
    
    log(f"Final LSTM Forecast: {forecast}")

    opt = optimize(
        forecast=forecast,
        inventory=50,
        emission_factor=float(last_week_data["emission_factor"]),
        risk=float(last_week_data["disruption_prob"])
    )

    print("\nAI Recommendation (Explanation):\n")
    print(llm_generate(
        f"Forecast: {forecast}, Order Qty: {opt['optimized_qty']}, Emissions: {opt['emissions']}. Provide recommendation.",
        tokenizer,
        model,
        max_tokens=120,
        temperature=0.7 
    ))

    print("\nSuggested Order:", opt["optimized_qty"])
    print(f"Final MAE: {metrics['mae'][-1]:.2f}")
    
    user = input("Press Enter to approve or type new quantity: ").strip()

    if user.isdigit():
        new_qty = int(user)
        log_entry = {"event": "override", "new": new_qty}
    else:
        log_entry = {"event": "approved", "qty": opt['optimized_qty']}

    with open(os.path.join(SCConfig.LOG_DIR, "decision_log.json"), "w") as f:
        json.dump(to_serializable(log_entry), f, indent=2)

    log("Decision saved.")


if __name__ == "__main__":
    main()
