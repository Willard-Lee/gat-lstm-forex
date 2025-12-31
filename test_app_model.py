"""
Test script to verify the app.py model can load the trained weights.
Run this before starting Streamlit to ensure everything works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Copy exact model classes from app.py
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_f, out_f, dropout, alpha, concat=True):
        super().__init__()
        self.out_f, self.concat = out_f, concat
        self.W = nn.Parameter(torch.empty(in_f, out_f)); nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(2*out_f, 1)); nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W); B, N, _ = Wh.size()
        a_in = torch.cat([Wh.repeat_interleave(N,1), Wh.repeat(1,N,1)], 2).view(B,N,N,2*self.out_f)
        e = self.leakyrelu(torch.matmul(a_in, self.a).squeeze(3))
        attn = F.softmax(torch.where(adj>0, e, -9e15*torch.ones_like(e)), dim=2)
        out = torch.matmul(attn, Wh)
        return (F.elu(out), attn) if self.concat else (out, attn)

class HybridGATLSTM(nn.Module):
    def __init__(self, nfeat, n_nodes, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(1, 32)
        self.gat1 = GraphAttentionLayer(32,32,dropout,0.2,True)
        self.gat2 = GraphAttentionLayer(32,16,dropout,0.2,False)
        self.lstm = nn.LSTM(n_nodes, 128, 3, batch_first=True, dropout=dropout)
        comb = 128 + 16*n_nodes
        self.head_dir = nn.Sequential(nn.Linear(comb,64), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(64,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
        self.head_ret = nn.Sequential(nn.Linear(comb,64), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(64,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
        self.head_vol = nn.Sequential(nn.Linear(comb,64), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(64,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,2))
    def forward(self, x, adj):
        B,S,N,F = x.size(); x_emb = self.embedding(x).view(B*S,N,-1)
        adj_b = adj.unsqueeze(0).repeat(B*S,1,1) if adj.dim()==2 else adj
        xg,_ = self.gat1(x_emb, adj_b); xg,attn2 = self.gat2(xg, adj_b)
        xg_out = xg.view(B,S,-1)[:,-1,:]
        xl_out = self.lstm(x.view(B,S,N))[0][:,-1,:]
        c = torch.cat([xl_out, xg_out], 1)
        return self.head_dir(c), self.head_ret(c), self.head_vol(c), attn2

# Configuration
FEATURE_NODES = ['rsi_14', 'macd', 'macd_signal', 'macd_hist', 'ema_20',
                 'log_return', 'rolling_vol_14', 'momentum_5', 'rsi_momentum', 'macd_momentum', 'price_ema_dist']
DEVICE = torch.device('cpu')
MODEL_PATH = "models/gat_lstm_model.pth"

print("="*60)
print("TESTING APP.PY MODEL LOADING")
print("="*60)

# Step 1: Create model
print("\n1. Creating model...")
model = HybridGATLSTM(1, len(FEATURE_NODES), 0.3).to(DEVICE)
print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Step 2: Check model file exists
print(f"\n2. Checking for trained model at: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    file_size = os.path.getsize(MODEL_PATH) / (1024*1024)
    print(f"   ✅ Model file found ({file_size:.2f} MB)")
else:
    print(f"   ❌ Model file not found!")
    exit(1)

# Step 3: Load state dict
print("\n3. Loading trained weights...")
try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    print(f"   ✅ State dict loaded ({len(state_dict)} keys)")
except Exception as e:
    print(f"   ❌ Failed to load state dict: {e}")
    exit(1)

# Step 4: Load into model
print("\n4. Loading weights into model...")
try:
    model.load_state_dict(state_dict)
    print(f"   ✅ Weights loaded successfully!")
except Exception as e:
    print(f"   ❌ Failed to load weights into model: {e}")
    print("\nMismatched keys:")
    model_keys = set(model.state_dict().keys())
    saved_keys = set(state_dict.keys())
    print(f"   Missing in model: {saved_keys - model_keys}")
    print(f"   Missing in saved: {model_keys - saved_keys}")
    exit(1)

# Step 5: Test forward pass
print("\n5. Testing forward pass...")
try:
    model.eval()
    x = torch.randn(1, 30, 11, 1).to(DEVICE)
    adj = torch.eye(11).to(DEVICE)
    with torch.no_grad():
        p_d, p_r, p_v, attn = model(x, adj)
    prob = torch.sigmoid(p_d).item()
    print(f"   ✅ Forward pass successful!")
    print(f"   Sample prediction: {prob:.2%} (direction probability)")
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\nThe app.py model is working correctly.")
print("You can now run: streamlit run app.py")
print("\nIf you still see the error in Streamlit:")
print("1. Stop the Streamlit server (Ctrl+C)")
print("2. Clear browser cache (Ctrl+Shift+R)")
print("3. Run: streamlit run app.py --server.runOnSave false")
