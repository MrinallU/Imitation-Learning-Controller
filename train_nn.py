import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class ImitationPolicy(nn.Module):
    def __init__(self, n_actions=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),  nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, state_goal):
        """
        state_goal : [B,4]  (cx, cy, gx, gy)
        returns     : log-probs [B, n_actions]
        """
        logits = self.net(state_goal)
        return F.log_softmax(logits, dim=-1)


def load_dataset(csv_path="demos.csv"):
    import numpy as np, pandas as pd
    df = pd.read_csv(csv_path, header=None,
                     names=["cx","cy","gx","gy","a"])
    X = torch.tensor(df[["cx","cy","gx","gy"]].values,
                     dtype=torch.float32)
    y = torch.tensor(df["a"].values, dtype=torch.long)
    return TensorDataset(X, y)


def train_imitation(csv_file, epochs=5000, batch=64, lr=3e-4):
    ds   = load_dataset(csv_file)
    dl   = DataLoader(ds, batch_size=batch, shuffle=True)
    net  = ImitationPolicy()
    opt  = torch.optim.Adam(net.parameters(), lr=lr)

    for ep in range(epochs):
        running_loss = 0.0
        for X, y in dl:
            opt.zero_grad()
            logp = net(X)                    # [B,6]
            loss = F.nll_loss(logp, y)       # cross-entropy
            loss.backward()
            opt.step()
            running_loss += loss.item() * X.size(0)
        print(f"epoch {ep:02d}  loss {running_loss/len(ds):.4f}")
    torch.save(net.state_dict(), "imit_policy.pth")
    return net


ACTIONS = [
    (0.00, 0.00),  # 0
    (0.22, 0.00),  # 1
    (0.00, 0.50),  # 2
    (0.00,-0.50),  # 3
    (0.22, 0.50),  # 4
    (0.22,-0.50)   # 5
]

def select_action(net, cx, cy, gx, gy):
    with torch.no_grad():
        logp = net(torch.tensor([[cx, cy, gx, gy]],
                                dtype=torch.float32))
        a    = logp.argmax(dim=1).item()
    return ACTIONS[a]            # (linear, angular)


if __name__ == "__main__":
    policy = train_imitation("demos.csv")
    lin, ang = select_action(policy, cx=6, cy=0, gx=7.0, gy=1.0)
    print(f"cmd_vel: linear {lin:.2f}  angular {ang:.2f}")
