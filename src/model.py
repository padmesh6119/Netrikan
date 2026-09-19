import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """
    Learns network state-transition dynamics from windowed flow observations.

    Three heads share one LSTM trunk:
      stage_head   which ATT&CK stage the NEXT flow belongs to  (5 classes)
      breach_head  probability the next flow is malicious       (scalar)
      state_head   the next flow's feature vector itself        (input_size)

    state_head is what makes this a world model rather than a sequence
    classifier: it predicts P(S_t+1 | S_t) in state space, so the rollout in
    forecast.py can feed a predicted state back in and simulate K steps forward.
    """

    def __init__(self, input_size=24, hidden_size=128, num_layers=2,
                 num_stages=5, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.stage_head = nn.Linear(64, num_stages)
        self.breach_head = nn.Linear(64, 1)
        self.state_head = nn.Linear(64, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        shared = self.fc(last)
        stage_logits = self.stage_head(shared)
        breach_prob = torch.sigmoid(self.breach_head(shared)).squeeze(1)
        next_state = self.state_head(shared)
        return stage_logits, breach_prob, next_state

    @torch.no_grad()
    def rollout(self, x, steps=5):
        """
        K-step forward simulation in state space. Predict the next state, append
        it to the window, drop the oldest row, repeat. Returns the predicted
        state trajectory and the stage distribution at each step.
        """
        window = x.clone()
        states, stages = [], []
        for _ in range(steps):
            logits, _, nxt = self.forward(window)
            states.append(nxt)
            stages.append(torch.softmax(logits, dim=1))
            window = torch.cat([window[:, 1:, :], nxt.unsqueeze(1)], dim=1)
        return torch.stack(states, dim=1), torch.stack(stages, dim=1)
