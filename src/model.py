import torch
import torch.nn as nn

# Horizons, in windows, for the onset head. Must match the onset_k*.npy files
# written by pipeline_v2.make_onset_labels().
ONSET_HORIZONS = [1, 5, 15, 30]


class WorldModel(nn.Module):
    """
    Learns network state-transition dynamics from windowed flow observations.

    Four heads share one LSTM trunk:
      stage_head   which ATT&CK stage the NEXT flow belongs to  (5 classes)
      breach_head  probability the next flow is malicious       (scalar)
      state_head   the next flow's feature vector itself        (input_size)
      onset_head   P(a NEW stage begins within k windows)       (len(ONSET_HORIZONS))

    state_head is what makes this a world model rather than a sequence
    classifier: it predicts P(S_t+1 | S_t) in state space, so the rollout in
    forecast.py can feed a predicted state back in and simulate K steps forward.

    With attention=True the trunk pools the window by learned per-timestep
    attention instead of taking the last hidden state, and forward() can return
    those weights -- which flow inside the window drove the prediction. That is
    an explanation of the model's own computation, unlike gradient x input, which
    is computed after the fact. It is off by default because switching it on
    changes the trunk, so a checkpoint trained without it must not be loaded
    with it.

    onset_head is the forecasting target proper. Stage classification is
    dominated by persistence -- ~91% of windows carry the same label as the
    previous one, so a model can score well by copying the last label. Onset
    asks whether the stage *changes*, which a persistence baseline gets wrong
    by construction. It is the head to judge forecasting skill on.
    """

    def __init__(self, input_size=24, hidden_size=128, num_layers=2,
                 num_stages=5, dropout=0.3, onset_horizons=None,
                 attention=False):
        super().__init__()
        self.input_size = input_size
        self.onset_horizons = list(onset_horizons or ONSET_HORIZONS)
        self.attention = attention
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        # Additive attention over timesteps, replacing last-step pooling. Off by
        # default: turning it on changes how the trunk summarises the window, so
        # a checkpoint trained without it would produce different predictions if
        # it were silently enabled. Train with --attention to use it.
        self.attn = nn.Linear(hidden_size, 1) if attention else None
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.stage_head = nn.Linear(64, num_stages)
        self.breach_head = nn.Linear(64, 1)
        self.state_head = nn.Linear(64, input_size)
        self.onset_head = nn.Linear(64, len(self.onset_horizons))

    def forward(self, x, return_attention=False):
        out, _ = self.lstm(x)
        if self.attn is not None:
            w = torch.softmax(self.attn(out).squeeze(-1), dim=1)   # (batch, seq)
            pooled = (out * w.unsqueeze(-1)).sum(dim=1)
        else:
            # last-step pooling: the final timestep's hidden state already
            # summarises the window through the recurrence
            w = None
            pooled = out[:, -1, :]
        shared = self.fc(pooled)
        stage_logits = self.stage_head(shared)
        breach_prob = torch.sigmoid(self.breach_head(shared)).squeeze(1)
        next_state = self.state_head(shared)
        onset_logits = self.onset_head(shared)
        if return_attention:
            return stage_logits, breach_prob, next_state, onset_logits, w
        return stage_logits, breach_prob, next_state, onset_logits

    @torch.no_grad()
    def rollout(self, x, steps=5, temperature=1.0):
        """
        K-step forward simulation in state space. Predict the next state, append
        it to the window, drop the oldest row, repeat. Returns the predicted
        state trajectory and the stage distribution at each step.

        temperature rescales the logits before softmax. Free-running rollout
        compounds overconfidence across steps, so an uncalibrated trajectory is
        confidently wrong by the last one; pass the fitted T from
        models/temperature.json.
        """
        window = x.clone()
        states, stages = [], []
        for _ in range(steps):
            logits, _, nxt, _ = self.forward(window)
            states.append(nxt)
            stages.append(torch.softmax(logits / temperature, dim=1))
            window = torch.cat([window[:, 1:, :], nxt.unsqueeze(1)], dim=1)
        return torch.stack(states, dim=1), torch.stack(stages, dim=1)
