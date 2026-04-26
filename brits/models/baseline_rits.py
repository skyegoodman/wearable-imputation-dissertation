import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class FeatureRegression(nn.Module):
    def __init__(self, input_size, sensor_size=None):
        super(FeatureRegression, self).__init__()
        self.sensor_size = int(input_size if sensor_size is None else min(max(1, sensor_size), input_size))
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        if self.sensor_size < input_size:
            # Auxiliary channels are allowed as predictors but not as feature-regression targets.
            m[self.sensor_size:, :] = 0.0
        self.register_buffer("m", m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * self.m, self.b)
        if self.sensor_size < x.size(1):
            # Auxiliary channels are deterministic inputs, not feature-regression targets.
            z_h[:, self.sensor_size:] = x[:, self.sensor_size:]
        return z_h


class TemporalDecay(nn.Module):
    """
    Original BRITS temporal decay (kept for compatibility/ablation):
      gamma = exp(-ReLU(W d + b))
    If diag=True and input_size==output_size, it uses only element-wise mapping.
    """
    def __init__(self, input_size, output_size, diag=False):
        super(TemporalDecay, self).__init__()
        self.diag = diag
        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag is True:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag is True:
            gamma = F.relu(F.linear(d, self.W * self.m, self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class Model(nn.Module):
    def __init__(
        self,
        rnn_hid_size,
        impute_weight,
        label_weight,
        input_size=11,
        sensor_size=9,
        decay_mode="original",  # "original"
    ):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.input_size = int(input_size)
        self.sensor_size = int(min(max(1, sensor_size), self.input_size))
        self.decay_mode = str(decay_mode)

        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(self.input_size * 2, self.rnn_hid_size)

        # Keep original decays (for ablation/compatibility)
        self.temp_decay_h = TemporalDecay(
            input_size=self.input_size,
            output_size=self.rnn_hid_size,
            diag=False,
        )
        self.temp_decay_x = TemporalDecay(
            input_size=self.input_size,
            output_size=self.input_size,
            diag=True,
        )

        self.hist_reg = nn.Linear(self.rnn_hid_size, self.input_size)
        self.feat_reg = FeatureRegression(self.input_size, sensor_size=self.sensor_size)
        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)

        self.dropout = nn.Dropout(p=0.25)
        self.out = nn.Linear(self.rnn_hid_size, 1)  # unused when label_weight=0.0

    def _compute_decays(self, d):
        """
        Returns:
          gamma_h: (B, H)
          gamma_x: (B, D)
        """
        if self.decay_mode not in ("original", "feature_specific"):
            raise ValueError(f"Unknown decay_mode: {self.decay_mode}")
        # Feature-specific decay retired; both modes use original BRITS decay.
        gamma_h = self.temp_decay_h(d)
        gamma_x = self.temp_decay_x(d)

        return gamma_h, gamma_x

    def forward(self, data, direct):
        values = data[direct]["values"]
        masks = data[direct]["masks"]
        deltas = data[direct]["deltas"]

        evals = data[direct]["evals"]
        eval_masks = data[direct]["eval_masks"]
        K = self.sensor_size

        device = values.device
        h = torch.zeros((values.size(0), self.rnn_hid_size), device=device)
        c = torch.zeros((values.size(0), self.rnn_hid_size), device=device)

        x_loss = 0.0
        imputations = []

        T = values.size(1)
        for t in range(T):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h, gamma_x = self._compute_decays(d)

            h = h * gamma_h
            h_d = self.dropout(h)

            x_h = self.hist_reg(h_d)
            x_loss += torch.sum(torch.abs(x[:, :K] - x_h[:, :K]) * m[:, :K]) / (torch.sum(m[:, :K]) + 1e-5)

            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x[:, :K] - z_h[:, :K]) * m[:, :K]) / (torch.sum(m[:, :K]) + 1e-5)

            alpha = torch.sigmoid(self.weight_combine(torch.cat([gamma_x, m], dim=1)))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x[:, :K] - c_h[:, :K]) * m[:, :K]) / (torch.sum(m[:, :K]) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim=1)
            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim=1))

        imputations = torch.cat(imputations, dim=1)

        loss = x_loss * self.impute_weight

        return {
            "loss": loss,
            "imputations": imputations,
            "evals": evals,
            "eval_masks": eval_masks,
            "labels": data["labels"].view(-1, 1),
            "is_train": data["is_train"].view(-1, 1),
            "predictions": None,
        }

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data, direct="forward")

        if optimizer is not None:
            optimizer.zero_grad()
            ret["loss"].backward()
            optimizer.step()

        return ret
