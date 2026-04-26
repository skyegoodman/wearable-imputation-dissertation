from __future__ import annotations

import torch

from pypots.imputation import SAITS
from pypots.imputation.saits.model import _SAITS


class _SensorOnlyLossSAITS(_SAITS):
    """PyPOTS SAITS core with loss restricted to the first sensor_size channels."""

    def __init__(self, n_features: int, *args, sensor_size: int | None = None, **kwargs):
        super().__init__(n_features=n_features, *args, **kwargs)
        self.n_features = int(n_features)
        self.sensor_size = int(self.n_features if sensor_size is None else sensor_size)
        self.sensor_size = max(1, min(self.sensor_size, self.n_features))

    def _slice_sensor(self, *tensors):
        return tuple(t[:, :, : self.sensor_size] for t in tensors)

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
        diagonal_attention_mask: bool = True,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        if (self.training and self.diagonal_attention_mask) or ((not self.training) and diagonal_attention_mask):
            diagonal_attention_mask = (1 - torch.eye(self.n_steps)).to(X.device)
            diagonal_attention_mask = diagonal_attention_mask.unsqueeze(0)
        else:
            diagonal_attention_mask = None

        (
            X_tilde_1,
            X_tilde_2,
            X_tilde_3,
            first_DMSA_attn_weights,
            second_DMSA_attn_weights,
            combining_weights,
        ) = self.encoder(X, missing_mask, diagonal_attention_mask)

        imputed_data = missing_mask * X + (1 - missing_mask) * X_tilde_3

        results = {
            "first_DMSA_attn_weights": first_DMSA_attn_weights,
            "second_DMSA_attn_weights": second_DMSA_attn_weights,
            "combining_weights": combining_weights,
            "imputation": imputed_data,
            "reconstruction": X_tilde_3,
            "X_tilde_1": X_tilde_1,
            "X_tilde_2": X_tilde_2,
            "X_tilde_3": X_tilde_3,
        }

        if calc_criterion:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            X1_s, X_s, mask_s = self._slice_sensor(X_tilde_1, X, missing_mask)
            X2_s = X_tilde_2[:, :, : self.sensor_size]
            X3_s = X_tilde_3[:, :, : self.sensor_size]
            Xori_s, ind_s = self._slice_sensor(X_ori, indicating_mask)

            if self.training:
                ORT_loss = 0
                ORT_loss += self.training_loss(X1_s, X_s, mask_s)
                ORT_loss += self.training_loss(X2_s, X_s, mask_s)
                ORT_loss += self.training_loss(X3_s, X_s, mask_s)
                ORT_loss /= 3
                ORT_loss = self.ORT_weight * ORT_loss

                MIT_loss = self.MIT_weight * self.training_loss(X3_s, Xori_s, ind_s)
                loss = ORT_loss + MIT_loss

                results["ORT_loss"] = ORT_loss
                results["MIT_loss"] = MIT_loss
                results["loss"] = loss
            else:
                results["metric"] = self.validation_metric(X3_s, Xori_s, ind_s)

        return results


class SensorAwareSAITS(SAITS):
    """PyPOTS SAITS wrapper that can exclude auxiliary channels from losses."""

    def __init__(self, *args, sensor_size: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensor_size = int(self.n_features if sensor_size is None else sensor_size)
        self.sensor_size = max(1, min(self.sensor_size, self.n_features))

        if self.sensor_size < self.n_features:
            self.model = _SensorOnlyLossSAITS(
                n_layers=self.n_layers,
                n_steps=self.n_steps,
                n_features=self.n_features,
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_k=self.d_k,
                d_v=self.d_v,
                d_ffn=self.d_ffn,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout,
                diagonal_attention_mask=self.diagonal_attention_mask,
                ORT_weight=self.ORT_weight,
                MIT_weight=self.MIT_weight,
                training_loss=self.training_loss,
                validation_metric=self.validation_metric,
                sensor_size=self.sensor_size,
            )
            self._send_model_to_given_device()
            self.optimizer.init_optimizer(self.model.parameters())


def build_model(cfg: dict, n_steps: int, n_features: int) -> SAITS:
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    sensor_size = int(mcfg.get("sensor_size", n_features))

    model = SensorAwareSAITS(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=int(mcfg["n_layers"]),
        d_model=int(mcfg["d_model"]),
        d_ffn=int(mcfg["d_inner"]),
        n_heads=int(mcfg["n_heads"]),
        d_k=int(mcfg["d_k"]),
        d_v=int(mcfg["d_v"]),
        dropout=float(mcfg["dropout"]),
        attn_dropout=float(mcfg["attn_dropout"]),
        batch_size=int(tcfg["batch_size"]),
        epochs=int(tcfg["epochs"]),
        patience=int(tcfg["patience"]),
        saving_path=tcfg["saving_dir"],
        model_saving_strategy=tcfg["model_saving_strategy"],
        device=tcfg.get("device"),
        sensor_size=sensor_size,
    )
    return model
