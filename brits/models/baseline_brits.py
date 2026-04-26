import torch
import torch.nn as nn
from . import baseline_rits as rits 


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
        self.rits_f = rits.Model(
            self.rnn_hid_size,
            self.impute_weight,
            self.label_weight,
            input_size=self.input_size,
            sensor_size=self.sensor_size,
            decay_mode=self.decay_mode,
        )
        self.rits_b = rits.Model(
            self.rnn_hid_size,
            self.impute_weight,
            self.label_weight,
            input_size=self.input_size,
            sensor_size=self.sensor_size,
            decay_mode=self.decay_mode,
        )

    def forward(self, data):
        ret_f = self.rits_f(data, "forward")
        ret_b = self.reverse(self.rits_b(data, "backward"))  # align backward outputs to forward time
        ret = self.merge_ret(ret_f, ret_b)
        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f["loss"]
        loss_b = ret_b["loss"]
        loss_c = self.get_consistency_loss(ret_f["imputations"], ret_b["imputations"])

        loss = loss_f + loss_b + loss_c
        imputations = (ret_f["imputations"] + ret_b["imputations"]) / 2

        ret_f["loss"] = loss
        ret_f["imputations"] = imputations

        # If predictions exist (classification), merge; otherwise keep None
        pred_f = ret_f.get("predictions", None)
        pred_b = ret_b.get("predictions", None)
        if (pred_f is not None) and (pred_b is not None):
            ret_f["predictions"] = (pred_f + pred_b) / 2
        else:
            ret_f["predictions"] = None

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        return torch.abs(pred_f[:, :, :self.sensor_size] - pred_b[:, :, :self.sensor_size]).mean() * 1e-1

    def reverse(self, ret):
        """
        Reverse time dimension (dim=1) for any tensor outputs with shape (B, T, ...).
        """

        def reverse_tensor(tensor_):
            if tensor_ is None:
                return None
            if not torch.is_tensor(tensor_):
                return tensor_
            if tensor_.dim() <= 1:
                return tensor_
            idx = torch.arange(tensor_.size(1) - 1, -1, -1, device=tensor_.device)
            return tensor_.index_select(1, idx)

        for key in list(ret.keys()):
            ret[key] = reverse_tensor(ret[key])

        return ret

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret["loss"].backward()
            optimizer.step()

        return ret
