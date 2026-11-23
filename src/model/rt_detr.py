from __future__ import annotations

from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer

# HGNetV2 백본이 차지하는 top-level 레이어 인덱스 범위 (model.0 ~ model.9).
# rtdetr-l/rtdetr-x 등 동일 아키텍처 계열의 scale 변형은 레이어 수가 아니라
# 채널/반복 수만 달라지므로 이 경계는 scale에 관계없이 유지된다.
_BACKBONE_LAYER_COUNT = 10


class RTDETRTrainerWithBackboneLR(RTDETRTrainer):
    """backbone(HGNetV2)에 낮은 LR을 적용하는 RT-DETR 트레이너.

    ultralytics 기본 build_optimizer는 backbone/transformer 구분 없이 단일 lr을 쓴다.
    이 저장소의 dino_*.yaml은 이미 backbone/attention에 lr factor(0.05)를 낮게 주는데,
    rf_detr도 lr/lr_encoder를 분리해서 쓴다 — RT-DETR(ultralytics wrapper)만 이 차등이
    없어 발산에 더 취약하다는 진단(2026-09-11)에 따라 동일한 방식을 이식한다.
    """

    backbone_lr_factor: float = 0.1

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        import torch.nn as nn

        g_backbone = [{}, {}, {}]  # weight(decay), norm/weight(no decay), bias
        g_rest = [{}, {}, {}]
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)

        for module_name, module in model.named_modules():
            layer_idx = _layer_index(module_name)
            is_backbone = layer_idx is not None and layer_idx < _BACKBONE_LAYER_COUNT
            target = g_backbone if is_backbone else g_rest
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:
                    target[2][fullname] = param
                elif isinstance(module, bn):
                    target[1][fullname] = param
                else:
                    target[0][fullname] = param

        backbone_lr = lr * self.backbone_lr_factor
        optimizer = self._optimizer_from_groups(name, lr, momentum, decay, g_rest)
        for weight_group, param_lr, param_decay in zip(g_backbone, [backbone_lr, backbone_lr, backbone_lr], [decay, 0.0, 0.0]):
            if weight_group:
                optimizer.add_param_group({"params": list(weight_group.values()), "lr": param_lr, "weight_decay": param_decay})

        return optimizer

    def _optimizer_from_groups(self, name, lr, momentum, decay, groups):
        """ultralytics DetectionTrainer.build_optimizer와 동일한 규칙으로 옵티마이저를 조립한다."""
        import torch

        weight, norm_weight, bias = groups
        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD"}
        resolved_name = {x.lower(): x for x in optimizers}.get(str(name).lower(), "AdamW")

        if resolved_name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optim_cls = getattr(torch.optim, resolved_name)
            optim_args = dict(lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif resolved_name == "RMSProp":
            optim_cls = torch.optim.RMSprop
            optim_args = dict(lr=lr, momentum=momentum)
        else:
            optim_cls = torch.optim.SGD
            optim_args = dict(lr=lr, momentum=momentum, nesterov=True)

        param_groups = [
            {"params": list(weight.values()), **optim_args, "weight_decay": decay},
            {"params": list(norm_weight.values()), **optim_args, "weight_decay": 0.0},
            {"params": list(bias.values()), **optim_args, "weight_decay": 0.0},
        ]
        return optim_cls(param_groups)


def _layer_index(module_name: str) -> int | None:
    """'model.3.conv' 같은 이름에서 top-level 레이어 인덱스(3)를 뽑는다."""
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
        return int(parts[1])
    return None


def build_rt_detr_model(runtime_cfg: dict) -> RTDETR:
    weight_ref = runtime_cfg["paths"]["weights"]
    if not weight_ref:
        raise ValueError("model weight is missing for RT-DETR")
    return RTDETR(weight_ref)


def resolve_rt_detr_trainer(runtime_cfg: dict):
    """backbone_lr_factor가 config에 있으면 차등 LR 트레이너를, 없으면 기본 동작(None)을 반환한다."""
    factor = runtime_cfg.get("train", {}).get("backbone_lr_factor")
    if factor is None:
        return None

    trainer_cls = type(
        "RTDETRTrainerWithBackboneLR",
        (RTDETRTrainerWithBackboneLR,),
        {"backbone_lr_factor": float(factor)},
    )
    return trainer_cls