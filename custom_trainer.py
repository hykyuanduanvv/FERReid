"""Copyright: Nabarun Goswami (2024)."""
import math
import time
from typing import Dict, List, Optional, Union

import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl, \
    is_torch_xla_available, is_datasets_available
from transformers.debug_utils import DebugOption
from transformers.modeling_utils import unwrap_model
from transformers.trainer_utils import speed_metrics
from transformers.utils import logging
import numpy as np
import torchvision

logger = logging.get_logger(__name__)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class AddExtraLossesToTrainerState(TrainerCallback):
    def __init__(self, extra_losses: List[str]):
        self.extra_losses = extra_losses

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.extra_losses = {k: torch.tensor(0.0).to(args.device) for k in self.extra_losses}
        return control


class CustomTrainer(Trainer):

    def __init__(self, extra_losses: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        if extra_losses is not None:
            self.add_callback(AddExtraLossesToTrainerState(extra_losses))

        self.eval_dataloader = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if hasattr(self.control, 'extra_losses') and model.training:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

            if not isinstance(outputs, dict):
                raise ValueError("The model output should be a dictionary or ModelOutput and not a tuple or list.")
            for k, v in outputs.items():
                if k in self.control.extra_losses:
                    if v is not None:
                        if self.args.n_gpu > 1:
                            v = v.mean()
                        self.control.extra_losses[k] += v.detach() / self.args.gradient_accumulation_steps

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = learning_rate

            if hasattr(self.control, 'extra_losses'):
                for k, v in self.control.extra_losses.items():
                    logs[k] = self._nested_gather(v).mean().item()
                    # reset the loss
                    self.control.extra_losses[k] -= self.control.extra_losses[k]

                    logs[k] = round(logs[k] / (self.state.global_step - self._globalstep_last_logged), 4)

            logs.update(unwrap_model(model).get_extra_logging_dict()
                        if hasattr(unwrap_model(model), 'get_extra_logging_dict') else {})

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            try:
                self._save_checkpoint(model, trial, metrics=metrics)
            except TypeError as exc:
                if "unexpected keyword argument 'metrics'" not in str(exc):
                    raise
                self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    @torch.no_grad()
    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        test_cats = []
        for name in ['cat','chimp','chinchilla','degus','dog','ferret','guineapig','hamster','hedgehog','javasparrow','parakeet','pig','rabbit']:
            if self.args.cluster_index != -1:
                if name in self.categories[self.args.cluster_index]:
                    test_cats.append(name)
        from ops.evaluation import evaluate_verification
        all_results = evaluate_verification(self.model, test_cats, self.args.image_path)
        self.log(all_results)
        return all_results

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if not self.args.replace_sampler:
            return super()._get_train_sampler()
        
        print('Using custom sampler')

        dataset = self.train_dataset
        pids = np.array(list(dataset.label2images.keys()))
        labels = np.array([dataset.pid2label[x] for x in pids])
        print(labels)
        label_to_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}
        num_batches = 10000
        batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.accelerator.num_processes
        print('batch_size: ', batch_size, 'num_batches: ', num_batches)
        all_batches = []
        
        while len(all_batches) < num_batches:
            current_label = np.random.choice(list(label_to_indices.keys()))
            indices = label_to_indices[current_label]
            # print(current_label, indices, len(indices))
            # exit(0)
            batch_indices = np.random.choice(indices, batch_size, replace=True)
            all_batches.append(batch_indices)
        indices = np.array(all_batches).flatten()
        # print(indices)
        from torch.utils.data import DataLoader, Dataset, Sampler
        class OrderedSampler(Sampler):
            def __init__(self, indices):
                self.indices = indices  # List of specific indices

            def __iter__(self):
                return iter(self.indices)  # Yield indices in order

            def __len__(self):
                return len(self.indices)
        return OrderedSampler(indices)
