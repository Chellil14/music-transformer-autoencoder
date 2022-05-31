#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

from data_loader import DatasetMusicAE
from model import MusicAETransformer
import metrics


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class InverseSqrtSchedule:
    def __init__(self, lr: float, warmup_steps: int, hidden_size: int, optimizer: optim.Optimizer):
        super(InverseSqrtSchedule, self).__init__()

        # self.max_lr: float = lr * ((warmup_steps * hidden_size) ** -0.5)
        self.max_lr: float = lr * (warmup_steps ** -0.5)
        self.warmup_steps: int = warmup_steps
        self.optimizer: optim.Optimizer = optimizer

        self._step = 0

    def step(self) -> None:
        """
        Update parameters and rate
        """
        self._step += 1
        rate: float = self.get_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate

    def get_cur_step(self) -> int:
        """
        Get the current step
        """
        return self._step

    def get_rate(self) -> float:
        """
        Get the learning rate for this step
        """
        # FIXME incorrect
        # lr * min(step / warmup_steps) * (max(warmup_steps, step) ** -0.5) * (hidden_size ** -0.5)
        step_ratio: float = self._step / self.warmup_steps
        return self.max_lr * min(step_ratio, step_ratio ** -0.5)

def transform_inputs(input_data, device):
    if "melody" in input_data:
        performance = input_data["performance"].to(device=device)
        melody = input_data["melody"].to(device=device)
    else:
        performance = input_data["inputs"].to(device=device)
        melody = None
    targets = input_data["targets"].to(device=device)

    return performance, melody, targets

def get_nll_loss(output_probs, targets):
    return F.nll_loss(torch.log(output_probs).transpose(-2, -1), targets)


def do_train(data_dir, prefix, train_dir, configs={}):
    data_dir = Path(data_dir)
    train_dir = Path(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)

    # defaults
    batch_size = configs.get('batch_size', 1)
    hidden_size = configs.get('hidden_size', 384)
    warmup_steps = configs.get('warmup_steps', 8000)
    epochs = configs.get('epochs', 50000)
    eval_steps = configs.get('eval_steps', 1000)
    lr = configs.get('lr', 0.1)
    melody_combine_method=configs.get('melody_combine_method', 'NONE')

    # Dataset
    train_dataset = DatasetMusicAE(data_dir, prefix, "train")
    validation_dataset = DatasetMusicAE(data_dir, prefix, "validation")
    test_dataset = DatasetMusicAE(data_dir, prefix, "test")

    train_loader = DataLoader(train_dataset, batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size)

    # Model
    model = MusicAETransformer(
        num_layers=6,
        num_heads=8,
        max_att_len=512,
        hidden_dim=384,
        filter_dim=1024,
        dropout_rate=0.1,
        input_bias=True,
        qk_dim=512,
        v_dim=384,
        normalize_eps=1e-9,
        melody_combine_method=melody_combine_method
    )
    device = get_default_device()
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # NOTE Don't use the default betas
    scheduler = InverseSqrtSchedule(lr, warmup_steps, hidden_size, optimizer)

    # Train Start
    # TODO add progress bar using tqdm
    print("Starting training")
    epoch = 0
    with tqdm(total=epochs) as pbar:
        while epoch < epochs:
            for input_data in train_loader:
                optimizer.zero_grad()
                model.train()
                performance, melody, targets = transform_inputs(input_data, device)
                output_probs = model(performance=performance, melody=melody)
                loss = get_nll_loss(output_probs, targets)
                pbar.set_postfix_str(f"Train loss: {loss.item()}")
                loss.backward()
                optimizer.step()
                scheduler.step()

                # result_metrics = metric_set(sample, batch_y)
                if epoch % eval_steps == 0:
                    model.eval()
                    torch.save(model.state_dict(), train_dir / f'{prefix}-{epoch}.pth')
                    total_loss = 0.0
                    total_samples = 0
                    with tqdm(test_loader) as pbar_eval:
                        for input_data in pbar_eval:
                            performance, melody, targets = transform_inputs(input_data, device)
                            output_probs = model(performance=performance, melody=melody)
                            loss = get_nll_loss(output_probs, targets)
                            total_loss += loss.item()
                            total_samples += 1
                    print(f"Epoch: {epoch} Test loss: {total_loss / total_samples}")
                epoch += 1
                pbar.update(1)
                if epoch >= epochs:
                    break
                torch.cuda.empty_cache()

    torch.save(model.state_dict(), train_dir / f'{prefix}-final.pth')


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
            help="The directory where the preprocessed data is located")
    parser.add_argument("--prefix", type=str, required=True,
            help="Dataset file prefix(score2perf problem name in Magenta)")
    parser.add_argument("--train_dir", type=str, required=True,
            help="The directory where the trained model is saved")
    parser.add_argument("--config_file",
            help="Path to YAML config file")
    parser.add_argument("--configs", dest="configs", default=[], nargs="*",
            help="A list of configuration items. "
                 "An item is a file path or a 'key=value' formatted string. "
                 "The type of a value is determined by applying int(), float(), and str() "
                 "to it sequentially.")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args, _ = parser.parse_known_args()

    data_dir = Path(args.data_dir)
    prefix = args.prefix
    train_dir = Path(args.train_dir)

    # Training options
    configs = {}
    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            try:
                configs = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f'Error while loading {args.config_file}: {e}')

    def parse_value(v):
        try:
            return int(v)
        except:
            try:
                return float(v)
            except:
                return str(v)

    for x in args.configs:
        k, v = x.split('=', 1)
        configs[k] = parse_value(v)

    do_train(args.data_dir, args.prefix, args.train_dir, configs)
