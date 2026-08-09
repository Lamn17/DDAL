import csv
import sys
import json
import yaml
import time
import pandas
import argparse
import subprocess
import numpy as np
import os
import tempfile
from pathlib import Path
from loguru import logger
from typing import Dict, Union
from dotenv import load_dotenv
from setproctitle import setproctitle

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env.training')


logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

def parse_args():
    parser = argparse.ArgumentParser(description='Run active learning experiment')
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent.parent / 'configs' / 'config.yaml'),
                        help='Path to config YAML (default: configs/config.yaml)')
    parser.add_argument('--dataset_yaml', type=str, help='Override dataset_yaml')
    parser.add_argument('--epochs', type=int, help='Override epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch_size')
    parser.add_argument('--val_batch_size', type=int, help='Override val_batch_size')
    parser.add_argument('--imgsz', type=int, help='Override imgsz')
    parser.add_argument('--device', type=str, help='Override device (can be single device or comma-separated list for multiple GPUs)')
    parser.add_argument('--patience', type=int, help='Override patience')
    parser.add_argument('--experiments_root', type=str, help='Override experiments_root')
    parser.add_argument('--seed', type=str, help='Override seed (int or path to folder)')
    parser.add_argument('--strategy', type=str, help='Override strategy (can be single strategy or comma-separated for chaining)')
    parser.add_argument('--model_name', type=str, help='Override model_name')
    parser.add_argument('--initial_labeled_count', type=int, help='Override initial_labeled_count')
    parser.add_argument('--max_rounds', type=int, help='Override max_rounds')
    parser.add_argument('--samples_per_round', type=int, help='Override samples_per_round')
    parser.add_argument('--num_inference', type=str, help='Num images to run inference, -1 for full dataset, or path to folder')
    parser.add_argument('--resume_experiment_dir', type=str, help='Resume an existing experiment directory instead of creating a new one')
    parser.add_argument('--start_round', type=int, help='Round to resume from; defaults to the first incomplete round')
    parser.add_argument('--override', action='append', help='Override nested config keys, e.g., --override strategy_args.fdal.supporter_embedding_size=512')

    return parser.parse_args()


def load_config(config_path: Union[str, Path]) -> Dict:
    config_path = Path(config_path) if config_path else Path(__file__).parent.parent / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['_config_path'] = str(config_path)
    return config


def set_nested(d, key, value):
    keys = key.split('.')
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def apply_overrides(config: Dict, args) -> Dict:
    if args.dataset_yaml:
        config['dataset_yaml'] = args.dataset_yaml
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.val_batch_size is not None:
        config['val_batch_size'] = args.val_batch_size
    if args.imgsz is not None:
        config['imgsz'] = args.imgsz
    if args.device:
        if ',' in args.device:
            config['device'] = [d.strip() for d in args.device.split(',')]
        else:
            config['device'] = args.device
    if args.patience is not None:
        config['patience'] = args.patience
    if args.experiments_root:
        config['experiments_root'] = args.experiments_root
    if args.seed is not None:
        config['seed'] = args.seed
    if args.strategy:
        if '-' in args.strategy:
            config['strategy'] = [s.strip() for s in args.strategy.split('-')]
        else:
            config['strategy'] = args.strategy
    if args.model_name:
        config['model_name'] = args.model_name
    if args.initial_labeled_count is not None:
        config['initial_labeled_count'] = args.initial_labeled_count
    if args.max_rounds is not None:
        config['max_rounds'] = args.max_rounds
    if args.samples_per_round is not None:
        config['samples_per_round'] = args.samples_per_round
    if args.num_inference is not None:
        config['num_inference'] = args.num_inference
    
    if args.override:
        for ov in args.override:
            if '=' not in ov:
                logger.warning(f"Invalid override format: {ov}")
                continue
            key, value_str = ov.split('=', 1)
            try:
                if value_str.isdigit():
                    value = int(value_str)
                elif '.' in value_str and value_str.replace('.', '').replace('-', '').isdigit():
                    value = float(value_str)
                else:
                    value = value_str
            except:
                value = value_str
            set_nested(config, key, value)
    
    return config


def set_process_title(config: Dict):
    proctitle_startstr = os.environ.get('PROCTITLE_STARTSTR') or 'recal'
    proctitle_midstr = config['proctitle_midstr']
    proctitle_endstr_of = config.get('proctitle_endstr_of', '')
    
    endstr_value = str(config[proctitle_endstr_of]) if proctitle_endstr_of and proctitle_endstr_of in config else ''
    process_title = f"{proctitle_startstr}_{proctitle_midstr}_{endstr_value}"
    setproctitle(process_title)


def run_data_setup(config: Dict) -> str:
    dataset_yaml = config.get('dataset_yaml')
    dataset_path = None
    dataset_name = None
    class_names = []
    
    if dataset_yaml:
        try:
            with open(dataset_yaml, 'r') as f:
                dy = yaml.safe_load(f)
            dataset_path = str(dy.get('path') or dy.get('root') or '')
            names = dy.get('names')
            if isinstance(names, dict):
                class_names = [names[k] for k in sorted(names, key=lambda x: int(x))]
            elif isinstance(names, list):
                class_names = names

            dataset_name = None
            if isinstance(dy.get('name'), str) and dy.get('name').strip():
                dataset_name = dy.get('name').strip()
            else:
                parent_name = Path(dataset_yaml).parent.name
                if parent_name and parent_name.strip():
                    dataset_name = parent_name
                elif dataset_path:
                    dataset_name = Path(dataset_path).stem
                else:
                    dataset_name = Path(dataset_yaml).stem
        except Exception as e:
            raise ValueError(f"Failed to read dataset YAML {dataset_yaml}: {e}")
    
    if not dataset_path:
        dataset_path = config.get('dataset_path', '/default/path')
    if not dataset_name:
        dataset_name = config.get('dataset_name', 'default_dataset')
    if not class_names:
        class_names = config.get('class_names', ['default_class'])

    strategy = config.get('strategy', 'experiment')
    if isinstance(strategy, list):
        strategy_str = '-'.join(strategy)
    else:
        strategy_str = str(strategy)
    
    setup_args = [
        sys.executable, str(Path(__file__).parent / 'setup_data.py'),
        '--dataset_yaml', str(dataset_yaml) if dataset_yaml else '',
        '--initial_count', str(config.get('initial_labeled_count', 0)),
        '--experiments_root', config.get('experiments_root', 'experiments'),
        '--seed', str(config.get('seed', 42)),
        '--model_name', config.get('model_name', 'yolo'),
        '--strategy', strategy_str
    ]
    
    result = subprocess.run(setup_args, check=True, capture_output=True, text=True)
    experiment_dir = result.stdout.strip().split('\n')[-1]

    return experiment_dir


def run_training(config: Dict, experiment_dir: str, round_num: int = 0) -> Dict:
    dataset_yaml_path = Path(experiment_dir) / f"round_{round_num}" / "data.yaml"
    model_path_file = Path(experiment_dir) / f"round_{round_num}" / "model_path.txt"
    metrics = {}
    
    model_name = config.get('model_name', 'yolo11n.pt')
    model_run_name = model_name.split('/')[-1].replace('.pt', '').replace('.yaml', '')
    strategy = config.get('strategy', 'random')
    
    if isinstance(strategy, list):
        strategy_str = '-'.join(strategy)
    else:
        strategy_str = str(strategy)
    device = config.get('device', 'auto')
    device_str = ','.join(str(d) for d in device) if isinstance(device, list) else str(device)
    
    train_args = [
        sys.executable, str(Path(__file__).parent / 'train.py'),
        '--dataset_yaml', str(dataset_yaml_path),
        '--model', model_name,
        '--epochs', str(config.get('epochs', 100)),
        '--batch_size', str(config.get('batch_size', 16)),
        '--imgsz', str(config.get('imgsz', 640)),
        '--device', device_str,
        '--project', str(Path(experiment_dir) / f"round_{round_num}" / "train"),
        '--save_dir', str(Path(experiment_dir) / f"round_{round_num}" / "train" / model_run_name),
        '--name', model_run_name,
        '--model_path_file', str(model_path_file),
        '--strategy', strategy_str
    ]

    if config.get('patience') is not None:
        train_args.extend(['--patience', str(config.get('patience'))])
    if config.get('workers') is not None:
        train_args.extend(['--workers', str(config.get('workers'))])
    strategy_args = config.get('strategy_args', {})
    quality_cfg = strategy_args.get('recal', strategy_args.get('ddus', {}))
    train_args.extend([
        '--quality_momentum', str(quality_cfg.get('quality_momentum', quality_cfg.get('momentum', 0.9))),
        '--quality_initial_value', str(quality_cfg.get('quality_initial', quality_cfg.get('initial_quality', 0.5))),
        '--quality_xi', str(quality_cfg.get('quality_xi', quality_cfg.get('xi', 0.5))),
    ])
    strategy_names = {item.strip() for item in strategy_str.replace(',', '-').split('-') if item.strip()}
    if 'recal' in strategy_names and round_num > 0:
        previous_quality = sorted(
            (Path(experiment_dir) / f"round_{round_num - 1}" / "train").glob("*/weights/classwise_quality.npy")
        )
        if previous_quality:
            train_args.extend(['--quality_initial_path', str(previous_quality[-1])])
    
    env = os.environ.copy()
    
    try:
        subprocess.run(train_args, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Training failed for round {round_num}: {e}")
        raise
    metrics_file = Path(experiment_dir) / f"round_{round_num}" / "train" / model_run_name / "results.csv"
    if model_path_file.exists():
        trained_model_path = Path(model_path_file.read_text().strip())
        candidate_metrics_file = trained_model_path.parent.parent / "results.csv"
        if candidate_metrics_file.exists():
            metrics_file = candidate_metrics_file
    if metrics_file.exists():
        df = pandas.read_csv(metrics_file)
        last_row = df.iloc[-1]
        metrics = {
            'train/box_loss': float(last_row['train/box_loss']),
            'train/cls_loss': float(last_row['train/cls_loss']),
            'train/dfl_loss': float(last_row['train/dfl_loss']),
            'metrics/precision': float(last_row['metrics/precision(B)']),
            'metrics/recall': float(last_row['metrics/recall(B)']),
            'map50': float(last_row['metrics/mAP50(B)']),
            'map50-95': float(last_row['metrics/mAP50-95(B)']),
            'val/box_loss': float(last_row['val/box_loss']),
            'val/cls_loss': float(last_row['val/cls_loss']),
            'val/dfl_loss': float(last_row['val/dfl_loss']),
        }
    return metrics


def model_run_name(config: Dict) -> str:
    model_name = config.get('model_name', 'yolo11n.pt')
    return model_name.split('/')[-1].replace('.pt', '').replace('.yaml', '')


def round_model_path(config: Dict, experiment_dir: str, round_num: int) -> Path:
    model_path_file = Path(experiment_dir) / f"round_{round_num}" / "model_path.txt"
    if model_path_file.exists():
        model_path = Path(model_path_file.read_text().strip())
        if model_path.exists():
            return model_path

    run_name = model_run_name(config)
    weights_dir = Path(experiment_dir) / f"round_{round_num}" / "train" / run_name / "weights"
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"
    if best_path.exists():
        return best_path
    return last_path


def round_has_trained_model(config: Dict, experiment_dir: str, round_num: int) -> bool:
    return round_model_path(config, experiment_dir, round_num).exists()


def read_round_metrics(config: Dict, experiment_dir: str, round_num: int) -> Dict:
    model_path = round_model_path(config, experiment_dir, round_num)
    if model_path.exists():
        metrics_file = model_path.parent.parent / "results.csv"
    else:
        metrics_file = Path(experiment_dir) / f"round_{round_num}" / "train" / model_run_name(config) / "results.csv"

    if not metrics_file.exists():
        return {}

    df = pandas.read_csv(metrics_file)
    if df.empty:
        return {}

    last_row = df.iloc[-1]
    return {
        'train/box_loss': float(last_row['train/box_loss']),
        'train/cls_loss': float(last_row['train/cls_loss']),
        'train/dfl_loss': float(last_row['train/dfl_loss']),
        'metrics/precision': float(last_row['metrics/precision(B)']),
        'metrics/recall': float(last_row['metrics/recall(B)']),
        'map50': float(last_row['metrics/mAP50(B)']),
        'map50-95': float(last_row['metrics/mAP50-95(B)']),
        'val/box_loss': float(last_row['val/box_loss']),
        'val/cls_loss': float(last_row['val/cls_loss']),
        'val/dfl_loss': float(last_row['val/dfl_loss']),
    }


def append_round_result(results: Dict, round_num: int, metrics: Dict):
    if round_num in results['rounds']:
        idx = results['rounds'].index(round_num)
        results['metrics_history'][idx] = metrics
        return
    results['rounds'].append(round_num)
    results['metrics_history'].append(metrics)


def collect_existing_results(config: Dict, experiment_dir: str, max_rounds: int) -> Dict:
    results = {'rounds': [], 'metrics_history': []}
    for round_num in range(0, max_rounds + 1):
        if round_has_trained_model(config, experiment_dir, round_num):
            append_round_result(results, round_num, read_round_metrics(config, experiment_dir, round_num))
    return results


def infer_resume_start_round(config: Dict, experiment_dir: str, max_rounds: int, is_yoloe: bool) -> int:
    if not is_yoloe and not round_has_trained_model(config, experiment_dir, 0):
        return 0

    first_al_round = 1
    for round_num in range(first_al_round, max_rounds + 1):
        if not round_has_trained_model(config, experiment_dir, round_num):
            return round_num
    return max_rounds + 1


def write_results_file(experiment_dir: str, results: Dict):
    ordered = sorted(zip(results['rounds'], results['metrics_history']), key=lambda x: x[0])
    results['rounds'] = [r for r, _ in ordered]
    results['metrics_history'] = [m for _, m in ordered]

    results_file = Path(experiment_dir) / "results.csv"
    with open(results_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        for i in range(len(results['rounds'])):
            writer.writerow({k: v[i] for k, v in results.items() if i < len(v)})


def should_train_maple(config: Dict) -> bool:
    maple_training = config.get("maple_training", {})
    if not maple_training.get("enabled", False):
        return False
    strategy = config.get("strategy", [])
    if isinstance(strategy, str):
        strategy_items = strategy.split("-") if "-" in strategy else [strategy]
    else:
        strategy_items = list(strategy)
    return "maple" in strategy_items


def run_maple_training(config: Dict, experiment_dir: str, round_num: int) -> str:
    maple_cfg = config.get("maple_training", {})
    round_dir = Path(experiment_dir) / f"round_{round_num}"
    data_yaml_path = round_dir / "data.yaml"
    crop_root = round_dir / maple_cfg.get("crop_dirname", "maple_crops")
    checkpoint_path = Path(experiment_dir) / "maple_checkpoints" / f"round_{round_num:03d}.pt"
    log_dir = Path(experiment_dir) / "maple_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    crop_args = [
        sys.executable,
        str(Path(__file__).parent / "create_yolo_crops.py"),
        "--data_yaml",
        str(data_yaml_path),
        "--output",
        str(crop_root),
        "--splits",
        "train",
        "val",
        "--val_output_name",
        "test",
        "--padding",
        str(maple_cfg.get("padding", 0.15)),
        "--overwrite",
    ]
    crop_log = log_dir / f"round_{round_num:03d}_crops.log"
    with crop_log.open("w") as f:
        subprocess.run(crop_args, check=True, stdout=f, stderr=subprocess.STDOUT)

    device = maple_cfg.get("device", config.get("device", "auto"))
    if isinstance(device, list):
        device = f"cuda:{device[0]}" if str(device[0]).isdigit() else str(device[0])
    elif str(device).isdigit():
        device = f"cuda:{device}"

    train_args = [
        sys.executable,
        str(Path(__file__).parent / "train_maple_round.py"),
        "--crop_root",
        str(crop_root),
        "--round",
        str(round_num),
        "--output",
        str(checkpoint_path),
        "--backbone",
        str(maple_cfg.get("backbone", "ViT-B/16")),
        "--n_ctx",
        str(maple_cfg.get("n_ctx", 2)),
        "--ctx_init",
        str(maple_cfg.get("ctx_init", "")),
        "--prompt_depth",
        str(maple_cfg.get("prompt_depth", 9)),
        "--epochs",
        str(maple_cfg.get("epochs", 50)),
        "--batch_size",
        str(maple_cfg.get("batch_size", 32)),
        "--num_workers",
        str(maple_cfg.get("num_workers", 0)),
        "--val_interval",
        str(maple_cfg.get("val_interval", 1)),
        "--learning_rate",
        str(maple_cfg.get("learning_rate", 0.0025)),
        "--weight_decay",
        str(maple_cfg.get("weight_decay", 0.0005)),
        "--momentum",
        str(maple_cfg.get("momentum", 0.9)),
        "--device",
        str(device),
    ]
    train_log = log_dir / f"round_{round_num:03d}_train.log"
    logger.info(f"MaPLe crop log: {crop_log}")
    logger.info(f"MaPLe train log: {train_log}")
    with train_log.open("w") as f:
        try:
            subprocess.run(train_args, check=True, stdout=f, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            tail = ""
            if train_log.exists():
                lines = train_log.read_text(errors="replace").splitlines()
                tail = "\n".join(lines[-80:])
            raise RuntimeError(
                f"MaPLe training failed for round {round_num}. "
                f"See log: {train_log}\nLast log lines:\n{tail}"
            ) from exc
    return str(checkpoint_path)

def run_strategy(config: Dict, experiment_dir: str, round_num: int):
    model_path_file = Path(experiment_dir) / f"round_{round_num-1}" / "model_path.txt"
    if model_path_file.exists():
        with open(model_path_file, 'r') as f:
            model_path = f.read().strip()
    else:
        model_name = config.get('model_name', 'model')
        if model_name.endswith('.pt'):
            model_name = model_name[:-3]
        model_path = Path(experiment_dir) / f"round_{round_num-1}" / "train" / model_name / "weights" / "best.pt"
        print(f"Model path file not found, using default path: {model_path}")
    
    output_file = Path(experiment_dir) / f"round_{round_num}" / "selected_indices.npy"
    dataset_yaml = config.get('dataset_yaml')
    dataset_path = None
    dataset_name = None
    class_names = []
    
    if dataset_yaml:
        try:
            with open(dataset_yaml, 'r') as f:
                dy = yaml.safe_load(f)
            dataset_path = str(dy.get('path') or dy.get('root') or '')
            names = dy.get('names')
            if isinstance(names, dict):
                class_names = [names[k] for k in sorted(names, key=lambda x: int(x))]
            elif isinstance(names, list):
                class_names = names

            dataset_name = None
            if isinstance(dy.get('name'), str) and dy.get('name').strip():
                dataset_name = dy.get('name').strip()
            else:
                parent_name = Path(dataset_yaml).parent.name
                if parent_name and parent_name.strip():
                    dataset_name = parent_name
                elif dataset_path:
                    dataset_name = Path(dataset_path).stem
                else:
                    dataset_name = Path(dataset_yaml).stem
        except Exception as e:
            raise ValueError(f"Failed to read dataset YAML {dataset_yaml}: {e}")
    
    if not dataset_path:
        dataset_path = config.get('dataset_path', '/default/path')
    if not dataset_name:
        dataset_name = config.get('dataset_name', 'default_dataset')
    if not class_names:
        class_names = config.get('class_names', ['default_class'])

    strategy = config.get('strategy', 'random')
    if isinstance(strategy, list):
        strategy_str = '-'.join(strategy)
    else:
        strategy_str = str(strategy)
    
    device = config.get('device', 'auto')
    device_str = ','.join(str(d) for d in device) if isinstance(device, list) else str(device)
    
    strategy_args = [
        sys.executable, str(Path(__file__).parent / 'strategy.py'),
        '--dataset_yaml', str(dataset_yaml) if dataset_yaml else '',
        '--strategy', strategy_str,
        '--model', str(model_path),
        '--num_samples', str(config.get('samples_per_round', 50)),
        '--num_inference', str(config['num_inference']),
        '--experiment_dir', experiment_dir,
        '--round', str(round_num),
        '--output_file', str(output_file),
        '--config', str(Path(config.get('_temp_config_path', config.get('_config_path', Path(__file__).parent.parent / 'configs' / 'config.yaml')))),
        '--device', device_str
    ]
    
    env = os.environ.copy()
    cuda_visible = env.get('CUDA_VISIBLE_DEVICES', 'not set')
    
    subprocess.run(strategy_args, check=True, env=env)
    
    if output_file.exists():
        return np.load(output_file, allow_pickle=True)
    return np.array([])


def simulate_labeling_cli(selected_indices: np.ndarray, config: Dict, experiment_dir: str, round_num: int):
    dataset_yaml = config.get('dataset_yaml')
    dataset_path = None
    
    if dataset_yaml:
        try:
            with open(dataset_yaml, 'r') as f:
                dy = yaml.safe_load(f)
            dataset_path = str(dy.get('path') or dy.get('root') or '')
        except Exception as e:
            raise ValueError(f"Failed to read dataset YAML {dataset_yaml}: {e}")
    
    if not dataset_path:
        dataset_path = config.get('dataset_path', '/default/path')

    indices_file = Path(experiment_dir) / f"round_{round_num}" / "selected_indices.npy"
    indices_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(indices_file, selected_indices)

    label_args = [
        sys.executable, str(Path(__file__).parent / 'simulate_labeling.py'),
        '--dataset_yaml', str(dataset_yaml) if dataset_yaml else '',
        '--dataset', dataset_path,
        '--indices_file', str(indices_file),
        '--experiment_dir', experiment_dir,
        '--round', str(round_num)
    ]
    
    subprocess.run(label_args, check=True)


def main():
    args = parse_args()
    
    config = load_config(args.config)
    config = apply_overrides(config, args)
    
    temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, temp_config_file)
    temp_config_file.close()
    config['_temp_config_path'] = temp_config_file.name
    
    device_config = config.get('device', ['0'])
    
    # Normalize device_config to handle both string and list inputs consistently
    if isinstance(device_config, str):
        if ',' in device_config:
            # Handle comma-separated device string like "0,1,2"
            device_list = [d.strip() for d in device_config.split(',')]
            device_str = device_config
            is_list_config = True
        else:
            # Single device string like "3"
            device_list = [device_config]
            device_str = device_config
            is_list_config = False
    elif isinstance(device_config, list):
        device_list = [str(d) for d in device_config]
        device_str = ','.join(device_list)
        is_list_config = True
    else:
        device_str = str(device_config)
        device_list = [device_str]
        is_list_config = False
    
    if device_str not in ['auto', 'cpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        logger.info(f"Set CUDA_VISIBLE_DEVICES = {device_str}")
    
    set_process_title(config)
    
    logger.info("Loaded configuration")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    if not config.get('dataset_yaml'):
        logger.error('Dataset YAML path is required')
        return 1
    if config.get('max_rounds', 0) <= 0:
        logger.error('max_rounds must be > 0')
        return 1
    
    max_rounds = config.get('max_rounds', 1)
    model_name = config.get('model_name', 'yolo11n.pt')
    is_yoloe = 'yoloe' in model_name.lower()

    if args.resume_experiment_dir:
        experiment_dir = str(Path(args.resume_experiment_dir))
        if not Path(experiment_dir).exists():
            logger.error(f"Resume experiment directory does not exist: {experiment_dir}")
            return 1
        results = collect_existing_results(config, experiment_dir, max_rounds)
        start_round = args.start_round if args.start_round is not None else infer_resume_start_round(config, experiment_dir, max_rounds, is_yoloe)
        logger.info(f"Resuming experiment in: {experiment_dir}")
        logger.info(f"Existing completed rounds: {results['rounds']}")
        logger.info(f"Starting from round: {start_round}")
    else:
        try:
            experiment_dir = run_data_setup(config)
            logger.info(f"Initialized experiment in: {experiment_dir}")
            logger.info(f"Initial selection log: {Path(experiment_dir) / 'selection_log.txt'}")
        except Exception as e:
            logger.exception(f"Failed to initialize experiment: {e}")
            return 1
        results = {'rounds': [], 'metrics_history': []}
        start_round = 0 if not is_yoloe else 1

    if start_round > max_rounds:
        logger.info(f"No work to do: start_round={start_round} is greater than max_rounds={max_rounds}")
        logger.info(f"Results saved in: {experiment_dir}")
        write_results_file(experiment_dir, results)
        return 0

    if not is_yoloe and start_round <= 0:
        logger.info("=== Initial Training (Round 0) ===")
        try:
            start_time = time.time()
            metrics = run_training(config, experiment_dir, 0)
            append_round_result(results, 0, metrics)
            if should_train_maple(config):
                logger.info("Training MaPLe for round 0...")
                maple_checkpoint = run_maple_training(config, experiment_dir, 0)
                logger.info(f"MaPLe round 0 checkpoint: {maple_checkpoint}")
            end_time = time.time()
            logger.info(f"Initial training completed in {end_time - start_time:.2f} seconds")
            start_round = 1
        except Exception as e:
            logger.exception(f"Initial training failed: {e}")
            return 1
    else:
        if is_yoloe:
            logger.info("=== YOLOE Model: Skipping initial training ===")
        elif start_round > 1 and not round_has_trained_model(config, experiment_dir, start_round - 1):
            logger.error(f"Cannot start round {start_round}: previous round model is missing")
            return 1

    for round_num in range(max(1, start_round), max_rounds + 1):
        try:
            logger.info(f"=== Round {round_num} ===")

            if round_has_trained_model(config, experiment_dir, round_num):
                metrics = read_round_metrics(config, experiment_dir, round_num)
                append_round_result(results, round_num, metrics)
                logger.info(f"Round {round_num} already has a trained model; skipping")
                continue

            round_dir = Path(experiment_dir) / f"round_{round_num}"
            if (round_dir / "metadata.yaml").exists() and (round_dir / "selected_indices.npy").exists():
                logger.info(f"Round {round_num} already has selected/labeled data; reusing it")
            else:
                start_time = time.time()
                selected_indices = run_strategy(config, experiment_dir, round_num)

                if len(selected_indices) == 0:
                    logger.info("No more samples to select. Stopping experiment.")
                    break

                logger.info(f"Selected {len(selected_indices)} samples for labeling")
                end_time = time.time()
                logger.info(f"Strategy completed in {end_time - start_time:.2f} seconds")
                simulate_labeling_cli(selected_indices, config, experiment_dir, round_num)

            start_time = time.time()
            logger.info(f"Training model for round {round_num}...")
            metrics = run_training(config, experiment_dir, round_num)
            if should_train_maple(config):
                logger.info(f"Training MaPLe for round {round_num}...")
                maple_checkpoint = run_maple_training(config, experiment_dir, round_num)
                logger.info(f"MaPLe round {round_num} checkpoint: {maple_checkpoint}")
            
            append_round_result(results, round_num, metrics)
            write_results_file(experiment_dir, results)

            end_time = time.time()
            logger.info(f"Training completed in {end_time - start_time:.2f} seconds")

            if metrics:
                logger.info(f"Metrics for round {round_num}:\n{json.dumps(metrics, indent=2)}")
            else:
                logger.info(f"No metrics produced for round {round_num}")

        except Exception as e:
            logger.exception(f"Round {round_num} failed with error: {e}")
            continue
    
    logger.info("=== Experiment Complete ===")
    logger.info(f"Completed {len(results['rounds'])} rounds")
    logger.info(f"Results saved in: {experiment_dir}")
    write_results_file(experiment_dir, results)
    
    if results['metrics_history']:
        if not is_yoloe and len(results['metrics_history']) > 1:
            initial_map = results['metrics_history'][0].get('map50', 0)
            final_map = results['metrics_history'][-1].get('map50', 0)
            logger.info(f"Initial mAP: {initial_map:.4f}")
            logger.info(f"Final mAP: {final_map:.4f}")
            logger.info(f"Improvement: {final_map - initial_map:.4f}")
        elif is_yoloe:
            final_map = results['metrics_history'][-1].get('map50', 0)
            logger.info(f"Final mAP: {final_map:.4f}")
    
    # Clean up temp config file
    if '_temp_config_path' in config:
        os.unlink(config['_temp_config_path'])
    
    return 0


if __name__ == "__main__":
    exit(main())
