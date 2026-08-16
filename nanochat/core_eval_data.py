"""Dataset loading, aggregation, and output helpers for CORE evaluation."""

import csv
import json
import os
import random
import re
import tempfile

import yaml

from nanochat.common import get_base_dir, get_eval_bundle_dir


def load_core_tasks(max_per_task=-1):
    """Load deterministic CORE task data and random baselines."""
    eval_bundle_dir = get_eval_bundle_dir()

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row['Eval Task']] = float(row['Random baseline'])

    loaded_tasks = []
    for task in tasks:
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' '),
        }
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        loaded_tasks.append({
            'label': label,
            'task_meta': task_meta,
            'data': data,
            'random_baseline': random_baselines[label],
        })
    return loaded_tasks


def build_core_results(results, random_baselines):
    """Build centered task results and the aggregate CORE metric."""
    if not results:
        raise ValueError("CORE results must not be empty")
    if set(results) != set(random_baselines):
        raise ValueError("CORE results and random baselines must have the same tasks")
    centered_results = {
        label: (accuracy - 0.01 * random_baselines[label])
        / (1.0 - 0.01 * random_baselines[label])
        for label, accuracy in results.items()
    }
    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": sum(centered_results.values()) / len(centered_results),
    }


def write_core_results_csv(core_results, model_slug):
    """Write CORE results using the existing base-evaluation CSV format."""
    base_dir = get_base_dir()
    output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
        for label in core_results["results"]:
            accuracy = core_results["results"][label]
            centered = core_results["centered_results"][label]
            f.write(f"{label:<35}, {accuracy:<10.6f}, {centered:<10.6f}\n")
        f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
    return output_csv_path


def _safe_path_component(value):
    """Convert a display name to a portable, non-empty path component."""
    component = re.sub(r'[^A-Za-z0-9._-]+', '-', str(value)).strip('.-_')
    return component or 'unnamed'


def write_core_task_details_jsonl(
    records,
    output_dir,
    model_slug,
    model_name,
    backend,
    task_label,
    task_order,
    task_meta,
):
    """Atomically write ordered per-example CORE details for one task."""
    if not records:
        raise ValueError("CORE detail records must not be empty")

    ordered_records = sorted(records, key=lambda record: record['example_index'])
    example_indices = [record['example_index'] for record in ordered_records]
    if example_indices != list(range(len(ordered_records))):
        raise ValueError("CORE detail records must contain each example index exactly once")

    model_dir = os.path.join(
        os.path.abspath(os.path.expanduser(os.fspath(output_dir))),
        _safe_path_component(model_slug),
    )
    os.makedirs(model_dir, exist_ok=True)
    task_slug = _safe_path_component(task_label)
    output_path = os.path.join(model_dir, f"{task_order:02d}-{task_slug}.jsonl")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=model_dir,
            prefix=f".{task_order:02d}-{task_slug}.",
            suffix='.tmp',
            delete=False,
        ) as f:
            temporary_path = f.name
            for record in ordered_records:
                output_record = {
                    'model': model_name,
                    'backend': backend,
                    'task': task_label,
                    'task_order': task_order,
                    'task_meta': task_meta,
                    **record,
                }
                json.dump(
                    output_record,
                    f,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(',', ':'),
                )
                f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return output_path
