"""Evaluate a text causal language model served by vLLM on CORE."""

import argparse
import asyncio
import os
from pathlib import Path
import time

from nanochat.common import print0
from nanochat.core_eval_data import (
    build_core_results,
    load_core_tasks,
    write_core_results_csv,
    write_core_task_details_jsonl,
)
from nanochat.report import get_report
from nanochat.vllm_core_eval import (
    VLLMCoreClient,
    VLLMCoreError,
    evaluate_vllm_task,
    evaluate_vllm_task_with_details,
)


def resolve_output_name(model, output_name):
    """Return a safe CSV stem without changing an explicit valid name."""
    if output_name is None:
        return model.replace('/', '-').replace('\\', '-') + '-vllm'
    if not output_name or output_name in ('.', '..') or Path(output_name).name != output_name:
        raise ValueError("output_name must be a non-empty file name without directories")
    return output_name


async def run_evaluation(args, output_name=None):
    if output_name is None:
        output_name = resolve_output_name(args.model, args.output_name)
    core_details_dir = getattr(args, 'core_details_dir', None)
    show_progress = getattr(args, 'show_progress', True)
    api_key = args.api_key or os.environ.get('VLLM_API_KEY')
    client = VLLMCoreClient(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        concurrency=args.concurrency,
        timeout=args.timeout,
        retry_limit=args.retry_limit,
    )

    async with client:
        await client.validate_model()
        print0(f"Connected to vLLM model: {args.model}")
        print0(f"Endpoint: {args.base_url.rstrip('/')}")
        print0(f"Concurrent requests: {args.concurrency}")

        tasks = load_core_tasks(max_per_task=args.max_per_task)
        results = {}
        random_baselines = {}
        for task_order, task in enumerate(tasks):
            start_time = time.time()
            label = task['label']
            task_meta = task['task_meta']
            print0(
                f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, "
                f"type: {task_meta['task_type']})...",
            )
            try:
                if core_details_dir is None:
                    accuracy = await evaluate_vllm_task(
                        client,
                        task['data'],
                        task_meta,
                        show_progress=show_progress,
                        progress_desc=f"CORE {label}",
                    )
                else:
                    accuracy, detail_records = await evaluate_vllm_task_with_details(
                        client,
                        task['data'],
                        task_meta,
                        show_progress=show_progress,
                        progress_desc=f"CORE {label}",
                    )
                    details_path = write_core_task_details_jsonl(
                        detail_records,
                        core_details_dir,
                        output_name,
                        args.model,
                        'vllm',
                        label,
                        task_order,
                        task_meta,
                    )
                    print0(f"Details written to: {details_path}")
            except Exception as exc:
                raise VLLMCoreError(f"CORE task {label!r} failed: {exc}") from exc
            results[label] = accuracy
            random_baseline = task['random_baseline']
            random_baselines[label] = random_baseline
            centered = (accuracy - 0.01 * random_baseline) / (
                1.0 - 0.01 * random_baseline
            )
            elapsed = time.time() - start_time
            print0(
                f"accuracy: {accuracy:.4f} | centered: {centered:.4f} | "
                f"time: {elapsed:.2f}s"
            )
    return build_core_results(results, random_baselines)


def main(args):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size > 1:
        raise RuntimeError(
            "core_eval_vllm does not support torchrun; use --concurrency instead"
        )
    if args.max_per_task == 0 or args.max_per_task < -1:
        raise ValueError("max_per_task must be -1 or a positive integer")

    output_name = resolve_output_name(args.model, args.output_name)
    core_results = asyncio.run(run_evaluation(args, output_name))
    output_csv_path = write_core_results_csv(core_results, output_name)
    print0(f"\nResults written to: {output_csv_path}")
    print0(f"CORE metric: {core_results['core_metric']:.4f}")

    report_data = [{
        'model': f"{args.model} (vLLM)",
        'CORE metric': core_results['core_metric'],
    }]
    report_data.append(core_results['centered_results'])
    get_report().log(section="Base model evaluation", data=report_data)


def build_parser():
    """Build the command-line parser for remote CORE evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a vLLM-hosted causal language model on CORE",
        allow_abbrev=False,
    )
    parser.add_argument(
        '--base-url',
        default='http://localhost:5000',
        help='vLLM server root URL (default: http://localhost:5000)',
    )
    parser.add_argument('--model', required=True, help='Served model name')
    parser.add_argument(
        '--api-key',
        default=None,
        help='Optional API key (defaults to VLLM_API_KEY)',
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=16,
        help='Maximum concurrent HTTP requests (default: 16)',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=300.0,
        help='Per-request timeout in seconds (default: 300)',
    )
    parser.add_argument(
        '--retry-limit',
        type=int,
        default=3,
        help='Retries for transport errors, HTTP 429, and HTTP 5xx (default: 3)',
    )
    parser.add_argument(
        '--max-per-task',
        type=int,
        default=-1,
        help='Maximum examples per CORE task (-1 = all)',
    )
    parser.add_argument(
        '--output-name',
        default=None,
        help='CSV file stem (default: served model name plus -vllm)',
    )
    parser.add_argument(
        '--core-details-dir',
        type=Path,
        default=None,
        help='Optional directory for per-example CORE JSONL files',
    )
    parser.add_argument(
        '--no-progress',
        dest='show_progress',
        action='store_false',
        help='Disable per-task tqdm progress bars',
    )
    parser.set_defaults(show_progress=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
