"""Evaluate a chat model served by vLLM."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from nanochat.chat_eval_data import (
    ChatTaskDetailsSpool,
    write_chat_task_details_jsonl,
)
from nanochat.chat_eval_common import ALL_CHAT_TASKS, calculate_chatcore, create_chat_task
from nanochat.common import print0
from nanochat.report import get_report
from nanochat.vllm_chat_eval import (
    VLLMChatClient,
    VLLMChatError,
    evaluate_vllm_chat_task,
    evaluate_vllm_chat_task_with_details,
)


def parse_json_object(value):
    """Parse one CLI JSON object."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a valid JSON object: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def resolve_task_names(task_name):
    """Resolve a pipe-delimited task selection and validate every name."""
    task_names = list(ALL_CHAT_TASKS) if task_name is None else task_name.split('|')
    if not task_names or any(not name for name in task_names):
        raise ValueError("task_name must contain at least one non-empty task name")
    unknown = [name for name in task_names if name not in ALL_CHAT_TASKS]
    if unknown:
        available = ', '.join(ALL_CHAT_TASKS)
        raise ValueError(
            f"unknown chat evaluation task {unknown[0]!r}; available tasks: {available}"
        )
    return task_names


def validate_args(args):
    """Validate CLI values before contacting the server."""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size > 1:
        raise RuntimeError(
            "chat_eval_vllm does not support torchrun; use --concurrency instead"
        )
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if args.temperature < 0:
        raise ValueError("temperature must be non-negative")
    if args.top_k == 0 or args.top_k < -1:
        raise ValueError("top_k must be -1 or a positive integer")
    if args.max_problems is not None and args.max_problems <= 0:
        raise ValueError("max_problems must be positive")
    resolve_task_names(args.task_name)


async def run_evaluation(args):
    """Run the selected tasks through one shared vLLM client session."""
    api_key = args.api_key or os.environ.get('VLLM_API_KEY')
    client = VLLMChatClient(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        concurrency=args.concurrency,
        timeout=args.timeout,
        retry_limit=args.retry_limit,
        chat_template_kwargs=args.chat_template_kwargs,
    )

    results = {}
    async with client:
        await client.validate_model()
        print0(f"Connected to vLLM model: {args.model}")
        print0(f"Endpoint: {args.base_url.rstrip('/')}")
        print0(f"Concurrent requests: {args.concurrency}")

        for task_name in resolve_task_names(args.task_name):
            print0(f"\nEvaluating: {task_name}")
            task_object = create_chat_task(task_name)
            try:
                eval_kwargs = {
                    'num_samples': args.num_samples,
                    'max_new_tokens': args.max_new_tokens,
                    'temperature': args.temperature,
                    'top_k': args.top_k,
                    'max_problems': args.max_problems,
                }
                if args.chat_details_dir is None:
                    accuracy = await evaluate_vllm_chat_task(
                        client,
                        task_object,
                        **eval_kwargs,
                    )
                else:
                    task_order = ALL_CHAT_TASKS.index(task_name)
                    evaluation_config = build_evaluation_config(args)
                    spool = ChatTaskDetailsSpool(
                        args.chat_details_dir,
                        f"{args.model}-vllm",
                        args.model,
                        'vllm',
                        task_name,
                        task_order,
                        evaluation_config,
                    )
                    print0(f"In-progress details: {spool.path}")
                    try:
                        accuracy, records = (
                            await evaluate_vllm_chat_task_with_details(
                                client,
                                task_object,
                                detail_callback=spool.append,
                                **eval_kwargs,
                            )
                        )
                        spool.close()
                        details_path = write_chat_task_details_jsonl(
                            records,
                            args.chat_details_dir,
                            f"{args.model}-vllm",
                            args.model,
                            'vllm',
                            task_name,
                            task_order,
                            evaluation_config,
                        )
                        spool.discard()
                    except BaseException:
                        spool.close()
                        print0(
                            "In-progress details retained after failure: "
                            f"{spool.path}"
                        )
                        raise
                    print0(f"Details written to: {details_path}")
            except Exception as exc:
                raise VLLMChatError(
                    f"chat evaluation task {task_name!r} failed: {exc}"
                ) from exc
            results[task_name] = accuracy
            for subtask, score in getattr(task_object, 'last_subtask_scores', {}).items():
                print0(f"  {subtask}: {100 * score:.2f}%")
            print0(f"{task_name} accuracy: {100 * accuracy:.2f}%")
    return results


def build_evaluation_config(args):
    """Return non-secret settings repeated in each task detail record."""
    return {
        'num_samples': args.num_samples,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'max_problems': args.max_problems,
        'chat_template_kwargs': args.chat_template_kwargs,
    }


def sanitized_report_config(args):
    """Return reportable CLI settings without authentication material."""
    config = vars(args).copy()
    config.pop('api_key', None)
    return config


def main(args):
    validate_args(args)
    results = asyncio.run(run_evaluation(args))
    chatcore_metric = calculate_chatcore(results)
    if chatcore_metric:
        print0(f"ChatCORE metric: {chatcore_metric['ChatCORE metric']:.4f}")
    get_report().log(section="Chat evaluation vllm", data=[
        sanitized_report_config(args),
        results,
        chatcore_metric,
    ])


def build_parser():
    """Build the command-line parser for remote chat evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a vLLM-hosted chat model",
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
        '-a',
        '--task-name',
        default=None,
        help='Task name (default: all tasks; use | to separate multiple tasks)',
    )
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument(
        '-x',
        '--max-problems',
        type=int,
        default=None,
        help='Maximum problems per task',
    )
    parser.add_argument(
        '--chat-template-kwargs',
        type=parse_json_object,
        default=None,
        help='JSON object passed to the server-side chat template',
    )
    parser.add_argument(
        '--chat-details-dir',
        type=Path,
        default=None,
        help='Optional directory for per-example chat evaluation JSONL files',
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
