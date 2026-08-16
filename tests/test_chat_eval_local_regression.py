import torch
import pytest

import scripts.chat_eval as chat_eval


class LocalTask:
    def __init__(self, eval_type, size=1):
        self.eval_type = eval_type
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        conversation = {
            'messages': [
                {'role': 'user', 'content': 'question'},
                {'role': 'assistant', 'content': 'B'},
            ],
        }
        if self.eval_type == 'categorical':
            conversation['letters'] = ('A', 'B')
        return conversation

    def evaluate(self, conversation, response):
        return response == 'B'


class LocalTokenizer:
    def render_for_completion(self, conversation):
        return [10, 20]

    def decode(self, token_ids):
        return 'B' if token_ids == [30] else 'prompt'

    def encode(self, text):
        return {'A': [1], 'B': [2]}[text]

    def get_bos_token_id(self):
        return 0


class LocalModel:
    def get_device(self):
        return torch.device('cpu')

    def __call__(self, prompt_ids):
        batch_size, sequence_length = prompt_ids.shape
        logits = torch.zeros(batch_size, sequence_length, 4)
        logits[:, :, 2] = 1.0
        return logits


class LocalEngine:
    def __init__(self):
        self.calls = 0

    def generate_batch(self, prompt, **kwargs):
        self.calls += 1
        return [prompt + [30]], None


class MultiSampleEngine(LocalEngine):
    def generate_batch(self, prompt, **kwargs):
        self.calls += 1
        return [prompt + [31], prompt + [30]], None


class RecordingProgress:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.updates = 0
        self.postfixes = []
        self.closed = False

    def update(self, amount):
        self.updates += amount

    def set_postfix(self, **kwargs):
        self.postfixes.append(kwargs)

    def write(self, message):
        pass

    def close(self):
        self.closed = True


def test_existing_local_generative_loop_keeps_best_sample_behavior(monkeypatch):
    monkeypatch.setattr(chat_eval, 'get_dist_info', lambda: (False, 0, 0, 1))
    score = chat_eval.run_generative_eval(
        LocalTask('generative'),
        LocalTokenizer(),
        LocalModel(),
        LocalEngine(),
        num_samples=1,
        max_new_tokens=16,
        temperature=0.0,
        top_k=50,
    )
    assert score == 1.0


def test_existing_local_categorical_loop_keeps_restricted_argmax(monkeypatch):
    monkeypatch.setattr(chat_eval, 'get_dist_info', lambda: (False, 0, 0, 1))
    score = chat_eval.run_categorical_eval(
        LocalTask('categorical'),
        LocalTokenizer(),
        LocalModel(),
        batch_size=1,
    )
    assert score == 1.0


def test_chat_eval_cli_progress_defaults_and_disable_flag():
    default_args = chat_eval.build_parser().parse_args(['--source', 'sft'])
    disabled_args = chat_eval.build_parser().parse_args([
        '--source', 'sft', '--no-progress',
    ])
    assert default_args.show_progress is True
    assert disabled_args.show_progress is False


def test_chat_eval_cli_replaces_show_decoded_with_details_dir(tmp_path):
    args = chat_eval.build_parser().parse_args([
        '--source', 'sft', '--chat-details-dir', str(tmp_path),
    ])
    assert args.chat_details_dir == tmp_path
    with pytest.raises(SystemExit):
        chat_eval.build_parser().parse_args([
            '--source', 'sft', '--show-decoded',
        ])


def test_local_model_identity_uses_source_tag_and_unpadded_step():
    args = chat_eval.build_parser().parse_args([
        '--source', 'sft', '--model-tag', 'custom-tag', '--step', '42',
    ])
    model_name, model_slug = chat_eval.resolve_local_model_identity(
        args,
        {
            'step': 42,
            'model_config': {'n_layer': 24},
            'user_config': {},
        },
    )
    assert model_name == 'sft/custom-tag (step 42)'
    assert model_slug == 'sft-custom-tag-step-42'


def test_generative_progress_counts_local_examples(monkeypatch):
    monkeypatch.setattr(chat_eval, 'get_dist_info', lambda: (False, 0, 0, 1))
    progress_instances = []

    def fake_tqdm(**kwargs):
        progress = RecordingProgress(**kwargs)
        progress_instances.append(progress)
        return progress

    monkeypatch.setattr(chat_eval, 'tqdm', fake_tqdm)
    score = chat_eval.run_generative_eval(
        LocalTask('generative', size=3),
        LocalTokenizer(),
        LocalModel(),
        LocalEngine(),
        num_samples=1,
        max_new_tokens=16,
        temperature=0.0,
        top_k=50,
        show_progress=True,
        progress_desc='Chat test',
    )

    assert score == 1.0
    assert progress_instances[0].kwargs == {
        'total': 3,
        'desc': 'Chat test',
        'unit': 'example',
        'dynamic_ncols': True,
        'disable': False,
    }
    assert progress_instances[0].updates == 3
    assert progress_instances[0].closed is True


def test_categorical_progress_is_rank_zero_assigned_work(monkeypatch):
    monkeypatch.setattr(chat_eval, 'get_dist_info', lambda: (True, 0, 0, 2))
    monkeypatch.setattr(chat_eval.dist, 'all_reduce', lambda tensor, op: None)
    progress_instances = []

    def fake_tqdm(**kwargs):
        progress = RecordingProgress(**kwargs)
        progress_instances.append(progress)
        return progress

    monkeypatch.setattr(chat_eval, 'tqdm', fake_tqdm)
    score = chat_eval.run_categorical_eval(
        LocalTask('categorical', size=5),
        LocalTokenizer(),
        LocalModel(),
        batch_size=2,
        show_progress=True,
        progress_desc='Chat test',
    )

    assert score == 1.0
    assert progress_instances[0].kwargs['total'] == 3
    assert progress_instances[0].kwargs['disable'] is False
    assert progress_instances[0].updates == 3
    assert progress_instances[0].closed is True


def test_progress_is_not_created_when_disabled(monkeypatch):
    monkeypatch.setattr(chat_eval, 'get_dist_info', lambda: (False, 0, 0, 1))
    monkeypatch.setattr(
        chat_eval,
        'tqdm',
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError('tqdm should not be created')
        ),
    )
    assert chat_eval.run_generative_eval(
        LocalTask('generative'),
        LocalTokenizer(),
        LocalModel(),
        LocalEngine(),
        num_samples=1,
        max_new_tokens=16,
        temperature=0.0,
        top_k=50,
    ) == 1.0


def test_generative_details_keep_all_samples_without_extra_inference(monkeypatch):
    monkeypatch.setattr(chat_eval, 'get_dist_info', lambda: (False, 0, 0, 1))
    engine = MultiSampleEngine()
    completed_records = []

    score, records = chat_eval.run_generative_eval_with_details(
        LocalTask('generative'),
        LocalTokenizer(),
        LocalModel(),
        engine,
        num_samples=2,
        max_new_tokens=16,
        temperature=0.0,
        top_k=50,
        detail_callback=completed_records.append,
    )

    assert score == 1.0
    assert engine.calls == 1
    assert completed_records == records
    assert records[0]['rendered_prompt'] == 'prompt'
    assert records[0]['input_token_count'] == 2
    assert records[0]['selected_sample_index'] == 1
    assert records[0]['samples'] == [
        {
            'sample_index': 0,
            'completion': 'prompt',
            'generated_token_count': 1,
            'score': 0.0,
        },
        {
            'sample_index': 1,
            'completion': 'B',
            'generated_token_count': 1,
            'score': 1.0,
        },
    ]


def test_categorical_details_keep_candidate_logprobs(monkeypatch):
    monkeypatch.setattr(chat_eval, 'get_dist_info', lambda: (False, 0, 0, 1))
    completed_records = []

    score, records = chat_eval.run_categorical_eval_with_details(
        LocalTask('categorical'),
        LocalTokenizer(),
        LocalModel(),
        batch_size=1,
        detail_callback=completed_records.append,
    )

    assert score == 1.0
    assert completed_records == records
    assert records[0]['predicted_label'] == 'B'
    assert records[0]['gold_label'] == 'B'
    assert records[0]['correct'] is True
    assert [candidate['token_id'] for candidate in records[0]['candidates']] == [
        1, 2,
    ]
    assert [candidate['logprob'] for candidate in records[0]['candidates']] == pytest.approx([
        -1.3132616, -0.31326166,
    ])


def test_normal_path_does_not_build_detail_records(monkeypatch):
    monkeypatch.setattr(chat_eval, 'get_dist_info', lambda: (False, 0, 0, 1))
    monkeypatch.setattr(
        chat_eval,
        'build_generation_detail_record',
        lambda **kwargs: pytest.fail('details should not be built'),
    )
    engine = LocalEngine()

    assert chat_eval.run_generative_eval(
        LocalTask('generative'),
        LocalTokenizer(),
        LocalModel(),
        engine,
        num_samples=1,
        max_new_tokens=16,
        temperature=0.0,
        top_k=50,
    ) == 1.0
    assert engine.calls == 1


def test_detail_records_are_gathered_and_ordered_on_rank_zero(monkeypatch):
    def fake_gather(local_records, gathered, dst):
        gathered[0] = local_records
        gathered[1] = [{'example_index': 1}]

    monkeypatch.setattr(chat_eval.dist, 'gather_object', fake_gather)
    records = chat_eval._gather_detail_records(
        [{'example_index': 2}, {'example_index': 0}],
        ddp=True,
        ddp_rank=0,
        ddp_world_size=2,
    )
    assert [record['example_index'] for record in records] == [0, 1, 2]
