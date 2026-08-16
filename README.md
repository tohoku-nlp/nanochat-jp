# nanochat-jp

[karpathy/nanochat](https://github.com/karpathy/nanochat) の日本語版フォークです．トークナイザ学習・事前学習・SFT・RL・評価・推論を単一リポジトリで完結させます．
上流からの主な変更点は，SentencePiece Unigram トークナイザ，日本語コーパスと日本語 SFT/RL データ，日本語ベンチマーク（CORE / ChatCORE），HuggingFace 形式への変換器です．

学習・評価は `runs/` 以下のシェルスクリプトから実行します．

## 実行環境

動作検証は NGC の PyTorch コンテナ [`nvcr.io/nvidia/pytorch:26.05-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) 上で実施しています．

## セットアップ

```bash
export NANOCHAT_BASE_DIR=/work/outputs/nanochat   # 全成果物の出力先（未設定だと ~/.cache/nanochat）
bash runs/prepare.sh
```

`runs/prepare.sh` ではuvによる環境構築とデータセットのダウンロードを実施します．


| パス | 内容 | Hugging Face |
|------|------|--------------|
| `datasets/nanochat-jp-pretrain/v3/` | 事前学習コーパスの parquet | [tohoku-nlp/nanochat-jp-pretrain](https://huggingface.co/datasets/tohoku-nlp/nanochat-jp-pretrain) |
| `datasets/nanochat-jp-sft/v1/` | SFT データ（既定の混合は `scripts/chat_sft.py` を参照） | [tohoku-nlp/nanochat-jp-sft](https://huggingface.co/datasets/tohoku-nlp/nanochat-jp-sft) |
| `datasets/nanochat-jp-rl/v0/` | RL データ | [tohoku-nlp/nanochat-jp-rl](https://huggingface.co/datasets/tohoku-nlp/nanochat-jp-rl) |
| `eval_bundle/` | 評価データ（CORE は `core.yaml`，Chat は `eval_data/{jmmlu,jamcqa,pfgen,yomi_bench}/`） | [tohoku-nlp/nanochat-jp-eval-bundle](https://huggingface.co/datasets/tohoku-nlp/nanochat-jp-eval-bundle) |


W&B を使う場合は `.env` に `WANDB_API_KEY` と `WANDB_ENTITY` を記載し，run 名を `WANDB_RUN` で渡してください．


## 実行

| スクリプト | 内容 |
|-----------|------|
| `runs/prepare.sh` | `.venv` の構築とデータセット・eval bundle の取得 |
| `runs/speedrun.sh` | トークナイザ学習 → 事前学習 → SFT →（RL）|
| `runs/base_eval.sh` | ベースモデルの日本語タスク評価 |
| `runs/chat_eval.sh` | チャットモデルの日本語タスク評価 |
| `runs/convert_ckpt_to_hf.sh` | チェックポイントを HuggingFace 形式へ変換 |
| `runs/vllm_core_eval.sh`，`runs/vllm_chat_eval.sh` | vLLM でホストした外部モデルを同じ評価コードで測定 |

### 学習

```bash
bash runs/speedrun.sh
```

`runs/speedrun.sh` の既定は `--depth=22`，`--nproc_per_node=4`，事前学習 `--device-batch-size=32`，SFT `--device-batch-size=16` です．


九州大学の玄界や東京科学大学のTSUBAMEなどの，日本国内の教養算機環境のよくある構成である
1ノードH100 (96 GBメモリ) x 4での実行を想定しています．


### 評価

```bash
bash runs/base_eval.sh   # CORE / bpb / サンプル生成
bash runs/chat_eval.sh   # JMMLU, JamC-QA, PFGen, YOMI-Bench（生成／分類）
```

### 対話

```bash
source .venv/bin/activate
python -m scripts.chat_cli -i sft -t 0.6          # CLI
python -m scripts.base_cli  # ベースモデルの続き生成
```

### HuggingFace 形式への変換

```bash
bash runs/convert_ckpt_to_hf.sh <INPUT_DIR> <OUTPUT_BASE_DIR> <STEP>
```

### ベンチマークの出典

CORE（`$NANOCHAT_BASE_DIR/eval_bundle/core.yaml`）:

| タスク | 出典 | ライセンス |
|--------|------|-----------|
| JAQKET（v2.0 test，closed-book QA として利用） | [kumapo/JAQKET](https://huggingface.co/datasets/kumapo/JAQKET)（原典: [AI王](https://sites.google.com/view/project-aio/dataset)） | CC BY-SA 4.0 |
| COPA-ja | [nlp-titech/copa-japanese](https://github.com/nlp-titech/copa-japanese) | BSD 2-Clause |
| Global-PIQA（parallel / nonparallel 日本語部分） | [mrlbenchmarks/global-piqa-parallel](https://huggingface.co/datasets/mrlbenchmarks/global-piqa-parallel)，[同 nonparallel](https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel) | CC BY-SA 4.0 |
| JCommonsenseQA，JSQuAD | [yahoojapan/JGLUE](https://github.com/yahoojapan/JGLUE) | CC BY-SA 4.0 |

Chat（`nanochat/chat_eval_common.py`）:

| タスク | 出典 | ライセンス |
|--------|------|-----------|
| JMMLU | [nlp-waseda/JMMLU](https://github.com/nlp-waseda/JMMLU) | CC BY-SA 4.0（CC BY-NC-ND の `JMMLU_NC_ND` 科目は不使用） |
| JamC-QA | [sbintuitions/JamC-QA](https://huggingface.co/datasets/sbintuitions/JamC-QA) | CC BY-SA 4.0 |
| PFGen | [pfnet-research/pfgen-bench](https://github.com/pfnet-research/pfgen-bench) | Apache-2.0（採点コードを `tasks/pfgen_bench/` に同梱） |
| YOMI-Bench（生成／分類） | [benchmark-release/YOMI-Bench](https://github.com/benchmark-release/YOMI-Bench) | CC BY-SA 4.0 |

いずれも元データを nanochat の評価形式へ変換して利用しています．

## ライセンス

本リポジトリの主ライセンスは [Apache License 2.0](LICENSE) です．nanochat-jp 独自のコードと変更部分は，同ライセンスで提供します．

[karpathy/nanochat](https://github.com/karpathy/nanochat) 由来の部分，OpenAI HumanEval 由来の実行コード，modded-nanogpt 由来の最適化コードは，それぞれ元の MIT ライセンスと著作権表示を保持します．`tasks/pfgen_bench/` に同梱した pfgen-bench 由来のコードは Apache License 2.0 です．詳細は [NOTICE](NOTICE)，[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，`LICENSES/` を参照してください．

学習データセット，評価データ，モデル成果物には，本リポジトリのライセンスは適用されません．それぞれのライセンスと利用条件に従ってください．


## 謝辞
本リポジトリのベースとなった [karpathy/nanochat](https://github.com/karpathy/nanochat) に感謝申し上げます．
また，本プロジェクトの推進にあたり，多方面でご協力を賜りました[Tohoku NLP Group](https://www.nlp.ecei.tohoku.ac.jp/)の皆様に心より御礼申し上げます．
