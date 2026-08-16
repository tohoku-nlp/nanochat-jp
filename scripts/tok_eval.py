"""
Evaluate compression ratio of the tokenizer.
"""

from tqdm import tqdm

from nanochat.tokenizer import get_tokenizer, RustBPETokenizer
from nanochat.dataset import parquets_iter_batched

# Japanese news text
news_text = r"""
（東京 2026年6月8日）気象庁は8日、西日本から東日本にかけての広い範囲が梅雨入りしたとみられると発表した。平年より2日早く、昨年より1日遅い梅雨入りとなる。向こう一週間は前線の影響で曇りや雨の日が多く、特に九州南部では雷を伴った激しい雨の降るおそれがあるという。

気象庁は、土砂災害や河川の増水、低い土地の浸水に警戒するよう呼びかけている。一方、農業関係者からは長雨による日照不足を心配する声も上がっている。県内のある農家は「先月の高温で生育が早まったところに長雨が重なると、品質への影響が避けられない」と話した。専門家は、こまめな排水対策と病害の早期発見が重要だと指摘している。
""".strip()

# English text (to test non-Japanese compression)
english_text = r"""
The rapid rise of open-source language models has changed how researchers think about the cost of training. A decade ago, building a capable model demanded an industrial-scale lab and a budget of many millions of dollars. Today a single researcher can train a small but genuinely useful model in a few hours on a handful of GPUs, and then share both the weights and the full training recipe with anyone who is curious. This shift has lowered the barrier to entry and invited a much wider community to study, criticize, and improve these systems in the open.
""".strip()

# Code with Japanese comments
code_text = r"""
import re

def normalize_text(text: str) -> str:
    # 全角スペースを半角スペースに置き換える
    text = text.replace("　", " ")
    # 連続する空白文字をひとつにまとめる
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class WordCounter:
    # 単語の出現回数を数える簡単なクラス

    def __init__(self):
        self.counts = {}  # 単語 -> 出現回数 の辞書

    def add(self, word: str) -> None:
        # 未登録の単語は 0 から数え始める
        self.counts[word] = self.counts.get(word, 0) + 1

    def most_common(self, n: int = 5):
        # 出現回数の多い順に上位 n 件を返す
        return sorted(self.counts.items(), key=lambda kv: kv[1], reverse=True)[:n]
""".strip()

# Math text with Japanese explanation
math_text = r"""
定理（立方和の公式）：任意の自然数 $n$ に対して、次の等式が成り立つ。
\[
\sum_{k=1}^{n} k^{3} = \left( \frac{n(n+1)}{2} \right)^{2}.
\]

証明は数学的帰納法による。まず $n = 1$ のとき、左辺は $1$、右辺は $(1 \cdot 2 / 2)^2 = 1$ となり、等式は成立する。
次に、ある $n$ で等式が成り立つと仮定する。このとき
\[
\sum_{k=1}^{n+1} k^{3} = \left( \frac{n(n+1)}{2} \right)^{2} + (n+1)^{3}
= \left( \frac{(n+1)(n+2)}{2} \right)^{2}
\]
が成り立つので、$n+1$ のときも等式は成立する。したがって、すべての自然数 $n$ について成り立つ。

この公式は、「$1$ から $n$ までの立方の和は、$n$ 番目の三角数の平方に等しい」ことを意味している。
""".strip()

# Japanese science text
science_text = r"""
光合成は、植物・藻類・シアノバクテリアが光エネルギーを化学エネルギーに変換する反応である。葉緑体のチラコイド膜に存在する光化学系IIと光化学系Iが光子を吸収し、水を分解して電子を取り出す。この電子伝達の過程でチラコイド膜を挟んだプロトンの濃度勾配が形成され、ATP合成酵素がこれを利用してATPを合成する。

こうして得られたATPとNADPHは、ストロマで進行するカルビン回路を駆動するために使われる。カルビン回路では、二酸化炭素がリブロース1,5-ビスリン酸に固定され、一連の酵素反応を経て還元・再生される。最終的に三炭糖を経て糖が合成され、生態系全体の一次生産を支えている。
""".strip()

# The tokenizer was trained on data from earlier shards, so it has seen this data
with tqdm(total=2, desc="Loading dataset samples", unit="split", dynamic_ncols=True) as progress:
    progress.set_postfix_str("train")
    train_docs = next(parquets_iter_batched(split="train"))
    train_text = "\n".join(train_docs)
    progress.update()

    progress.set_postfix_str("val")
    val_docs = next(parquets_iter_batched(split="val"))
    val_text = "\n".join(val_docs)
    progress.update()

all_text = [
    ("news", news_text),
    ("english", english_text),
    ("code", code_text),
    ("math", math_text),
    ("science", science_text),
    ("fwe-train", train_text),
]
if val_text:
    all_text.append(("fwe-val", val_text))

# Try out current default compared to GPT-2 and GPT-4 tokenizers
tokenizer_results = {}
vocab_sizes = {}

tokenizer_names = ["gpt2", "gpt4", "ours"]
tokenizer_progress = tqdm(
    tokenizer_names,
    desc="Evaluating tokenizers",
    unit="tokenizer",
    dynamic_ncols=True,
)
for tokenizer_name in tokenizer_progress:
    tokenizer_progress.set_postfix_str(f"loading {tokenizer_name}")

    if tokenizer_name == "gpt2":
        tokenizer = RustBPETokenizer.from_pretrained("gpt2") # gpt-2 base model tokenizer
    elif tokenizer_name == "gpt4":
        tokenizer = RustBPETokenizer.from_pretrained("cl100k_base") # gpt-4 base model tokenizer
    else:
        tokenizer = get_tokenizer()

    vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
    tokenizer_results[tokenizer_name] = {}

    text_progress = tqdm(
        all_text,
        desc=f"Encoding with {tokenizer_name}",
        unit="text",
        leave=False,
        dynamic_ncols=True,
    )
    for name, text in text_progress:
        text_progress.set_postfix_str(name)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        encoded_bytes = text.encode('utf-8')
        ratio = len(encoded_bytes) / len(encoded)
        tokenizer_results[tokenizer_name][name] = {
            'bytes': len(encoded_bytes),
            'tokens': len(encoded),
            'ratio': ratio
        }
    tokenizer_progress.set_postfix_str(f"completed {tokenizer_name}")

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

# Print vocab sizes
print(f"\nVocab sizes:")
print(f"GPT-2: {vocab_sizes['gpt2']}")
print(f"GPT-4: {vocab_sizes['gpt4']}")
print(f"Ours: {vocab_sizes['ours']}")

def print_comparison(baseline_name, baseline_results, ours_results, all_text):
    """Print comparison table between baseline tokenizer and ours."""
    print(f"\nComparison with {baseline_name}:")
    print("=" * 95)
    print(f"{'Text Type':<10} {'Bytes':<8} {baseline_name:<15} {'Ours':<15} {'Relative':<12} {'Better':<10}")
    print(f"{'':10} {'':8} {'Tokens':<7} {'Ratio':<7} {'Tokens':<7} {'Ratio':<7} {'Diff %':<12}")
    print("-" * 95)

    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]

        # Calculate relative difference (positive means ours is better, negative means worse)
        # Using tokens: fewer tokens is better, so we calculate (baseline_tokens - ours_tokens) / baseline_tokens
        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100

        # Determine which has better compression (higher ratio = better)
        if baseline_data['ratio'] > ours_data['ratio']:
            baseline_color, ours_color = GREEN, RED
            better = baseline_name
            diff_color = RED
        elif ours_data['ratio'] > baseline_data['ratio']:
            baseline_color, ours_color = RED, GREEN
            better = "Ours"
            diff_color = GREEN
        else:
            baseline_color, ours_color = "", ""
            better = "Tie"
            diff_color = ""

        print(f"{name:<10} {baseline_data['bytes']:<8} "
              f"{baseline_color}{baseline_data['tokens']:<7}{RESET} "
              f"{baseline_color}{baseline_data['ratio']:<7.2f}{RESET} "
              f"{ours_color}{ours_data['tokens']:<7}{RESET} "
              f"{ours_color}{ours_data['ratio']:<7.2f}{RESET} "
              f"{diff_color}{relative_diff:+7.1f}%{RESET}     "
              f"{better:<10}")

# Print comparisons
print_comparison("GPT-2", tokenizer_results['gpt2'], tokenizer_results['ours'], all_text)
print_comparison("GPT-4", tokenizer_results['gpt4'], tokenizer_results['ours'], all_text)

# Log to report
from nanochat.report import get_report
lines = []
for baseline_name in ["GPT-2", "GPT-4"]:
    baseline_key = baseline_name.lower().replace('-', '')
    baseline_results = tokenizer_results[baseline_key]
    ours_results = tokenizer_results['ours']
    lines.append(f"### Comparison with {baseline_name}")
    lines.append("")
    lines.append("| Text Type | Bytes | " + baseline_name + " Tokens | " + baseline_name + " Ratio | Ours Tokens | Ours Ratio | Relative Diff % |")
    lines.append("|-----------|-------|--------------|--------------|-------------|------------|-----------------|")
    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]
        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100
        lines.append(f"| {name} | {baseline_data['bytes']} | {baseline_data['tokens']} | {baseline_data['ratio']:.2f} | {ours_data['tokens']} | {ours_data['ratio']:.2f} | {relative_diff:+.1f}% |")
    lines.append("")
report_markdown = "\n".join(lines)
get_report().log(section="Tokenizer evaluation", data=[
    report_markdown,
])
