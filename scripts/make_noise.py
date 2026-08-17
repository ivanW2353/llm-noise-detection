"""Build 6 training datasets from databricks-dolly-15k with controlled noise.

Datasets (all at the same noise ratio, default 10%):
  clean       - original data (baseline)
  garbled     - 10% samples corrupted with mojibake/garbage characters
  duplicate   - 10% extra rows that are exact copies
  unrelated   - 10% samples whose response is fluent/correct but from a
                different category (contextually unrelated)
  keyword     - 10% samples with only key words (entities/numbers) replaced
  mixed       - 10% total noise, evenly split among the four types

Every row carries a noise label for later detection analysis.
Sample order is identical across datasets (fixed seed) except appended
duplicate copies, so per-sample metrics are directly comparable.
"""

import argparse
import json
import os
import random
import re
import time

import yaml


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)
from datasets import load_dataset

# ----------------------------------------------------------------------------
# garbled: curated "mojibake" character pool
# ----------------------------------------------------------------------------
GARBAGE_CHARS = (
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇ"
    "１２３４５６７８９０！＠＃＄％＾＆＊（）"
    "ъѫѭӓԉ҈ԆӜԒԨԬ"
    "のをんアイウエオカキクケコ"
    "㊙㊗㋡㍿㍿ⅫⅪⅧ↯⌘℮№℗"
    "乱码测试无关联内容"
    "¤€£¥§¶©®°±×÷"
)

TEMPLATE_ANSWER = "The answer to this question is 42."

# light-paraphrase synonym bank for near_duplicate (WordNet fallback)
_SYNONYMS = {
    "great": ["excellent", "outstanding", "superb"],
    "important": ["significant", "crucial", "essential"],
    "good": ["fine", "great", "solid"],
    "big": ["large", "huge", "massive"],
    "small": ["little", "tiny", "compact"],
    "fast": ["quick", "rapid", "swift"],
    "use": ["utilize", "employ", "apply"],
    "make": ["create", "produce", "build"],
    "show": ["demonstrate", "display", "reveal"],
    "start": ["begin", "commence", "initiate"],
}


def _synonym(word, rng):
    if word in _SYNONYMS:
        return rng.choice(_SYNONYMS[word])
    try:
        from nltk.corpus import wordnet as wn
        syns = wn.synsets(word)
        if syns:
            lemmas = [l for s in syns[:2] for l in s.lemma_names()
                      if "_" not in l and l.lower() != word]
            if lemmas:
                return rng.choice(lemmas)
    except Exception:
        pass
    return word


def paraphrase(text, rng, word_prob=0.15, swap_prob=0.35):
    """Light paraphrase: semantics preserved, surface wording changed."""
    import re as _re
    # 1. sentence-level: swap adjacent sentences with some probability
    sentences = _re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) >= 2:
        for k in range(len(sentences) - 1):
            if rng.random() < swap_prob:
                sentences[k], sentences[k + 1] = sentences[k + 1], sentences[k]
    # 2. word-level: replace content words with synonyms
    out_sents = []
    for sent in sentences:
        tokens = _re.split(r"(\s+)", sent)   # keep whitespace tokens
        for i, tok in enumerate(tokens):
            word = tok.strip(".,!?;:'\"()")
            if word.isalpha() and len(word) > 3 and rng.random() < word_prob:
                s = _synonym(word.lower(), rng)
                if s != word.lower():
                    tokens[i] = s + tok[len(word):] if tok.startswith(word) else s
        out_sents.append("".join(tokens))
    return " ".join(out_sents)

PERSON_NAMES = [
    "Jonathan Miller", "Amanda Chen", "Robert Blackwell", "Sofia Reyes",
    "David Okafor", "Emily Zhang", "Marcus Johnson", "Lena Petrova",
    "Oliver Hughes", "Priya Sharma", "Daniel Kowalski", "Hannah Fischer",
    "Thomas Nguyen", "Isabella Rossi", "Samuel Osei", "Grace Kim",
    "Lucas Moreau", "Ava Patel", "Nathan Brooks", "Zoe Lindqvist",
]

ORGS = [
    "Acme Corporation", "Global Dynamics", "Vertex Industries",
    "Northbridge Group", "Helios Technologies", "Crestline Partners",
    "Falcon Systems", "Meridian Health", "Oakwood Holdings", "Zenith Motors",
]

CITIES = [
    "Springfield", "Riverdale", "Fairview", "Lakewood", "Cedar Falls",
    "Kingston", "Ashford", "Brookhaven", "Maple Grove", "Harbor City",
]


def make_user_content(instruction, context):
    if context:
        return f"Task: {instruction}\n\n{context}\n\nAnswer:"
    return f"Task: {instruction}\n\nAnswer:"


def corrupt_text(text, rng, replace_prob=0.12, insert_prob=0.03, swap_prob=0.02):
    """Character-level mojibake corruption. Keeps whitespace for structure."""
    chars = list(text)
    out = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch.isspace():
            out.append(ch)
            i += 1
            continue
        r = rng.random()
        if r < replace_prob:
            out.append(rng.choice(GARBAGE_CHARS))
        elif r < replace_prob + insert_prob:
            out.append(ch)
            out.append(rng.choice(GARBAGE_CHARS))
        elif r < replace_prob + insert_prob + swap_prob and i + 1 < len(chars):
            nxt = chars[i + 1]
            out.append(nxt)
            out.append(ch)
            i += 1
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def replace_keywords(text, rng):
    """Replace only key words: numbers/dates and capitalized proper nouns.
    Grammar and fluency are preserved; semantics are altered."""
    def repl_num(m):
        digits = m.group(0)
        if re.fullmatch(r"\d{4}", digits):  # year
            return str(rng.randint(1950, 2023))
        n = len(digits)
        return "".join(str(rng.randint(0, 9)) for _ in range(n))

    text = re.sub(r"\d+", repl_num, text)
    text = re.sub(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b",
        lambda m: rng.choice(PERSON_NAMES + ORGS + CITIES),
        text,
        count=4,
    )
    return text


def build(config, with_extra=False):
    seed = config["noise"]["seed"]
    ratio = config["noise"]["ratio"]
    data_root = config["paths"]["data_root"]
    tag = config["paths"].get("experiment_tag", "")
    rng = random.Random(seed)

    print("loading dolly-15k ...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = []
    for i, ex in enumerate(ds):
        user = make_user_content(ex["instruction"], ex["context"])
        rows.append({
            "src_id": f"dolly_{i}",
            "category": ex["category"],
            "user": user,
            "assistant": ex["response"],
        })

    # same order across all datasets; first 400 rows are a CLEAN held-out set
    order = list(range(len(rows)))
    rng.shuffle(order)
    ordered = [rows[i] for i in order]

    n_holdout = config["train"]["ref_samples"] + config["train"]["heldout_samples"]
    heldout_rows = [ordered[j] for j in range(n_holdout)]
    train_rows = ordered[n_holdout:]
    n_train = len(train_rows)
    n_noise = int(round(n_train * ratio))
    noise_idx = set(rng.sample(range(n_train), n_noise))
    log(f"total={len(rows)} holdout={n_holdout} train={n_train} noise={n_noise} ({ratio:.0%})")

    out_dir = os.path.join(data_root, "data", tag)
    os.makedirs(out_dir, exist_ok=True)

    def emit(dataset_name, row_items):
        t0 = time.time()
        path = os.path.join(out_dir, dataset_name, "train.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for k, item in enumerate(row_items):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                if k and k % 5000 == 0:
                    log(f"  [{dataset_name}] {k}/{len(row_items)} rows")
        log(f"{dataset_name}: {len(row_items)} rows in {time.time()-t0:.1f}s -> {path}")

    def base_item(meta, sample_id):
        return {
            "id": f"{meta['src_id']}_{sample_id}",
            "sample_id": sample_id,
            "src_id": meta["src_id"],
            "category": meta["category"],
            "noise_label": 0,
            "noise_type": "none",
            "messages": [
                {"role": "user", "content": meta["user"]},
                {"role": "assistant", "content": meta["assistant"]},
            ],
        }

    # shared CLEAN held-out set (ref direction + held-out eval loss)
    with open(os.path.join(out_dir, "heldout.jsonl"), "w") as f:
        for j, r in enumerate(heldout_rows):
            f.write(json.dumps(base_item(r, j), ensure_ascii=False) + "\n")
    log(f"heldout: {len(heldout_rows)} clean rows -> {os.path.join(out_dir, 'heldout.jsonl')}")

    # 1. clean
    emit("clean", [base_item(r, i) for i, r in enumerate(train_rows)])

    # 2. garbled
    garbled = []
    for i, r in enumerate(train_rows):
        it = base_item(r, i)
        if i in noise_idx:
            it["noise_label"] = 1
            it["noise_type"] = "garbled"
            it["messages"][0]["content"] = corrupt_text(it["messages"][0]["content"], rng)
            it["messages"][1]["content"] = corrupt_text(it["messages"][1]["content"], rng)
        garbled.append(it)
    emit("garbled", garbled)

    # 3. duplicate (extra exact-copy rows appended; noise applied to copies)
    dup_meta = [r for i, r in enumerate(train_rows) if i in noise_idx]
    duplicates = []
    for k, r in enumerate(dup_meta):
        it = base_item(r, n_train + k)
        it["id"] = it["id"].replace("_dolly", "_copy_dolly")
        it["noise_label"] = 1
        it["noise_type"] = "duplicate"
        duplicates.append(it)
    emit("duplicate", [base_item(r, i) for i, r in enumerate(train_rows)] + duplicates)

    # 4. unrelated: response from a different category sample
    by_cat = {}
    for r in train_rows:
        by_cat.setdefault(r["category"], []).append(r)
    unrelated = []
    for i, r in enumerate(train_rows):
        it = base_item(r, i)
        if i in noise_idx:
            pool = by_cat[r["category"]]
            other = r
            tries = 0
            while other is r or other["category"] == r["category"]:
                other = rng.choice(train_rows)
                tries += 1
                if tries > 50:
                    break
            it["noise_label"] = 1
            it["noise_type"] = "unrelated"
            it["messages"][1]["content"] = other["assistant"]
            it["src_category"] = other["category"]
        unrelated.append(it)
    emit("unrelated", unrelated)

    # 5. keyword
    keyword = []
    for i, r in enumerate(train_rows):
        it = base_item(r, i)
        if i in noise_idx:
            it["noise_label"] = 1
            it["noise_type"] = "keyword"
            it["messages"][0]["content"] = replace_keywords(it["messages"][0]["content"], rng)
            it["messages"][1]["content"] = replace_keywords(it["messages"][1]["content"], rng)
        keyword.append(it)
    emit("keyword", keyword)

    # 5b. template (optional, --with-extra): consistent-pattern noise.
    # dynanoise Noise E / qa-noise fixed_wrong showed this family is the most
    # harmful AND detectable only via consistency/IFD signals.
    template = []
    if with_extra:
        for i, r in enumerate(train_rows):
            it = base_item(r, i)
            if i in noise_idx:
                it["noise_label"] = 1
                it["noise_type"] = "template"
                it["messages"][1]["content"] = TEMPLATE_ANSWER
            template.append(it)
        emit("template", template)

    # 5c. truncation (optional, --with-extra): information LOSS noise.
    # The response is cut off at ~50% (mid-sentence); the missing half has
    # no label tokens, so training dynamics differ from all error-type noises.
    truncation = []
    if with_extra:
        for i, r in enumerate(train_rows):
            it = base_item(r, i)
            if i in noise_idx:
                it["noise_label"] = 1
                it["noise_type"] = "truncation"
                content = it["messages"][1]["content"]
                cut = max(1, int(len(content) * 0.5))
                it["messages"][1]["content"] = content[:cut]
            truncation.append(it)
        emit("truncation", truncation)

    # 5d. near_duplicate (optional, --with-extra): light paraphrase.
    # Semantic content identical, surface wording changed (WordNet synonyms,
    # adjacent-sentence swaps, digit changes) - mimics content-farm reprints.
    near_duplicate = []
    if with_extra:
        for i, r in enumerate(train_rows):
            it = base_item(r, i)
            if i in noise_idx:
                it["noise_label"] = 1
                it["noise_type"] = "near_duplicate"
                it["messages"][1]["content"] = paraphrase(it["messages"][1]["content"], rng)
            near_duplicate.append(it)
        emit("near_duplicate", near_duplicate)

    # 6. mixed: evenly split among all noise types, applied to disjoint subsets
    noise_idx_list = sorted(noise_idx)
    mixed_types = ["garbled", "duplicate", "unrelated", "keyword"]
    if with_extra:
        mixed_types += ["template", "truncation", "near_duplicate"]
    chunk = max(1, n_noise // len(mixed_types))
    parts = {}
    for k, t in enumerate(mixed_types):
        parts[t] = set(noise_idx_list[k * chunk:(k + 1) * chunk])
    parts[mixed_types[-1]] |= set(noise_idx_list[(len(mixed_types) - 1) * chunk:])
    mixed = []
    for i, r in enumerate(train_rows):
        it = base_item(r, i)
        for t, idxs in parts.items():
            if i in idxs:
                it["noise_label"] = 1
                it["noise_type"] = t
                if t == "garbled":
                    it["messages"][0]["content"] = corrupt_text(it["messages"][0]["content"], rng)
                    it["messages"][1]["content"] = corrupt_text(it["messages"][1]["content"], rng)
                elif t == "unrelated":
                    other = rng.choice(train_rows)
                    it["messages"][1]["content"] = other["assistant"]
                    it["src_category"] = other["category"]
                elif t == "keyword":
                    it["messages"][0]["content"] = replace_keywords(it["messages"][0]["content"], rng)
                    it["messages"][1]["content"] = replace_keywords(it["messages"][1]["content"], rng)
                elif t == "template":
                    it["messages"][1]["content"] = TEMPLATE_ANSWER
                elif t == "truncation":
                    content = it["messages"][1]["content"]
                    cut = max(1, int(len(content) * 0.5))
                    it["messages"][1]["content"] = content[:cut]
                elif t == "near_duplicate":
                    it["messages"][1]["content"] = paraphrase(it["messages"][1]["content"], rng)
        mixed.append(it)
    dup_mixed = []
    for k, r in enumerate([train_rows[i] for i in sorted(parts["duplicate"])]):
        it = base_item(r, n_train + k)
        it["id"] = it["id"].replace("_dolly", "_copy_dolly")
        it["noise_label"] = 1
        it["noise_type"] = "duplicate"
        dup_mixed.append(it)
    emit("mixed", mixed + dup_mixed)

    manifest = {
        "source": "databricks/databricks-dolly-15k",
        "noise_ratio": ratio,
        "seed": seed,
        "experiment_tag": tag,
        "n_total": len(rows),
        "n_holdout": n_holdout,
        "n_train": n_train,
        "n_noise": n_noise,
        "datasets": (["clean", "garbled", "duplicate", "unrelated", "keyword",
                      "template", "truncation", "near_duplicate", "mixed"]
                     if with_extra else
                     ["clean", "garbled", "duplicate", "unrelated", "keyword", "mixed"]),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log("manifest written: " + os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/noisedetect/config.yaml")
    ap.add_argument("--ratio", type=float, default=None, help="override noise ratio, e.g. --ratio 0.20")
    ap.add_argument("--tag", type=str, default=None, help="experiment tag (output dir suffix), e.g. --tag ratio20")
    ap.add_argument("--with-extra", action="store_true",
                    help="add 3 extra noise types: template (consistent pattern, Noise E/fixed_wrong "
                         "family), truncation (information loss), near_duplicate (light paraphrase)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.ratio:
        cfg["noise"]["ratio"] = args.ratio
    if args.tag:
        cfg["paths"]["experiment_tag"] = args.tag
    build(cfg, with_extra=args.with_extra)
