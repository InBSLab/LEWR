import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


GENERIC_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "as", "of",
    "to", "in", "on", "for", "with", "by", "at", "from", "into", "onto",
    "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "it", "its", "their", "his", "her", "our", "your",
    "we", "you", "they", "he", "she", "i", "who", "which", "what", "not",
    "no", "can", "will", "would", "should", "could", "may", "might",
    "do", "does", "did", "have", "has", "had", "ha", "one", "two", "also",
    "such", "than", "there", "here", "up", "down", "out", "over", "under",
    "more", "most", "some", "any", "all", "each", "other", "only", "own",
    "same", "just", "about", "above", "below", "between", "through",
}


GENERIC_TECH_FILLERS = {
    "api", "apis", "service", "services", "data", "using", "used", "use",
    "user", "users", "information", "allow", "allows", "allowing",
    "developer", "developers", "application", "applications", "app",
    "apps", "web", "website", "provide", "provides", "providing",
    "access", "available", "based", "include", "includes", "including",
    "support", "supports", "supporting", "platform", "system", "systems",
    "interface", "method", "methods", "call", "calls", "request",
    "requests", "response", "responses", "page", "purely", "historical",
    "research", "purpose", "purposesthe", "maintained", "longer",
    "http", "https", "json", "xml", "rest", "restful",
}

STOPWORDS = GENERIC_STOPWORDS | GENERIC_TECH_FILLERS

MIN_TOKEN_LEN = 3
TOKEN_PATTERN = re.compile(r"^[a-zA-Z]+$")


def clean_tokens(tokens: List[str]) -> List[str]:

    out = []
    for t in tokens:
        t = t.lower().strip()
        if len(t) < MIN_TOKEN_LEN:
            continue
        if not TOKEN_PATTERN.match(t):
            continue
        if t in STOPWORDS:
            continue
        out.append(t)
    return out


def _guess_provider(api_name: str) -> str:

    first = re.split(r"[-_]+", api_name.lower())[0]
    return first or api_name.lower()


def mine(
    name_path: str,
    description_path: str,
    category_path: str,
    output_path: str,
    min_apis: int = 8,
    top_k: int = 5,
    min_word_df_ratio: float = 0.03,
    min_word_df_floor: int = 2,
    generic_percentile: float = 90.0,
    min_word_provider_df: int = 2,
):
    names: List[str] = json.load(open(name_path, encoding="utf-8"))
    descs: List[List[str]] = json.load(open(description_path, encoding="utf-8"))
    cats: List[List[str]] = json.load(open(category_path, encoding="utf-8"))

    assert len(names) == len(descs) == len(cats), (
        f"三个文件长度不一致: name={len(names)}, description={len(descs)}, "
        f"category={len(cats)}，请确认它们按同一份 API 列表下标对齐。"
    )

    print(f"[mine_domain_keywords] 载入 {len(names)} 条 API 记录")


    category_word_counter: Dict[str, Counter] = defaultdict(Counter)
    category_word_df: Dict[str, Counter] = defaultdict(Counter)
    category_api_count: Dict[str, int] = defaultdict(int)

    category_word_providers: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

    for api_name, desc_tokens, cat_tags in zip(names, descs, cats):

        name_tokens = re.split(r"[-_]+", api_name)
        cleaned_name_tokens = clean_tokens(name_tokens)
        cleaned_desc_tokens = clean_tokens(desc_tokens)

        if not cat_tags:
            continue


        distinct_tokens_this_api = set(cleaned_name_tokens) | set(cleaned_desc_tokens)
        provider = _guess_provider(api_name)

        for cat in set(cat_tags):
            cat = cat.lower().strip()
            if not cat:
                continue
            category_api_count[cat] += 1

            category_word_counter[cat].update(cleaned_name_tokens)
            category_word_counter[cat].update(cleaned_name_tokens)
            category_word_counter[cat].update(cleaned_desc_tokens)
            category_word_df[cat].update(distinct_tokens_this_api)
            for tok in distinct_tokens_this_api:
                category_word_providers[cat][tok].add(provider)

    print(f"[mine_domain_keywords] 共发现 {len(category_api_count)} 个原生 category 标签")


    all_counts = sorted(category_api_count.values())
    if all_counts:
        idx = min(len(all_counts) - 1, int(math.ceil(generic_percentile / 100.0 * len(all_counts))) - 1)
        idx = max(0, idx)
        generic_threshold = all_counts[idx]
    else:
        generic_threshold = float("inf")
    print(f"[mine_domain_keywords] 第 {generic_percentile:.0f} 分位 api_count 阈值 = "
          f"{generic_threshold}（>= 此值的 category 会标记 is_generic=true）")

    result = {}
    skipped = 0
    for cat, api_count in category_api_count.items():
        if api_count < min_apis:
            skipped += 1
            continue

        counter = category_word_counter[cat]
        df_counter = category_word_df[cat]
        provider_sets = category_word_providers[cat]

        min_df = max(min_word_df_floor, math.ceil(min_word_df_ratio * api_count))


        cat_self_tokens = set(re.split(r"[-_]+", cat))


        candidates = [
            (w, tf) for w, tf in counter.items()
            if w not in cat_self_tokens
            and df_counter.get(w, 0) >= min_df
            and len(provider_sets.get(w, ())) >= min_word_provider_df
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_words = [w for w, _ in candidates[:top_k]]

        if not top_words:
            skipped += 1
            continue

        injection = f"{cat} API ({' '.join(top_words)})"
        result[cat] = {
            "injection": injection,
            "top_words": top_words,
            "api_count": api_count,
            "is_generic": api_count >= generic_threshold,
            "min_word_df_used": min_df,
            "min_word_provider_df_used": min_word_provider_df,
        }

    n_generic = sum(1 for v in result.values() if v["is_generic"])
    print(f"[mine_domain_keywords] 保留 {len(result)} 个 category "
          f"(min_apis>={min_apis})，其中 {n_generic} 个标记为 is_generic，"
          f"跳过 {skipped} 个样本不足/无有效关键词（df 过滤后）的 category")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[mine_domain_keywords] 已写出: {output_path}")


    print("\n[样例抽查]（按 api_count 降序取前 8 个）")
    for cat, info in sorted(result.items(), key=lambda x: -x[1]["api_count"])[:8]:
        tag = "GENERIC" if info["is_generic"] else "specific"
        print(f"  {cat:15s} (n={info['api_count']:4d}, {tag:8s})  -> {info['injection']}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="从 ProgrammableWeb 原生 category 元数据自动挖掘领域关键词表")
    p.add_argument("--name", default="api_name.json")
    p.add_argument("--description", default="api_description.json")
    p.add_argument("--category", default="api_category.json")
    p.add_argument("--output", default="domain_keywords_auto.json")
    p.add_argument("--min-apis", type=int, default=8,
                    help="一个 category 至少要有多少个 API 样本才会被纳入词表（默认 8）")
    p.add_argument("--top-k", type=int, default=5,
                    help="每个 category 保留多少个代表词（默认 5）")
    p.add_argument("--min-word-df-ratio", type=float, default=0.03,
                    help="代表词至少要出现在该 category 下多少比例的不同 API 里"
                         "（默认 0.03，即 3%%），用于过滤只在个别 API 里反复出现的品牌名")
    p.add_argument("--min-word-df-floor", type=int, default=2,
                    help="代表词文档频次的绝对下限（默认 2，小 category 时防止比例阈值过低）")
    p.add_argument("--generic-percentile", type=float, default=90.0,
                    help="按 api_count 分布的第几分位数标记 is_generic（默认 90，"
                         "即 api_count 排在全体 category 前 10%% 的会被标记为过于宽泛）")
    p.add_argument("--min-word-provider-df", type=int, default=2,
                    help="代表词至少要来自多少个不同的发布方（按 api_name 首段粗略"
                         "识别，默认 2），用于过滤"
                         "\"同一发布方发布多条产品线记录、每条都提自己品牌名\""
                         "导致的品牌名噪声（例如某银行的个人/企业/信用卡多个 API "
                         "都提到自己的银行名）。")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    mine(
        name_path=args.name,
        description_path=args.description,
        category_path=args.category,
        output_path=args.output,
        min_apis=args.min_apis,
        top_k=args.top_k,
        min_word_df_ratio=args.min_word_df_ratio,
        min_word_df_floor=args.min_word_df_floor,
        generic_percentile=args.generic_percentile,
        min_word_provider_df=args.min_word_provider_df,
    )
