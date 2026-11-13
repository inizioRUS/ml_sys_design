"""
RAG corpus EDA for Habr articles.

Install:
    pip install requests beautifulsoup4 markdownify pandas numpy matplotlib seaborn \
      scikit-learn tiktoken tqdm langdetect openai

Examples:
    python habr_rag_eda.py --ids 1095320 1061238 815309
    python habr_rag_eda.py --ids-file ids.txt --out output
    python habr_rag_eda.py --ids-file ids.txt --generate-qa --qa-count 100

ids.txt: one numeric ID or https://habr.com/ru/articles/ID/ URL per line.
Optional QA generation requires OPENAI_API_KEY. Generated QA must be reviewed manually.
Habr's /kek/v2/ endpoint is unofficial and may change. Respect site rules and content rights.
"""

import argparse
import hashlib
import json
import os
import random
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import tiktoken
from bs4 import BeautifulSoup
from langdetect import DetectorFactory, detect, LangDetectException
from markdownify import markdownify
from requests.adapters import HTTPAdapter
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from urllib3.util.retry import Retry

DetectorFactory.seed = 42
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
RNG = random.Random(42)
ENC = tiktoken.get_encoding("cl100k_base")  # approximation for the chosen RAG embedding model


def tokens(text):
    return len(ENC.encode(str(text), disallowed_special=()))


def clean_spaces(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def html_text(value):
    return clean_spaces(BeautifulSoup(str(value or ""), "html.parser").get_text(" ", strip=True))


def extract_ids(raw_items):
    result = []
    for item in raw_items:
        for part in re.split(r"[,\s]+", item.strip()):
            if not part or part.startswith("#"):
                continue
            match = re.search(r"habr\.com/(?:ru/|en/)?(?:articles|publications)/(\d+)", part)
            if match:
                result.append(int(match.group(1)))
            elif part.isdigit():
                result.append(int(part))
            else:
                print(f"[SKIP] Invalid article ID: {part}")
    return list(dict.fromkeys(result))


def session_with_retries():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; AcademicCorpusEDA/1.0; respectful-request-rate)",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    })
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], respect_retry_after_header=True)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_raw(ids, output, delay=2.0, refresh=False):
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = session_with_retries()
    failed = []
    for article_id in tqdm(ids, desc="Download articles"):
        path = raw_dir / f"{article_id}.json"
        if path.exists() and not refresh:
            continue
        try:
            url = f"https://habr.com/kek/v2/articles/{article_id}/"
            response = session.get(url, params={"fl": "ru", "hl": "ru"}, timeout=30)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and isinstance(data.get("data"), dict):
                data = data["data"]
            if not isinstance(data, dict) or not data.get("textHtml"):
                raise ValueError("No textHtml in API response")
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            failed.append({"article_id": article_id, "error": str(e)})
            print(f"[FAIL] {article_id}: {e}")
        finally:
            time.sleep(max(1.0, delay))
    pd.DataFrame(failed, columns=["article_id", "error"]).to_csv(
        output / "download_errors.csv", index=False)


def code_language(tag):
    code_tag = tag.find("code")
    classes = (code_tag.get("class", []) if code_tag else []) + tag.get("class", [])
    for cls in classes:
        if cls.startswith("language-"):
            return cls.removeprefix("language-")
    return ""


def article_to_record(data, md_dir):
    article_id = str(data["id"])
    body_html = data.get("textHtml") or ""
    soup = BeautifulSoup(body_html, "html.parser")
    for bad in soup.select("script, style, noscript"):
        bad.decompose()

    md_body = markdownify(str(soup), heading_style="ATX", code_language_callback=code_language,
                          bullets="-", strip=["img"]).strip()
    title = html_text(data.get("titleHtml", ""))
    md = f"# {title}\n\n{md_body}\n"
    (md_dir / f"{article_id}.md").write_text(md, encoding="utf-8")

    content = soup.get_text(" ", strip=True)
    code_text = " ".join(pre.get_text(" ", strip=True) for pre in soup.find_all("pre"))
    hub_names = [html_text(h.get("titleHtml") or h.get("title")) for h in data.get("hubs", []) if isinstance(h, dict)]
    tag_names = [html_text(t.get("titleHtml") or t.get("title")) for t in data.get("tags", []) if isinstance(t, dict)]
    stats = data.get("statistics") or {}
    author = data.get("author") or {}
    try:
        inferred_lang = detect(content[:10000]) if len(content) > 100 else "unknown"
    except LangDetectException:
        inferred_lang = "unknown"
    return {
        "article_id": article_id,
        "url": f"https://habr.com/ru/articles/{article_id}/",
        "title": title,
        "author": author.get("alias", ""),
        "date": data.get("timePublished"),
        "api_lang": data.get("lang", ""),
        "detected_lang": inferred_lang,
        "complexity": data.get("complexity"),
        "format": data.get("format"),
        "hubs": hub_names,
        "tags": tag_names,
        "primary_hub": hub_names[0] if hub_names else "Без хаба",
        "views": stats.get("readingCount"),
        "favorites": stats.get("favoritesCount"),
        "score": stats.get("score"),
        "reading_time": data.get("readingTime"),
        "content": content,
        "markdown": md,
        "chars": len(content),
        "words": len(re.findall(r"\b\w+\b", content, re.UNICODE)),
        "tokens": tokens(md),
        "paragraphs": len(soup.find_all("p")),
        "headings": len(soup.find_all(re.compile("^h[1-6]$"))),
        "code_blocks": len(soup.find_all("pre")),
        "code_chars": len(code_text),
        "tables": len(soup.find_all("table")),
        "images": len(soup.find_all("img")),
        "links": len(soup.find_all("a")),
        "lists": len(soup.find_all(["ul", "ol"])),
        "code_share": round(len(code_text) / max(len(content), 1), 4),
        "sha256": hashlib.sha256(clean_spaces(content).lower().encode("utf-8")).hexdigest(),
    }


def load_corpus(output):
    md_dir = output / "articles"
    md_dir.mkdir(exist_ok=True, parents=True)
    records = []
    for path in tqdm(sorted((output / "raw").glob("*.json")), desc="Parse articles"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data.get("data"), dict):
                data = data["data"]
            records.append(article_to_record(data, md_dir))
        except Exception as exc:
            print(f"[PARSE ERROR] {path.name}: {exc}")
    if not records:
        raise RuntimeError("No articles downloaded. Provide valid IDs and check download_errors.csv.")
    df = pd.DataFrame(records).drop_duplicates("article_id").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["code_article"] = df["code_blocks"] > 0
    df["has_table"] = df["tables"] > 0
    df["has_images"] = df["images"] > 0
    df["short_article"] = df["tokens"] < 256
    df["empty_article"] = df["chars"] < 50
    df["exact_duplicate"] = df.duplicated("sha256", keep=False)
    export = df.drop(columns=["markdown", "content"])
    export.to_json(output / "articles_meta.jsonl", orient="records", lines=True, force_ascii=False, date_format="iso")
    export.assign(hubs=export["hubs"].map(lambda x: ", ".join(x)),
                  tags=export["tags"].map(lambda x: ", ".join(x))).to_csv(output / "articles_meta.csv", index=False)
    return df


def savefig(folder, name):
    plt.tight_layout()
    plt.savefig(folder / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_hist(df, field, title, folder, log_x=False):
    values = df[field].dropna().astype(float)
    values = values[values > 0] if log_x else values
    if values.empty:
        return
    plt.figure(figsize=(10, 5))
    sns.histplot(x=values, bins=min(40, max(5, len(values) // 2)), color="steelblue")
    if log_x:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(field)
    savefig(folder, field + ("_log" if log_x else ""))


def plot_bar(series, title, folder, filename, top=20):
    series = series.dropna().head(top)
    if series.empty:
        return
    plt.figure(figsize=(11, max(4, len(series) * 0.37)))
    ax = sns.barplot(x=series.values, y=series.index.astype(str), orient="h", color="steelblue")
    ax.set(title=title, xlabel="Количество", ylabel="")
    savefig(folder, filename)


def near_duplicate_pairs(df, max_articles=1500, threshold=0.83):
    sample = df.loc[df["chars"] >= 100].sort_values("article_id").head(max_articles)
    if len(sample) < 2:
        return pd.DataFrame(columns=["article_id_a", "article_id_b", "cosine_similarity"])
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                                 max_features=30000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(sample["content"].str[:20000])
    nn = NearestNeighbors(n_neighbors=min(4, len(sample)), metric="cosine", algorithm="brute")
    nn.fit(matrix)
    dist, idx = nn.kneighbors(matrix)
    pairs = {}
    ids = sample["article_id"].to_numpy()
    for i in range(len(sample)):
        for d, j in zip(dist[i], idx[i]):
            if i != j and 1 - d >= threshold:
                a, b = sorted([str(ids[i]), str(ids[j])])
                pairs[(a, b)] = max(pairs.get((a, b), 0.0), float(1 - d))
    return pd.DataFrame([{"article_id_a": a, "article_id_b": b,
                          "cosine_similarity": round(v, 4)} for (a, b), v in pairs.items()],
                        columns=["article_id_a", "article_id_b", "cosine_similarity"])


def topic_map(df, folder):
    sample = df[df["chars"] > 100].sort_values("article_id").head(1500)
    if len(sample) < 5:
        return
    vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), max_df=0.95,
                                 min_df=1, sublinear_tf=True)
    x = vectorizer.fit_transform(sample["content"].str[:12000])
    if min(x.shape) < 3:
        return
    z = TruncatedSVD(n_components=2, random_state=42).fit_transform(x)
    hub_counts = sample["primary_hub"].value_counts()
    labels = sample["primary_hub"].where(sample["primary_hub"].isin(hub_counts.head(8).index), "Другие")
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=labels, alpha=0.7, s=35)
    plt.xlabel("LSA component 1")
    plt.ylabel("LSA component 2")
    plt.title("Тематическая карта статей (TF-IDF + TruncatedSVD)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    savefig(folder, "topic_map_lsa")


def corpus_eda(df, output):
    folder = output / "plots" / "corpus"
    folder.mkdir(exist_ok=True, parents=True)
    cols = ["chars", "words", "tokens", "paragraphs", "headings", "code_blocks",
            "code_share", "tables", "images", "links", "lists", "views", "favorites"]
    df[cols].describe(percentiles=[.25, .5, .75, .9, .95, .99]).T.to_csv(output / "corpus_describe.csv")

    for field in ["chars", "tokens", "code_share", "headings", "code_blocks", "images", "tables"]:
        plot_hist(df, field, f"Распределение: {field}", folder, log_x=(field in ["chars", "tokens"]))

    if df["date"].notna().any():
        monthly = df.dropna(subset=["date"]).groupby(df["date"].dt.to_period("M")).size()
        monthly.index = monthly.index.astype(str)
        monthly.to_csv(output / "articles_by_month.csv", header=["n_articles"])
        plt.figure(figsize=(12, 5))
        monthly.plot(kind="bar", width=.85)
        plt.title("Публикации по месяцам")
        plt.xlabel("Месяц")
        plt.ylabel("Количество")
        if len(monthly) > 30:
            plt.xticks(np.arange(0, len(monthly), max(1, len(monthly) // 18)))
        plt.xticks(rotation=65)
        savefig(folder, "articles_by_month")

    hubs = Counter(hub for values in df["hubs"] for hub in values)
    tags = Counter(tag for values in df["tags"] for tag in values)
    pd.Series(hubs, dtype=int).sort_values(ascending=False).rename("count").to_csv(output / "hubs.csv")
    pd.Series(tags, dtype=int).sort_values(ascending=False).rename("count").to_csv(output / "tags.csv")
    plot_bar(pd.Series(hubs, dtype=int).sort_values(ascending=False), "Топ хабов", folder, "top_hubs")
    plot_bar(pd.Series(tags, dtype=int).sort_values(ascending=False), "Топ тегов", folder, "top_tags")
    plot_bar(df["detected_lang"].value_counts(), "Языки статей", folder, "languages")
    plot_bar(df["complexity"].fillna("unknown").value_counts(), "Уровни сложности", folder, "complexity")

    presence = pd.Series({
        "С блоками кода": int(df["code_article"].sum()),
        "С таблицами": int(df["has_table"].sum()),
        "С изображениями": int(df["has_images"].sum()),
        "Очень короткие (<256 токенов)": int(df["short_article"].sum()),
        "Пустые (<50 символов)": int(df["empty_article"].sum()),
        "Точные дубликаты (статьи)": int(df["exact_duplicate"].sum()),
    }).sort_values(ascending=False)
    presence.to_csv(output / "quality_flags.csv", header=["articles"])
    plot_bar(presence, "Структура и качество корпуса", folder, "quality_flags")

    near = near_duplicate_pairs(df)
    near.to_csv(output / "near_duplicates.csv", index=False)
    topic_map(df, folder)
    summary = {
        "articles": int(len(df)),
        "total_tokens": int(df["tokens"].sum()),
        "total_chars": int(df["chars"].sum()),
        "median_tokens": float(df["tokens"].median()),
        "p90_tokens": float(df["tokens"].quantile(.90)),
        "p95_tokens": float(df["tokens"].quantile(.95)),
        "articles_with_code_pct": round(float(df["code_article"].mean() * 100), 2),
        "articles_with_tables_pct": round(float(df["has_table"].mean() * 100), 2),
        "articles_with_images_pct": round(float(df["has_images"].mean() * 100), 2),
        "empty_articles": int(df["empty_article"].sum()),
        "exact_duplicate_articles": int(df["exact_duplicate"].sum()),
        "near_duplicate_pairs": int(len(near)),
        "near_duplicates_checked_articles": int(min(1500, (df["chars"] >= 100).sum())),
        "unique_hubs": len(hubs),
        "unique_tags": len(tags),
    }
    (output / "corpus_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def fixed_chunks(text, size, overlap=0):
    ids = ENC.encode(text, disallowed_special=())
    if not ids:
        return []
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and < size")
    step = size - overlap
    return [ENC.decode(ids[i:i + size]) for i in range(0, len(ids), step)
            if ids[i:i + size] and (i == 0 or len(ids) - i > overlap)]


def heading_chunks(text, size=512):
    sections = re.split(r"(?=^#{1,6}\s+)", text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]
    chunks = []
    current = ""
    for section in sections:
        if tokens(section) > size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(fixed_chunks(section, size, overlap=0))
        elif current and tokens(current + "\n\n" + section) > size:
            chunks.append(current.strip())
            current = section
        else:
            current = current + "\n\n" + section if current else section
    if current:
        chunks.append(current.strip())
    return chunks


def chunk_eda(df, output):
    folder = output / "plots" / "chunks"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    configs = [("fixed_256_o32", 256, 32), ("fixed_512_o64", 512, 64),
               ("fixed_1024_o128", 1024, 128), ("heading_512", 512, 0)]
    for article in tqdm(df.itertuples(index=False), total=len(df), desc="Chunking"):
        for strategy, size, overlap in configs:
            chunks = heading_chunks(article.markdown, size) if strategy.startswith("heading") else fixed_chunks(article.markdown, size, overlap)
            for idx, chunk in enumerate(chunks):
                rows.append({
                    "article_id": article.article_id,
                    "source_url": article.url,
                    "strategy": strategy,
                    "chunk_index": idx,
                    "chunk_id": f"{article.article_id}_{strategy}_{idx}",
                    "tokens": tokens(chunk),
                    "chars": len(chunk),
                    "short_chunk": tokens(chunk) < 128,
                    "contains_code": "```" in chunk,
                    "broken_code_fence": chunk.count("```") % 2 == 1,
                    "content": chunk,
                })
    chunks = pd.DataFrame(rows)
    if chunks.empty:
        return
    chunks.to_json(output / "chunks.jsonl", orient="records", lines=True, force_ascii=False)
    stats = chunks.groupby("strategy").agg(
        n_chunks=("tokens", "size"),
        mean_tokens=("tokens", "mean"),
        median_tokens=("tokens", "median"),
        p90_tokens=("tokens", lambda x: x.quantile(.90)),
        short_share=("short_chunk", "mean"),
        code_fence_broken_share=("broken_code_fence", "mean"),
        total_indexed_tokens=("tokens", "sum"),
    ).reset_index()
    stats["chunks_per_article"] = stats["n_chunks"] / len(df)
    stats.to_csv(output / "chunking_comparison.csv", index=False)

    plt.figure(figsize=(11, 5))
    sns.boxplot(data=chunks, x="strategy", y="tokens", showfliers=False, color="steelblue")
    plt.title("Длина чанков по стратегиям (без выбросов)")
    plt.xticks(rotation=20)
    savefig(folder, "chunk_length_boxplot")

    for col, title, name in [("chunks_per_article", "Среднее количество чанков на статью", "chunks_per_article"),
                             ("short_share", "Доля коротких чанков (<128 токенов)", "short_share"),
                             ("code_fence_broken_share", "Доля чанков с разорванным code fence", "broken_code")]:
        plt.figure(figsize=(10, 4))
        sns.barplot(data=stats, x="strategy", y=col, color="steelblue")
        plt.title(title)
        plt.xticks(rotation=15)
        savefig(folder, name)
    return chunks


def generate_qa(df, output, max_questions=100, model="gpt-4.1-mini"):
    # External LLM receives text snippets. Run only if you have rights to send the content.
    from openai import OpenAI
    client = OpenAI()
    qa_dir = output / "qa"
    qa_dir.mkdir(exist_ok=True)
    path = qa_dir / "synthetic_qa.jsonl"
    saved = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                saved.append(json.loads(line))
    seen = {clean_spaces(x["question"]).lower() for x in saved}
    candidates = df[df["tokens"] >= 256].sample(frac=1, random_state=42).to_dict("records")
    for article in tqdm(candidates, desc="Generate QA"):
        if len(saved) >= max_questions:
            break
        passage = article["content"][:9000]
        if len(passage) < 500:
            continue
        prompt = (
            "Сгенерируй ровно 2 реалистичных русскоязычных вопроса и эталонных ответа "
            "для RAG на основе текста. Не используй внешние знания. "
            "Для каждого ответа укажи source_quote — ТОЧНУЮ подстроку из текста, "
            "которая подтверждает ответ. Вопросы должны отличаться по формулировке "
            "от цитат. Верни только JSON вида: "
            '{"items":[{"question":"...","answer":"...","source_quote":"...",'
            '"question_type":"factual"}]}. '
            "question_type должен быть одним из: factual, howto, comparison, code, reasoning.\n\n"
            f"Заголовок: {article['title']}\nТекст: {passage}"
        )
        try:
            result = client.chat.completions.create(
                model=model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": "You create evidence-grounded Russian QA datasets."},
                          {"role": "user", "content": prompt}],
            )
            parsed = json.loads(result.choices[0].message.content or "{}")
            for item in parsed.get("items", []):
                question = clean_spaces(item.get("question", ""))
                quote = clean_spaces(item.get("source_quote", ""))
                answer = clean_spaces(item.get("answer", ""))
                kind = item.get("question_type", "factual")
                if kind not in {"factual", "howto", "comparison", "code", "reasoning"}:
                    kind = "factual"
                if len(question) < 12 or len(answer) < 12 or len(quote) < 15:
                    continue
                if quote not in clean_spaces(passage) or question.lower() in seen:
                    continue
                saved.append({
                    "question": question,
                    "ground_truth_answer": answer,
                    "source_quote": quote,
                    "article_id": article["article_id"],
                    "source_url": article["url"],
                    "primary_hub": article["primary_hub"],
                    "question_type": kind,
                    "is_answerable": True,
                    "review_status": "needs_manual_review",
                })
                seen.add(question.lower())
                if len(saved) >= max_questions:
                    break
            path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in saved) + "\n", encoding="utf-8")
        except Exception as e:
            print(f"[QA ERROR] article={article['article_id']}: {e}")
    return saved


def qa_eda(output):
    path = output / "qa" / "synthetic_qa.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return
    df = pd.read_json(path, lines=True)
    if df.empty:
        return
    folder = output / "plots" / "qa"
    folder.mkdir(exist_ok=True, parents=True)
    df["question_tokens"] = df["question"].map(tokens)
    df["answer_tokens"] = df["ground_truth_answer"].map(tokens)
    df["quote_tokens"] = df["source_quote"].map(tokens)
    df["duplicated_question"] = df["question"].map(lambda x: clean_spaces(x).lower()).duplicated(keep=False)
    df["split"] = df["article_id"].map(
        lambda a: "test" if int(hashlib.sha256(str(a).encode()).hexdigest()[:8], 16) % 5 == 0 else "dev")
    df.to_csv(output / "qa" / "synthetic_qa_review.csv", index=False)
    df[["question_tokens", "answer_tokens", "quote_tokens"]].describe().to_csv(output / "qa" / "qa_describe.csv")
    for field in ["question_tokens", "answer_tokens", "quote_tokens"]:
        plot_hist(df, field, f"QA: {field}", folder)
    plot_bar(df["question_type"].value_counts(), "Типы вопросов", folder, "question_types")
    plot_bar(df["primary_hub"].value_counts(), "Покрытие хабов вопросами", folder, "qa_hubs")
    plot_hist(df.groupby("article_id").size().rename("questions_per_article").reset_index()["questions_per_article"].to_frame(),
              "questions_per_article", "Число вопросов на статью", folder)
    qa_summary = {
        "n_questions": int(len(df)),
        "n_articles": int(df["article_id"].nunique()),
        "duplicate_questions": int(df["duplicated_question"].sum()),
        "needs_manual_review": int((df["review_status"] == "needs_manual_review").sum()),
        "split_counts": df["split"].value_counts().to_dict(),
    }
    (output / "qa" / "qa_summary.json").write_text(json.dumps(qa_summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Habr article parser + RAG EDA + synthetic QA")
    parser.add_argument("--ids", nargs="*", default=[], help="IDs or Habr article URLs")
    parser.add_argument("--ids-file", type=Path, help="Text file with IDs or URLs")
    parser.add_argument("--out", type=Path, default=Path("../habr_rag_output"))
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--generate-qa", action="store_true")
    parser.add_argument("--qa-count", type=int, default=100)
    parser.add_argument("--qa-model", default="gpt-4.1-mini")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw = list(args.ids)
    if args.ids_file:
        raw += args.ids_file.read_text(encoding="utf-8").splitlines()
    ids = extract_ids(raw)
    if ids:
        fetch_raw(ids, args.out, delay=args.delay, refresh=args.refresh)
    elif not list((args.out / "raw").glob("*.json")):
        parser.error("Provide --ids / --ids-file, or reuse a directory with cached raw/*.json")

    df = load_corpus(args.out)
    corpus_eda(df, args.out)
    chunk_eda(df, args.out)
    if args.generate_qa:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY to enable --generate-qa")
        generate_qa(df, args.out, max_questions=args.qa_count, model=args.qa_model)
    qa_eda(args.out)
    print(f"Done: {len(df)} articles; results in {args.out.resolve()}")


if __name__ == "__main__":
    main()
