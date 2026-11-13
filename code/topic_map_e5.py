"""Semantic article map: multilingual-e5-large embeddings + UMAP.

Install:
    pip install -U sentence-transformers umap-learn pandas numpy matplotlib beautifulsoup4 tqdm

Run (from project root):
    python topic_map_e5.py --out habr_rag_output
    python topic_map_e5.py --out habr_rag_output --html-dir habr_articles

If output/articles/*.md exists, it takes priority over HTML (unless --html-only).
Caches article embeddings and saves topic_map_e5_large.png, topic_map_e5_large.csv.
"""

import argparse
import hashlib
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("../habr_rag_output"))
    p.add_argument("--html-dir", type=Path, default=Path("habr_articles"))
    p.add_argument("--html-only", action="store_true")
    p.add_argument("--model", default="intfloat/multilingual-e5-large")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-tokens", type=int, default=400)
    p.add_argument("--max-chunks", type=int, default=6)
    p.add_argument("--max-articles", type=int, default=0, help="0 = all")
    p.add_argument("--top-hubs", type=int, default=8)
    return p.parse_args()


def extract_html(path):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for bad in soup.select("script, style, nav, footer, noscript"):
        bad.decompose()
    title_el = soup.select_one("h1.tm-title, h1")
    content_el = soup.select_one("div#post-content-body, div.tm-article-body, article")
    title = title_el.get_text(" ", strip=True) if title_el else path.stem
    body = content_el.get_text("\n", strip=True) if content_el else ""
    return title, body


def load_articles(args):
    meta_path = args.out / "articles_meta.csv"
    if meta_path.exists():
        metadata = pd.read_csv(meta_path, dtype={"article_id": str}).set_index("article_id")
    else:
        metadata = pd.DataFrame()

    md_files = sorted((args.out / "articles").glob("*.md")) if not args.html_only else []
    html_files = sorted(args.html_dir.glob("*.html"))
    files = md_files if md_files else html_files
    if not files:
        raise FileNotFoundError(
            f"No articles found in {args.out / 'articles'} or {args.html_dir}"
        )
    rows = []
    for path in tqdm(files, desc="Read articles"):
        article_id = path.stem
        if path.suffix.lower() == ".html":
            title, body = extract_html(path)
        else:
            body = path.read_text(encoding="utf-8", errors="replace")
            title = next((x.lstrip("# ") for x in body.splitlines() if x.startswith("# ")), article_id)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if len(body) < 100:
            continue
        hub = "Без хаба"
        if not metadata.empty and article_id in metadata.index:
            meta = metadata.loc[article_id]
            hub = str(meta.get("primary_hub", "Без хаба"))
            if not title or title == article_id:
                title = str(meta.get("title", article_id))
        rows.append({"article_id": article_id, "title": title, "primary_hub": hub, "content": body})

    df = pd.DataFrame(rows).drop_duplicates("article_id")
    if args.max_articles > 0:
        df = df.head(args.max_articles)
    if len(df) < 3:
        raise ValueError(f"Too few nonempty articles: {len(df)}")
    return df.reset_index(drop=True)


def token_chunks(text, tokenizer, max_tokens, max_chunks):
    # E5 supports 512 tokens. Leave room for prefix, title and special tokens.
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    starts = list(range(0, len(token_ids), max_tokens))
    if len(starts) > max_chunks:
        positions = np.linspace(0, len(starts) - 1, max_chunks, dtype=int)
        starts = [starts[i] for i in positions]
    return [tokenizer.decode(token_ids[s:s + max_tokens], skip_special_tokens=True) for s in starts]


def compute_embeddings(df, args):
    cache_path = args.out / "e5_large_article_embeddings.npz"
    previous = {}
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            for aid, sha, vec in zip(cache["ids"], cache["hashes"], cache["vectors"]):
                previous[(str(aid), str(sha))] = vec

    sha_list = [hashlib.sha256((row.title + "\n" + row.content + "\n" + args.model +
                              f"\n{args.chunk_tokens}:{args.max_chunks}").encode("utf-8")).hexdigest()
                for row in df.itertuples(index=False)]

    vecs = [None] * len(df)
    missing = []
    for i, (aid, sha) in enumerate(zip(df["article_id"], sha_list)):
        vec = previous.get((str(aid), sha))
        if vec is not None:
            vecs[i] = vec.astype("float32")
        else:
            missing.append(i)

    if missing:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.model)
        model.max_seq_length = 512
        tokenizer = model.tokenizer
        print(f"Computing {len(missing)} new article embeddings ({len(df) - len(missing)} cached)")

        # Encode each article separately to avoid retaining all chunk embeddings in memory.
        for i in tqdm(missing, desc="E5 encode"):
            item = df.iloc[i]
            # Reserve title/prefix budget, clip title for pathological cases.
            title = str(item["title"])[:150]
            chunks = token_chunks(item["content"], tokenizer, args.chunk_tokens, args.max_chunks)
            passages = [f"passage: {title}\n{chunk}" for chunk in chunks]
            chunk_vectors = model.encode(
                passages, batch_size=args.batch_size,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False,
            )
            mean = chunk_vectors.mean(axis=0)
            vecs[i] = (mean / (np.linalg.norm(mean) + 1e-12)).astype("float32")

        args.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            ids=df["article_id"].to_numpy(dtype=str),
            hashes=np.array(sha_list, dtype=str),
            vectors=np.stack(vecs).astype("float32"),
        )
    else:
        print(f"All {len(df)} article embeddings loaded from cache")

    return np.stack(vecs).astype("float32")


def make_plot(df, vectors, args):
    import umap

    xy = umap.UMAP(
        n_components=2,
        n_neighbors=min(20, len(df) - 1),
        min_dist=0.15,
        metric="cosine",
        random_state=42,
    ).fit_transform(vectors)

    meta = df[["article_id", "title", "primary_hub"]].copy()
    meta["x"] = xy[:, 0]
    meta["y"] = xy[:, 1]

    top = set(meta["primary_hub"].value_counts().head(args.top_hubs).index)
    meta["hub_group"] = meta["primary_hub"].where(meta["primary_hub"].isin(top), "Другие")

    out_dir = args.out / "plots" / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_dir / "topic_map_e5_large.csv", index=False, encoding="utf-8-sig")

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(12, 8))
    groups = list(meta["hub_group"].value_counts().index)
    cmap = plt.get_cmap("tab10")
    for i, group in enumerate(groups):
        subset = meta[meta["hub_group"] == group]
        ax.scatter(subset["x"], subset["y"], s=28, alpha=0.70,
                   color=cmap(i % 10), label=f"{group} ({len(subset)})")

    ax.set_title("Тематическая карта статей Хабра — multilingual-e5-large + UMAP")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    fig.tight_layout()
    output_image = out_dir / "topic_map_e5_large.png"
    fig.savefig(output_image, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_image.resolve()}")


if __name__ == "__main__":
    args = parse_args()
    if not 1 <= args.max_chunks:
        raise ValueError("--max-chunks must be >= 1")
    if not 50 <= args.chunk_tokens <= 450:
        raise ValueError("--chunk-tokens must be between 50 and 450")
    articles = load_articles(args)
    embeddings = compute_embeddings(articles, args)
    make_plot(articles, embeddings, args)
