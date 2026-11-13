import json
import requests
from tqdm import tqdm
from qdrant_client import QdrantClient
from code.gen_evals_dataset.prompt import PROMPT_GEN_DATASET

OUTPUT_PATH = r"examples\rag_eval_dataset.json"

OPENROUTER_API_KEY = ""
OPENROUTER_MODEL = "deepseek/deepseek-v3.2"

QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = ""  # если не нужен, оставь ""
QDRANT_COLLECTION = "documents"

# имя payload-поля, где лежит текст документа
TEXT_PAYLOAD_KEY = "text"
URL_PAYLOAD_KEY = "url"

# сколько points за один scroll-запрос
SCROLL_BATCH_SIZE = 100


def extract_json_from_response(content: str) -> dict:
    content = content.strip()

    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

    return json.loads(content)


def generate_question(text: str) -> tuple[str, str]:
    try:
        messages = [
            {"role": "system", "content": PROMPT_GEN_DATASET},
            {
                "role": "user",
                "content": f'''ТЕКСТ:
"""
{text}
"""''',
            },
        ]

        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.2,
            },
            timeout=60,
        )
        r.raise_for_status()

        response_json = r.json()
        content = response_json["choices"][0]["message"]["content"]
        llm_res = extract_json_from_response(content)

        return llm_res["question"], llm_res["answer"]

    except Exception as e:
        print(f"Error while generating QA: {e}")
        return "", ""


def get_all_qdrant_documents(
        client: QdrantClient,
        collection_name: str,
        batch_size: int = 100,
):
    """
    Возвращает весь payload коллекции Qdrant списком.
    """
    all_payloads = []
    next_page_offset = None

    while True:
        points, next_page_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            with_payload=True,
            with_vectors=False,
            offset=next_page_offset,
        )

        if not points:
            break

        all_payloads.extend(point.payload for point in points)

        if next_page_offset is None:
            break

    return all_payloads


def main():
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
    )

    dataset = []
    idx = 0

    for payload in tqdm(
            get_all_qdrant_documents(
                client=client,
                collection_name=QDRANT_COLLECTION,
                batch_size=SCROLL_BATCH_SIZE,
            ),
            desc="Processing Qdrant documents",
    ):
        if idx % 40 != 0:
            idx += 1
            continue
        text = payload.get(TEXT_PAYLOAD_KEY, "")
        url = payload.get("metadata", {}).get(URL_PAYLOAD_KEY, "")

        if not text:
            continue

        question, answer = generate_question(text)
        if not question:
            continue

        item = {
            "id": f"q_{idx:04d}",
            "question": question,
            "answers": [answer],
            "ground_truth_docs": [
                {
                    "url": url,
                    "text": text,
                }
            ],
        }

        dataset.append(item)
        idx += 1

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(dataset)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
