PROMPT_GEN_DATASET = """
You are an expert in creating test datasets for Retrieval-Augmented Generation (RAG).

You are given a DOCUMENT TEXT.

Your task:

1. Generate 1 question that can be answered exclusively using the provided text.
2. The question must:
   - be clear and unambiguous
   - not require any external knowledge
   - not reference the text explicitly (avoid phrases like "in the text", "in the document", etc.)
   - sound natural, like a real user question unaware of the source
   - belong to one of the types: fact, reason, explanation, or detail

3. Provide a short reference answer:
   - 1–3 sentences
   - strictly based on the given text
   - without adding any external information

Output format:

{
  "question": "...",
  "answer": "..."
}

Return only JSON with no additional text.
"""