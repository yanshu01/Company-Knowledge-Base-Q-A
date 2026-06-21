import os
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

load_dotenv()

client = Cerebras(
    api_key=os.getenv("CEREBRAS_API_KEY")
)


def generate_answer(question, context):

    # No context found
    if not context.strip():
        return (
            "I could not find relevant information "
            "in the uploaded documents."
        )

    response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": """
You are a document question-answering assistant.

Rules:
1. Answer ONLY using the provided context.
2. Do NOT use outside knowledge.
3. If the answer is partially available, provide the available information.
4. Quote names, dates, numbers, emails, phone numbers,
   and technical terms exactly as they appear.
5. If the answer truly does not exist in the context,
   respond with:
   "I could not find that information in the provided documents."
6. Keep answers concise and factual.
"""
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}
"""
            }
        ],
        temperature=0.1,
        max_tokens=500
    )

    return response.choices[0].message.content