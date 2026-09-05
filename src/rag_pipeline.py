from openai import OpenAI

from src.config import OPENAI_API_KEY


class RAGPipeline:

    def __init__(self):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

    def generate_answer(self, question, retrieved_documents):

        context = "\n\n".join(
            [
                doc["document"]
                for doc in retrieved_documents
            ]
        )

        prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY
the information provided in the context.

If the answer cannot be found in the context,
say:

"I don't have enough information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a reliable document QA assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content
