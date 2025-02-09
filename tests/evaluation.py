import os
import time
from datasets import load_dataset
import openai
from ragxo import Document, EvaluationExample, Ragxo  # Adjust the import as needed

# Import your Ragxo classes (Document, EvaluationExample, Ragxo) from your module.
# For example, if they are in a file named ragxo.py, you could do:
# from ragxo import Ragxo, Document, EvaluationExample
#
# For this script, we assume the definitions are available in the current scope.
# (You can adjust the import according to your project structure.)

# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Define an embedding function using OpenAI's embedding endpoint.
def embedding_fn(text: str) -> list[float]:
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# -----------------------------
# Load and Prepare SQuAD Dataset
# -----------------------------
# For demonstration purposes, we'll use a small subset (e.g., first 50 examples)
squad_data = load_dataset("squad", split="validation[:200]")

# Build a set of unique context passages to index.

documents = []
seen_contexts = set()
doc_id = 0

for item in squad_data:
    context = item["context"]
    # Avoid duplicate contexts to reduce redundancy.
    if context not in seen_contexts:
        seen_contexts.add(context)
        # Use the SQuAD "title" as metadata.
        documents.append(Document(text=context, metadata={"title": item.get("title", "unknown")}, id=doc_id))
        doc_id += 1

# Prepare evaluation examples: each example is a question paired with its first answer.
evaluation_examples = []
for item in squad_data:
    question = item["question"]
    answers = item["answers"]["text"]
    expected = answers[0] if answers else ""
    evaluation_examples.append(EvaluationExample(query=question, expected=expected))

# -----------------------------
# Initialize and Configure Ragxo
# -----------------------------
# The "text-embedding-ada-002" model returns vectors of dimension 1536.
ragxo = Ragxo(dimension=1536)
ragxo.add_embedding_fn(embedding_fn)

# Add a simple preprocessing step (e.g., trimming whitespace)
ragxo.add_preprocess(lambda text: text.strip())

# Set a system prompt that instructs the LLM to answer questions using the provided context.
ragxo.add_system_prompt(
    "You are a helpful question answering assistant. Use the provided context to answer the question accurately."
)

# Configure the chat model for answering and evaluation.
# You can use "gpt-3.5-turbo" (or another model) as desired.
ragxo.add_model("gpt-4o-mini", temperature=0.5, max_tokens=150)

# Index the context documents into the vector database.
ragxo.index(documents)

# -----------------------------
# Run Evaluation on SQuAD Examples
# -----------------------------
# Here we evaluate in batches; you can adjust the batch_size as desired.
batch_size = 20
accuracy = ragxo.evaluate(evaluation_examples, batch_size=1000)
print(f"Final Accuracy: {accuracy * 100:.2f}%")
