# x.py
from ragxo import Ragxo, Document
import openai

EMBEDDING_DIMENSION = 1536

# Create a callable class for the embedding function

def create_sample_documents():
    texts = [
        """Python is a high-level programming language known for its simplicity and readability. 
        It has a large standard library and extensive third-party packages.""",
        
        """Machine learning is a subset of artificial intelligence that enables systems to learn 
        and improve from experience without being explicitly programmed.""",
        
        """Natural Language Processing (NLP) is a branch of AI that helps computers understand, 
        interpret, and manipulate human language."""
    ]
    
    documents = []
    id = 0
    for text in texts:
        doc = Document(
            text=text,
            metadata={"source": "sample_data"},
            id=id
        )
        documents.append(doc)
        id += 1
    return documents

def preprocess_text(text: str) -> str:
    """Simple preprocessing function that splits text into sentences"""
    sentences = text.split(" ")
    sentences = [sentence.lower() for sentence in sentences]
    return " ".join(sentences)

def embedding_function(text: str) -> list[float]:
    client = openai.OpenAI()
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def main():
    # Initialize embedding model
    
    # Initialize Ragx
    ragx_retrieval = Ragxo(dimension=EMBEDDING_DIMENSION)
    
    # Add preprocessing and embedding functions
    ragx_retrieval.add_preprocess(preprocess_text)
    ragx_retrieval.add_embedding_fn(embedding_function)
    ragx_retrieval.add_system_prompt("You are a helpful assistant that can answer questions about the data provided.")
    ragx_retrieval.add_model("gpt-4o-mini")
    # Create and index sample documents
    documents = create_sample_documents()
    ragx_retrieval.index(documents)
    
    # Perform a query
    query = "What is Python programming?"
    results = ragx_retrieval.query(query)
    print("\nQuery:", query)
    print("Results:", results)
    
    # Export the index
    
    ragx_retrieval.export("ragx_export_v3", s3_bucket="ragxo")
    # Load the index - CORRECTED THIS PART
    loaded_ragx = Ragxo.load("ragx_export", s3_bucket="ragxo")  # Use class method directly
    
    # Verify it works after loading
    results_after_load = loaded_ragx.query(query)
    
    print("\nResults after loading:", results_after_load)


    response = loaded_ragx.generate_llm_response("query: What is Python programming? data: {}".format(results_after_load))
    print("\nResponse:", response.choices[0].message.content)
    

if __name__ == "__main__":
    main()