from typing import Self, Callable
from pymilvus import MilvusClient
from pydantic import BaseModel
import dill
import os
import shutil
import logging
import openai
from openai import ChatCompletion

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Document(BaseModel):
    text: str
    metadata: dict
    id: int

class Ragxo:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.collection_name = "ragx"
        self.db_path = "milvus.db"
        self.client = MilvusClient(self.db_path)
        self.client.create_collection(self.collection_name, dimension=dimension)
        self.processing_fn = []
        self.embedding_fn = None
        self.system_prompt = None
        self.model = "gpt-4o-mini"
    
    def add_preprocess(self, fn: Callable) -> Self:
        self.processing_fn.append(fn)
        return self
    
    def add_embedding_fn(self, fn: Callable) -> Self:
        if not fn:
            raise ValueError("Embedding function cannot be None")
        self.embedding_fn = fn
        return self
    
    def add_system_prompt(self, prompt: str) -> Self:
        self.system_prompt = prompt
        return self
    
    def add_model(self, model: str) -> Self:
        self.model = model
        return self
    
    def index(self, data: list[Document]) -> Self:
        if not self.embedding_fn:
            raise ValueError("Embedding function not set")
            
        processed_text = []
        for item in data:
            current_text = item.text
            for fn in self.processing_fn:
                current_text = fn(current_text)
            processed_text.append(current_text)
            
        embeddings = [
            self.embedding_fn(text)
            for text in processed_text
        ]
        
        self.client.insert(self.collection_name, [
            {
                "text": item.text,
                "metadata": item.metadata,
                "id": item.id,
                "vector": embedding
            }
            for item, embedding in zip(data, embeddings)
        ])
        return self
    
    def query(self, query: str, output_fields: list[str] = ['text', 'metadata']) -> list[list[dict]]:
        if not self.embedding_fn:
            raise ValueError("Embedding function not set. Please call add_embedding_fn first.")
            
        preprocessed_query = query
        for fn in self.processing_fn:
            preprocessed_query = fn(preprocessed_query)
        
        embedding = self.embedding_fn(preprocessed_query)
        
        return self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=10,
            output_fields=output_fields
        )
    
    def export(self, folder_path: str) -> Self:
        try:
            os.makedirs(folder_path, exist_ok=True)
            
            # Save using dill
            pickle_path = os.path.join(folder_path, "ragx.pkl")
            with open(pickle_path, "wb") as f:
                dill.dump(self, f)
            
            # Copy database
            db_dest = os.path.join(folder_path, "milvus.db")
            shutil.copy(self.db_path, db_dest)
            
            return self
            
        except Exception as e:
            logger.error(f"Error in export: {e}")
            raise
    
    @classmethod
    def load(cls, folder_path: str) -> 'Ragx':
        try:
            pickle_path = os.path.join(folder_path, "ragx.pkl")
            
            with open(pickle_path, "rb") as f:
                instance = dill.load(f)
            
            # Restore client
            instance.client = MilvusClient(os.path.join(folder_path, "milvus.db"))
            
            return instance
            
        except Exception as e:
            logger.error(f"Error in load: {e}")
            raise
    
    def generate_llm_response(self, query: str, data: list[dict] = None) -> ChatCompletion:
        
        if data is None:
            data = self.query(query)[0]
        
        if not self.system_prompt:
            raise ValueError("System prompt not set. Please call add_system_prompt first.")
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "query: {} data: {}".format(query, data)}
            ]
        )
        
        return response