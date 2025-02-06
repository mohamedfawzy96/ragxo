import time
from typing import Self, Callable
from pymilvus import MilvusClient
from pydantic import BaseModel
import boto3
import dill
import os
import shutil
import logging
import tempfile
from botocore.exceptions import ClientError
import openai
from openai import ChatCompletion

logger = logging.getLogger(__name__)

class Document(BaseModel):
    text: str
    metadata: dict
    id: int

class Ragxo:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.collection_name = "ragx"
        os.makedirs("ragx_artifacts", exist_ok=True)

        self.db_path = f"ragx_artifacts/milvus_{int(time.time())}.db"
        self.client = MilvusClient(self.db_path)
        self.client.create_collection(self.collection_name, dimension=dimension)
        self.processing_fn = []
        self.embedding_fn = None
        self.system_prompt = None
        self.model = "gpt-4o-mini"
    
    def add_preprocess(self, fn: Callable) -> Self:
        self.processing_fn.append(fn)
        return self
    
    def add_llm_response_fn(self, fn: Callable) -> Self:
        self.llm_response_fn = fn
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
    
    def query(self, query: str, output_fields: list[str] = ['text', 'metadata'], limit: int = 10) -> list[list[dict]]:
        if not self.embedding_fn:
            raise ValueError("Embedding function not set. Please call add_embedding_fn first.")
            
        preprocessed_query = query
        for fn in self.processing_fn:
            preprocessed_query = fn(preprocessed_query)
        
        embedding = self.embedding_fn(preprocessed_query)
        
        return self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=limit,
            output_fields=output_fields
        )

    def export(self, destination: str, s3_bucket: str = None) -> Self:
        """
        Export the Ragx instance to either local filesystem or S3.
        
        Args:
            destination: str - Local path or S3 key prefix
            s3_bucket: str, optional - S3 bucket name. If provided, export to S3
        """
        try:
            # If s3_bucket is provided, export to S3
            if s3_bucket:
                return self._export_to_s3(destination, s3_bucket)
            
            # Otherwise, export to local filesystem
            os.makedirs(destination, exist_ok=True)
            
            # Save using dill
            pickle_path = os.path.join(destination, "ragx.pkl")
            with open(pickle_path, "wb") as f:
                dill.dump(self, f)
            
            # Copy database
            db_dest = os.path.join(destination, "milvus.db")
            shutil.copy(self.db_path, db_dest)
            
            return self
            
        except Exception as e:
            logger.error(f"Error in export: {e}")
            raise

    def _export_to_s3(self, prefix: str, bucket: str) -> Self:
        """
        Internal method to handle S3 export.
        """
        try:
            s3_client = boto3.client('s3')
            
            # Create a temporary directory for the files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save pickle file
                pickle_path = os.path.join(temp_dir, "ragx.pkl")
                with open(pickle_path, "wb") as f:
                    dill.dump(self, f)
                
                # Copy database
                db_path = os.path.join(temp_dir, "milvus.db")
                shutil.copy(self.db_path, db_path)
                
                # Upload to S3
                s3_client.upload_file(
                    pickle_path,
                    bucket,
                    f"{prefix}/ragx.pkl"
                )
                s3_client.upload_file(
                    db_path,
                    bucket,
                    f"{prefix}/milvus.db"
                )
            
            return self
            
        except ClientError as e:
            logger.error(f"Error uploading to S3: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in S3 export: {e}")
            raise

    @classmethod
    def load(cls, source: str, s3_bucket: str = None) -> Self:
        """
        Load a Ragx instance from either local filesystem or S3.
        
        Args:
            source: str - Local path or S3 key prefix
            s3_bucket: str, optional - S3 bucket name. If provided, load from S3
        """
        try:
            # If s3_bucket is provided, load from S3
            if s3_bucket:
                return cls._load_from_s3(source, s3_bucket)
            
            # Otherwise, load from local filesystem
            pickle_path = os.path.join(source, "ragx.pkl")
            
            with open(pickle_path, "rb") as f:
                instance = dill.load(f)
            
            # Restore client
            instance.client = MilvusClient(os.path.join(source, "milvus.db"))
            
            return instance
            
        except Exception as e:
            logger.error(f"Error in load: {e}")
            raise

    @classmethod
    def _load_from_s3(cls, prefix: str, bucket: str) -> 'Ragx':
        """
        Internal classmethod to handle S3 loading.
        """
        try:
            s3_client = boto3.client('s3')
            
            # Create a temporary directory for the files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download files from S3
                pickle_path = os.path.join(temp_dir, "ragx.pkl")
                db_path = os.path.join(temp_dir, "milvus.db")
                
                s3_client.download_file(
                    bucket,
                    f"{prefix}/ragx.pkl",
                    pickle_path
                )
                s3_client.download_file(
                    bucket,
                    f"{prefix}/milvus.db",
                    db_path
                )
                
                # Load the pickle file
                with open(pickle_path, "rb") as f:
                    instance = dill.load(f)
                
                # Restore client with the downloaded database
                instance.client = MilvusClient(db_path)
                
                return instance
                
        except ClientError as e:
            logger.error(f"Error downloading from S3: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in S3 load: {e}")
            raise
    
    def generate_llm_response(self, 
                              query: str, 
                              limit: int = 10,
                              data: list[dict] = None, 
                              temperature: float = 0.5,
                              max_tokens: int = 1000,
                              top_p: float = 1.0,
                              frequency_penalty: float = 0.0,
                              presence_penalty: float = 0.0,
                              ) -> ChatCompletion:
        if data is None:
            data = self.query(query, limit=limit)[0]
        
        if not self.system_prompt:
            raise ValueError("System prompt not set. Please call add_system_prompt first.")
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "query: {} data: {}".format(query, data)}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        
        return response