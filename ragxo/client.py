import time
from typing import Literal, Self, Callable
from pymilvus import MilvusClient
from pydantic import BaseModel, Field
import boto3
import dill
import os
import shutil
import logging
import tempfile
from botocore.exceptions import ClientError
import openai
from openai import ChatCompletion
from ragxo.utils import with_loading

logger = logging.getLogger(__name__)

class Document(BaseModel):
    text: str
    metadata: dict
    id: int

class EvaluationExample(BaseModel):
    query: str
    expected: str

class EvaluationResults(BaseModel):
    results: list[str] = Field(description="A list of strings, each either 'correct' or 'incorrect'")


class Ragxo:
    """
    A RAG (Retrieval-Augmented Generation) system that combines vector search with LLM responses.
    
    Attributes:
        dimension (int): Dimension of the embedding vectors
        collection_name (str): Name of the Milvus collection
        db_path (str): Path to the Milvus database
        processing_fn (list): List of preprocessing functions
        embedding_fn (Callable): Function to generate embeddings
        system_prompt (str): System prompt for LLM
        model (str): LLM model name
    """
    
    def __init__(self, dimension: int) -> None:
        """
        Initialize the Ragxo instance.
        
        Args:
            dimension (int): Dimension of the embedding vectors
        """
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
        self.limit = 10
        self.temperature = 0.5
        self.max_tokens = 2000
        self.top_p = 1.0
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
        
    
    def add_preprocess(self, fn: Callable) -> Self:
        """
        Add a preprocessing function to the pipeline.
        
        Args:
            fn (Callable): Function that takes and returns a string
            
        Returns:
            Self: The current instance for method chaining
        """
        self.processing_fn.append(fn)
        return self
    
    def add_llm_response_fn(self, fn: Callable) -> Self:
        """
        Add a function to process LLM responses.
        
        Args:
            fn (Callable): Function to process LLM responses
            
        Returns:
            Self: The current instance for method chaining
        """
        self.llm_response_fn = fn
        return self
    
    def add_embedding_fn(self, fn: Callable) -> Self:
        """
        Set the embedding function for vector generation.
        
        Args:
            fn (Callable): Function that converts text to embeddings
            
        Returns:
            Self: The current instance for method chaining
            
        Raises:
            ValueError: If fn is None
        """
        if not fn:
            raise ValueError("Embedding function cannot be None")
        self.embedding_fn = fn
        return self
    
    def add_system_prompt(self, prompt: str) -> Self:
        """
        Set the system prompt for LLM interactions.
        
        Args:
            prompt (str): System prompt text
            
        Returns:
            Self: The current instance for method chaining
        """
        self.system_prompt = prompt
        return self
    
    def add_model(self, model: str,                              
                        limit: int = 10,
                        temperature: float = 0.5,
                        max_tokens: int = 1000,
                        top_p: float = 1.0,
                        frequency_penalty: float = 0.0,
                        presence_penalty: float = 0.0) -> Self:
        """
        Configure the LLM model and its parameters.
        
        Args:
            model (str): Name of the LLM model
            limit (int): Maximum number of results to return from vector search
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens in response
            top_p (float): Nucleus sampling parameter
            frequency_penalty (float): Frequency penalty parameter
            presence_penalty (float): Presence penalty parameter
            
        Returns:
            Self: The current instance for method chaining
        """
        self.model = model
        self.limit = limit
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        return self
    
    @with_loading("Indexing documents")
    def index(self, data: list[Document]) -> Self:
        """
        Index documents into the vector database.
        
        Args:
            data (list[Document]): List of documents to index
            
        Returns:
            Self: The current instance for method chaining
            
        Raises:
            ValueError: If embedding function is not set
        """
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
        """
        Search the vector database for similar documents.
        
        Args:
            query (str): Search query
            output_fields (list[str]): Fields to return in results
            limit (int): Maximum number of results
            
        Returns:
            list[list[dict]]: Search results
            
        Raises:
            ValueError: If embedding function is not set
        """
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

    @with_loading("Exporting Ragxo instance")
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
    def _load_from_s3(cls, prefix: str, bucket: str) -> 'Ragxo':
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
                              history: list[dict] = [],
                              messages: list[dict] = None,
                              data: list[dict] = None) -> ChatCompletion:
        """
        Generate LLM response based on query and retrieved data.
        
        Args:
            query (str): User query, this is used if messages is None
            data (list[dict], optional): Retrieved documents. If None, performs a new query
            history (list[dict], optional): History of messages
            messages (list[dict], optional): Messages to pass to the LLM: [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Some user message"}, {"role": "assistant", "content": "Some assistant message"}]
            
        Returns:
            ChatCompletion: LLM response
            
        Raises:
            ValueError: If system prompt is not set
        """
        if data is None:
            data = self.query(query, limit=self.limit)[0]
        
        if not self.system_prompt:
            raise ValueError("System prompt not set. Please call add_system_prompt first.")
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt}
            ] + history + [
                {"role": "user", "content": f"query: {query} data: {data}"}
            ] if messages is None else messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        
        return response

    
    
    @with_loading("Evaluating test dataset")
    def evaluate(self, test_data: list[EvaluationExample], batch_size: int = 10, judge_model: str = "gpt-4o-mini") -> float:
        """
        Evaluate the performance of the RAG system on a test dataset using a single prompt per batch.
        
        For each batch:
        1. Generates an answer for each query.
        2. Concatenates evaluation details (query, expected, generated answer) into one prompt.
        3. Instructs the judge to output a JSON object strictly adhering to our schema:
            {"results": ["correct", "incorrect", ...]}.
        4. Parses the structured output and computes overall accuracy.
        
        Args:
            test_data (list[EvaluationExample]): List of evaluation examples.
            batch_size (int): Number of examples to process per batch.
        
        Returns:
            float: Accuracy as a fraction of correct evaluations.
        """
        total = len(test_data)
        correct_count = 0

        for i in range(0, total, batch_size):
            batch = test_data[i : i + batch_size]
            batch_prompt = "Evaluate the following examples and output your answer as a JSON object with a single key \"results\" that maps to an array of strings. Each element in the array should be either \"correct\" or \"incorrect\", corresponding to each example in order.\n\n"
            
            # For each example in the batch, generate the answer and include details.
            for idx, example in enumerate(batch):
                query = example.query
                expected = example.expected

                # Generate the answer using the RAG system.
                llm_response = self.generate_llm_response(query)
                generated_answer = llm_response.choices[0].message.content.strip()
                
                batch_prompt += f"Example {idx+1}:\n"
                batch_prompt += f"Query: {query}\n"
                batch_prompt += f"Expected Answer: {expected}\n"
                batch_prompt += f"Generated Answer: {generated_answer}\n\n"
            
            # Append clear instructions for the structured output.
            batch_prompt += (
                "Return your output as a JSON object exactly in this format: "
                "{\"results\": [\"correct\", \"incorrect\", ...]} with no additional text or markdown formatting."
            )

            messages = [
                {"role": "system", "content": "You are an expert evaluator. Evaluate whether each generated answer meets the expected answer."},
                {"role": "user", "content": batch_prompt}
            ]
            
            # Call the OpenAI API with a structured response enforced via a JSON Schema.
            response = openai.beta.chat.completions.parse(
                model=judge_model,
                messages=messages,
                temperature=0,  # Deterministic output.
                response_format=EvaluationResults
            )

            output_text = response.choices[0].message.content.strip()
            
            try:
                # Parse the JSON output using the Pydantic model.
                eval_results = EvaluationResults.model_validate_json(output_text)
            except Exception as e:
                print(f"Error parsing JSON: {e}\nReceived output: {output_text}")
                eval_results = None
            
            if eval_results:
                for result in eval_results.results:
                    if result.lower() == "correct":
                        correct_count += 1
            else:
                print("Skipping batch due to parsing error.")

        accuracy = correct_count / total if total > 0 else 0.0
        print(f"Accuracy: {accuracy * 100:.2f}% ({correct_count}/{total})")
        return accuracy
