import pytest
from ragx.ragx import Ragx, Document
import os
import shutil

@pytest.fixture
def ragx_instance():
    return Ragx(dimension=2)

@pytest.fixture
def mock_embedding_fn():
    return lambda x: [0.1, 0.2, 0.3]  # Simple mock embedding function

@pytest.fixture
def sample_documents():
    return [
        Document(text="test document 1", metadata={"source": "test"}, id="1"),
        Document(text="test document 2", metadata={"source": "test"}, id="2"),
    ]

def test_ragx_initialization(ragx_instance):
    assert ragx_instance.collection_name == "ragx"
    assert ragx_instance.db_path == "milvus.db"
    assert isinstance(ragx_instance.processing_fn, list)
    assert len(ragx_instance.processing_fn) == 0

def test_add_preprocess(ragx_instance):
    def mock_process(text: str) -> str:
        return text.upper()
    
    ragx_instance.add_preprocess(mock_process)
    assert len(ragx_instance.processing_fn) == 1
    assert ragx_instance.processing_fn[0]("test") == "TEST"

def test_add_embedding_fn(ragx_instance, mock_embedding_fn):
    ragx_instance.add_embedding_fn(mock_embedding_fn)
    assert ragx_instance.embedding_fn("test") == [0.1, 0.2, 0.3]

def test_export_and_load(ragx_instance, mock_embedding_fn, tmp_path):
    # Setup
    ragx_instance.add_embedding_fn(mock_embedding_fn)
    export_path = str(tmp_path / "export_test")
    
    # Export
    ragx_instance.export(export_path)
    
    # Verify export
    assert os.path.exists(export_path)
    assert os.path.exists(os.path.join(export_path, "ragx.pkl"))
    
    # Load
    new_instance = Ragx(dimension=2)
    new_instance.load(export_path)
    
    # Verify loaded instance
    assert new_instance.collection_name == ragx_instance.collection_name
    assert new_instance.db_path == ragx_instance.db_path
    
    # Cleanup
    shutil.rmtree(export_path)

def test_index_and_query(ragx_instance, mock_embedding_fn, sample_documents):
    # Setup
    ragx_instance.add_embedding_fn(mock_embedding_fn)
    
    # Test indexing
    ragx_instance.index(sample_documents)
    
    # Test querying
    results = ragx_instance.query("test query")
    assert isinstance(results, list)

def test_preprocessing_pipeline(ragx_instance, mock_embedding_fn):
    def to_upper(text: str) -> str:
        return text.upper()
    
    def remove_spaces(text: str) -> str:
        return text.replace(" ", "")
    
    # Add preprocessing functions
    ragx_instance.add_preprocess(to_upper)
    ragx_instance.add_preprocess(remove_spaces)
    
    # Test preprocessing chain
    test_text = "hello world"
    processed = test_text
    for fn in ragx_instance.processing_fn:
        processed = fn(processed)
    
    assert processed == "HELLOWORLD" 