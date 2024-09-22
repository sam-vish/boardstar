import os
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAG:
    def __init__(self, database_path: str):
        self.database_path = os.path.abspath(database_path)
        self.vector_store = None
        logger.info(f"RAG initialized with database path: {self.database_path}")

    def create_vector_store(self):
        if not os.path.exists(self.database_path):
            logger.error(f"Directory not found: '{self.database_path}'")
            return

        logger.info(f"Loading documents from: {self.database_path}")
        documents = []
        for root, _, files in os.walk(self.database_path):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        doc = Document(page_content=content, metadata={"source": file_path})
                        documents.append(doc)
                        logger.info(f"Loaded document: {file_path}")
                    except Exception as e:
                        logger.error(f"Error loading file {file_path}: {str(e)}")

        if not documents:
            logger.warning(f"No documents found in {self.database_path}")
            return

        logger.info(f"Loaded {len(documents)} documents")

        logger.info("Splitting documents into chunks")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        if not texts:
            logger.warning("No text chunks created after splitting")
            return

        logger.info(f"Created {len(texts)} text chunks")

        logger.info("Creating embeddings")
        embeddings = HuggingFaceEmbeddings()

        try:
            logger.info("Creating vector store")
            self.vector_store = FAISS.from_documents(texts, embeddings)
            logger.info(f"Vector store created successfully with {len(texts)} text chunks")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")

    def query(self, query: str, k: int = 1) -> List[Tuple[str, float]]:
        if not self.vector_store:
            logger.warning("Vector store not created. Please run create_vector_store() first.")
            return []

        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [(doc.page_content, score) for doc, score in results]