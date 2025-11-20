import logging
import time
from pathlib import Path
from typing import Optional, List
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline

logger = logging.getLogger(__name__)



class RAGEngine:
    
    def __init__(
        self, 
        course_name: str, 
        model_path: Optional[str] = None, 
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        auto_ingest: bool = True, 
        n_gpu_layers: int = -1,
        lazy_load_llm: bool = False
    ):
        self.course_name = course_name
        self.data_path = Path("data")
        self.vectors_path = self.data_path / "vectors" / course_name
        self.vectors_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading embedding model: {embedding_model}")
        try:
            self.embed_model = HuggingFaceEmbedding(
                model_name=embedding_model,
                cache_folder=str(Path("models/embeddings")),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
        
        self.model_path = model_path if model_path else self._find_model()
        self.n_gpu_layers = n_gpu_layers
        self._llm = None
        
        if not lazy_load_llm:
            self._init_llm()
        
        self._init_vector_store()
        self._init_index()
        
        if auto_ingest:
            try:
                self.ingest_documents()
            except Exception as e:
                logger.warning(f"Auto-ingestion failed: {e}")
    
    @property
    def llm(self):
        if self._llm is None:
            self._init_llm()
        return self._llm
    
    def _init_llm(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please place your model in models/llm_models/"
            )
        
        logger.info(f"Loading LLM from: {self.model_path} with {self.n_gpu_layers} GPU layers")
        try:
            self._llm = LlamaCPP(
                model_path=self.model_path,
                temperature=0.1,
                max_new_tokens=256,  
                context_window=4096,  
                verbose=True,  
                model_kwargs={
                    "n_gpu_layers": self.n_gpu_layers,
                    "repeat_penalty": 1.2,
                    "top_p": 0.9,
                    "n_batch": 512,  
                    "n_threads": 4,  
                    "n_ctx": 4096,  
                    "use_mlock": True,  
                    "use_mmap": True,  
                },
                generate_kwargs={"stop": ["Query:", "Context:", "User:", "\n\n\n"]}
            )
            logger.info("LLM loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM: {e}")
    
    def _find_model(self) -> str:
        models_dir = Path("models/llm_models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        gguf_files = list(models_dir.glob("*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(
                "No .gguf model found. Please place your model in models/llm_models/"
            )
        
        return str(gguf_files[0])
    
    def _init_vector_store(self):
        chroma_settings = ChromaSettings(anonymized_telemetry=False)
        chroma_client = chromadb.PersistentClient(
            path=str(self.vectors_path),
            settings=chroma_settings,
        )
        
        collection = chroma_client.get_or_create_collection(
            name=self.course_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.vector_store = ChromaVectorStore(chroma_collection=collection)
    
    def _init_index(self):
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=SimpleDocumentStore(),
            index_store=SimpleIndexStore(),
        )
        
        try:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=storage_context,
                embed_model=self.embed_model,
            )
            logger.info(f"Loaded existing index for course: {self.course_name}")
        except Exception:
            self.index = VectorStoreIndex.from_documents(
                [],
                storage_context=storage_context,
                embed_model=self.embed_model,
            )
            logger.info(f"Created new index for course: {self.course_name}")
    
    def ingest_documents(self, pdf_folder: Optional[Path] = None):
        if pdf_folder is None:
            pdf_folder = Path("data/pdfs") / self.course_name
        
        if not pdf_folder.exists():
            logger.warning(f"PDF folder not found: {pdf_folder}")
            return
        
        pdf_files = list(pdf_folder.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_folder}")
            return
        
        logger.info(f"Ingesting {len(pdf_files)} PDF files...")
        
        documents = self._load_pdfs(pdf_files)
        
        if documents:
            node_parser = SentenceSplitter(
                chunk_size=1024,  
                chunk_overlap=128  
            )
            
            pipeline = IngestionPipeline(
                transformations=[node_parser]
            )
            
            nodes = pipeline.run(documents=documents, show_progress=True)
            
            self.index.insert_nodes(nodes)
            logger.info(f"Ingested {len(documents)} documents ({len(nodes)} nodes)")
        else:
            logger.warning("No documents were successfully loaded")
    
    def _load_pdfs(self, pdf_files: List[Path]) -> List[Document]:
        documents = []
        for pdf_file in pdf_files:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if text.strip():
                    doc = Document(
                        text=text,
                        metadata={
                            "source": str(pdf_file.name),
                            "course": self.course_name,
                            "file_path": str(pdf_file)
                        }
                    )
                    documents.append(doc)
                    logger.info(f"Loaded {pdf_file.name}: {len(text)} characters")
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
        
        return documents
    
    @lru_cache(maxsize=100)
    def _embed_query_cached(self, query: str):
        return self.embed_model.get_query_embedding(query)
    
    def query(self, question: str, top_k: int = 2, verbose: bool = True) -> str:
        t_start = time.time()
        
        question = question.strip()
        print(f"[DEBUG] Starting query: {question[:50]}...")
        
        t0 = time.time()
        print(f"[DEBUG] Creating retriever...")
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        
        print(f"[DEBUG] Retrieving nodes...")
        try:
            nodes = retriever.retrieve(question)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return "I encountered an error retrieving relevant information. Please try again."
        
        t1 = time.time()
        print(f"[TIMING] Retrieval took: {t1-t0:.2f}s")
        
        if not nodes:
            return "I couldn't find relevant information to answer your question."
        
        print(f"[DEBUG] Building context from {len(nodes)} nodes...")
        context = "\n\n".join([node.get_content() for node in nodes])
        print(f"[DEBUG] Context length: {len(context)} chars")
        
        prompt = f"""Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query concisely and accurately.
Query: {question}
Answer:"""
        
        print(f"[DEBUG] Prompt length: {len(prompt)} chars")
        
        t2 = time.time()
        print(f"[DEBUG] Calling LLM...")
        try:
            response = self.llm.complete(prompt)
            text = response.text.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I encountered an error generating a response. Please try again."
        
        t3 = time.time()
        print(f"[TIMING] LLM inference took: {t3-t2:.2f}s")
        print(f"[TIMING] Total query time: {t3-t_start:.2f}s")
        
        text = self._clean_response(text)
        
        return text
    
    def _clean_response(self, text: str) -> str:
        text = text.lstrip("#").strip()
        
        for marker in ["Query:", "Context:", "Answer:", "---------------------", "User:"]:
            if marker in text:
                text = text.split(marker)[0].strip()
        
        return text
    
    def stream_query(self, question: str, top_k: int = 2):
        question = question.strip()
        
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(question)
        
        if not nodes:
            yield "I couldn't find relevant information to answer your question."
            return
        
        context = "\n\n".join([node.get_content() for node in nodes])
        
        prompt = f"""Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query concisely and accurately.
Query: {question}
Answer:"""
        
        response_stream = self.llm.stream_complete(prompt)
        
        for chunk in response_stream:
            yield chunk.delta
    
    def get_stats(self) -> dict:
        try:
            collection = self.vector_store._collection
            count = collection.count()
        except:
            count = 0
        
        return {
            "course": self.course_name,
            "embedding_model": self.embed_model.model_name,
            "document_count": count,
            "gpu_layers": self.n_gpu_layers,
            "llm_loaded": self._llm is not None,
        }