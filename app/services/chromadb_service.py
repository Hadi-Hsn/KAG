import chromadb
from app.config.settings import CHROMA_HOST, CHROMA_PORT, COLLECTION_NAME

class ChromaDBService:
    def __init__(self):
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with fallback strategy"""
        try:
            # Try to connect to external ChromaDB server (Docker)
            self.client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            print(f"Connected to ChromaDB server at {CHROMA_HOST}:{CHROMA_PORT}")
        except Exception as e:
            # Fallback to local persistent client for development
            print(f"Failed to connect to ChromaDB server: {e}")
            print("Falling back to local persistent client")
            self.client = chromadb.PersistentClient(path="rag_db")
        
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
    
    def add_documents(self, documents, embeddings, ids, metadatas):
        """Add documents to the collection with batch size handling"""
        # ChromaDB has a maximum batch size limit (around 5,461 items)
        # We'll use a conservative batch size to avoid hitting this limit
        max_batch_size = 1000  # Conservative batch size
        
        total_docs = len(documents)
        if total_docs <= max_batch_size:
            # Small batch, add directly
            return self.collection.add(
                documents=documents, 
                embeddings=embeddings, 
                ids=ids, 
                metadatas=metadatas
            )
        
        # Large batch, split into smaller chunks
        print(f"Large batch detected ({total_docs} documents), splitting into batches of {max_batch_size}")
        
        for i in range(0, total_docs, max_batch_size):
            end_idx = min(i + max_batch_size, total_docs)
            batch_documents = documents[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_ids = ids[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            print(f"Adding batch {i//max_batch_size + 1}: documents {i+1}-{end_idx} of {total_docs}")
            
            try:
                self.collection.add(
                    documents=batch_documents, 
                    embeddings=batch_embeddings, 
                    ids=batch_ids, 
                    metadatas=batch_metadatas
                )
            except Exception as e:
                # If batch still too large, try with smaller batch size
                if "batch size" in str(e).lower() and "exceeds" in str(e).lower():
                    print(f"Batch still too large, retrying with smaller batches...")
                    smaller_batch_size = 500  # Even smaller batch
                    
                    for j in range(i, end_idx, smaller_batch_size):
                        small_end_idx = min(j + smaller_batch_size, end_idx)
                        small_batch_documents = documents[j:small_end_idx]
                        small_batch_embeddings = embeddings[j:small_end_idx]
                        small_batch_ids = ids[j:small_end_idx]
                        small_batch_metadatas = metadatas[j:small_end_idx]
                        
                        print(f"Adding small batch: documents {j+1}-{small_end_idx} of {total_docs}")
                        
                        self.collection.add(
                            documents=small_batch_documents, 
                            embeddings=small_batch_embeddings, 
                            ids=small_batch_ids, 
                            metadatas=small_batch_metadatas
                        )
                else:
                    # Re-raise other types of errors
                    raise e
        
        print(f"Successfully added all {total_docs} documents to ChromaDB in batches")
    
    def query_documents(self, query_embeddings, n_results=5):
        """Query documents from the collection"""
        return self.collection.query(
            query_embeddings=query_embeddings, 
            n_results=n_results
        )
    
    def get_all_documents(self):
        """Get all documents from the collection"""
        return self.collection.get()
    
    def delete_documents(self, ids):
        """Delete documents by IDs"""
        return self.collection.delete(ids=ids)
    
    def reset_collection(self):
        """Reset the entire collection by deleting it and recreating it"""
        try:
            # Delete the existing collection
            self.client.delete_collection(name=COLLECTION_NAME)
            print(f"Deleted collection: {COLLECTION_NAME}")
            
            # Recreate the collection
            self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
            print(f"Recreated collection: {COLLECTION_NAME}")
            
            return True
        except Exception as e:
            print(f"Error resetting collection: {e}")
            try:
                # If deletion fails, try to recreate anyway
                self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
                return True
            except Exception as e2:
                print(f"Error recreating collection: {e2}")
                return False
    
    def get_collection_info(self):
        """Get information about the current collection"""
        try:
            docs = self.get_all_documents()
            return {
                "collection_name": COLLECTION_NAME,
                "document_count": len(docs.get('documents', [])),
                "has_documents": len(docs.get('documents', [])) > 0
            }
        except Exception as e:
            return {"error": str(e)}

# Global instance
chromadb_service = ChromaDBService()
