# RAG/medication_rag.py
import pandas as pd
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

class MedicationRAG:
    def __init__(self, csv_path: str = "medications.csv"):
        """Initialize the RAG system with medication data."""
        self.csv_path = csv_path
        self.df = None
        self.vectorstore = None
        self.embeddings = None
        
    def load_and_process_data(self):
        """Load CSV and create searchable documents."""
        print("📊 Loading medication data...")
        self.df = pd.read_csv(self.csv_path)
        
        # Create rich text descriptions for each medication
        documents = []
        for idx, row in self.df.iterrows():
            # Create a comprehensive text description
            text = f"""
Medication: {row['NOM_COMMERCIAL']}
Active Ingredient (DCI): {row['DCI']}
Code: {row['CODE_PCT']}
Public Price: {row['PRIX_PUBLIC']} TND
Reference Price: {row['TARIF_REFERENCE']} TND
Category: {row['CATEGORIE']}
Authorization: {'Yes' if row['AP'] == 'O' else 'No'}
            """.strip()
            
            # Create metadata for filtering
            metadata = {
                'code': str(row['CODE_PCT']),
                'name': row['NOM_COMMERCIAL'],
                'dci': row['DCI'],
                'price': float(row['PRIX_PUBLIC']),
                'category': row['CATEGORIE'],
                'ap': row['AP']
            }
            
            documents.append(Document(page_content=text, metadata=metadata))
        
        print(f"✅ Created {len(documents)} medication documents")
        return documents
    
    def build_vectorstore(self, documents: List[Document]):
        """Create vector embeddings and store them."""
        print("🧠 Building vector embeddings...")
        
        # Use a lightweight embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="medications",
            persist_directory="./chroma_db"
        )
        
        print("✅ Vector store created!")
        
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant medications."""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Call setup() first.")
        
        results = self.vectorstore.similarity_search(query, k=k)
        
        # Format results
        formatted_results = []
        for doc in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        
        return formatted_results
    
    def setup(self):
        """Complete setup: load data, create embeddings, build vectorstore."""
        documents = self.load_and_process_data()
        self.build_vectorstore(documents)
        print("✅ RAG system ready!")
        
    def get_medication_by_name(self, name: str) -> Dict:
        """Direct lookup by medication name."""
        if self.df is None:
            self.df = pd.read_csv(self.csv_path)
        
        # Case-insensitive search
        result = self.df[self.df['NOM_COMMERCIAL'].str.contains(name, case=False, na=False)]
        
        if len(result) > 0:
            row = result.iloc[0]
            return {
                'name': row['NOM_COMMERCIAL'],
                'dci': row['DCI'],
                'price': row['PRIX_PUBLIC'],
                'reference_price': row['TARIF_REFERENCE'],
                'category': row['CATEGORIE'],
                'code': row['CODE_PCT']
            }
        return None
    
    def get_medications_by_ingredient(self, ingredient: str) -> List[Dict]:
        """Find medications by active ingredient."""
        if self.df is None:
            self.df = pd.read_csv(self.csv_path)
        
        results = self.df[self.df['DCI'].str.contains(ingredient, case=False, na=False)]
        
        medications = []
        for _, row in results.iterrows():
            medications.append({
                'name': row['NOM_COMMERCIAL'],
                'dci': row['DCI'],
                'price': row['PRIX_PUBLIC'],
                'code': row['CODE_PCT']
            })
        
        return medications
    
    def get_all_categories(self) -> List[str]:
        """Get list of all medication categories."""
        if self.df is None:
            self.df = pd.read_csv(self.csv_path)
        
        return self.df['CATEGORIE'].unique().tolist()
    
    def get_medications_by_category(self, category: str) -> List[Dict]:
        """Find medications by category."""
        if self.df is None:
            self.df = pd.read_csv(self.csv_path)
        
        results = self.df[self.df['CATEGORIE'] == category.upper()]
        
        medications = []
        for _, row in results.iterrows():
            medications.append({
                'name': row['NOM_COMMERCIAL'],
                'dci': row['DCI'],
                'price': row['PRIX_PUBLIC'],
                'category': row['CATEGORIE'],
                'code': row['CODE_PCT']
            })
        
        return medications
    
    def get_price_range(self, min_price: float = None, max_price: float = None) -> List[Dict]:
        """Find medications within a price range."""
        if self.df is None:
            self.df = pd.read_csv(self.csv_path)
        
        filtered = self.df
        
        if min_price is not None:
            filtered = filtered[filtered['PRIX_PUBLIC'] >= min_price]
        
        if max_price is not None:
            filtered = filtered[filtered['PRIX_PUBLIC'] <= max_price]
        
        medications = []
        for _, row in filtered.iterrows():
            medications.append({
                'name': row['NOM_COMMERCIAL'],
                'dci': row['DCI'],
                'price': row['PRIX_PUBLIC'],
                'code': row['CODE_PCT']
            })
        
        return medications