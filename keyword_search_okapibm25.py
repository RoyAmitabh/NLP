import numpy as np
import re
from collections import Counter
from typing import List, Dict

class CustomTermScoring:
    def __init__(self, docs: List[str], k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [self._tokenize(doc) for doc in docs]
        self.doc_lens = np.array([len(doc) for doc in self.tokenized_docs])  # Document lengths
        self.avg_doc_len = np.mean(self.doc_lens)  # Average document length
        
        # Build inverted index & document frequencies
        self.inverted_index = {}
        self.doc_freqs = Counter()
        
        for i, doc in enumerate(self.tokenized_docs):
            unique_terms = set(doc)
            for term in unique_terms:
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append(i)
                self.doc_freqs[term] += 1
        
        self.total_docs = len(docs)
        
        # Compute IDF values
        self.idf = {
            term: np.log((self.total_docs - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in self.doc_freqs.items()
        }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes text by removing punctuation, converting to lowercase, and splitting on word boundaries."""
        return re.findall(r'\b\w+\b', text.lower())

    def get_okapi_score(self, doc_length, query: str) -> List[float]:
        tokenized_query = self._tokenize(query)
        print(f"query tokens = {tokenized_query}")
        scores = np.zeros(doc_length)

        for term in tokenized_query:
            if term not in self.idf:
                continue  # Skip terms not in index
            
            idf_term = self.idf[term]
            for doc_idx in self.inverted_index.get(term, []):
                term_freq = self.tokenized_docs[doc_idx].count(term)
                doc_len = self.doc_lens[doc_idx]
                
                # Compute BM25 formula
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_idx] += idf_term * (numerator / denominator)
        
        return scores.tolist()

if __name__ == "__main__":

    # Dictionary of products with names and descriptions
    products: Dict[str, str] = {
        "Sony WH-1000XM5": "Premium noise-canceling headphones with 30-hour battery life.",
        "Bose QuietComfort 45": "High-quality noise-canceling headphones with long battery life.",
        "Apple AirPods Max": "Over-ear headphones with active noise cancellation and high-fidelity sound.",
        "Jabra Elite 85h": "Smart noise-canceling headphones with water resistance and long battery.",
        "Beats Studio3": "Wireless noise-canceling headphones with Apple W1 chip for seamless connectivity.",
        "Sennheiser Momentum 4": "Audiophile-grade headphones with adaptive noise cancellation.",
        "Anker Soundcore Life Q35": "Affordable ANC headphones with long battery life and LDAC support.",
        "Razer BlackShark V2": "Gaming headset with noise-isolating memory foam ear cups and THX Spatial Audio.",
        "SteelSeries Arctis Pro": "Gaming headphones with high-res audio and noise-canceling mic.",
        "HyperX Cloud II": "Comfortable gaming headset with 7.1 surround sound and noise-isolating ear cups.",
    }

    # Extract product names and descriptions
    product_names = list(products.keys())
    product_descriptions = list(products.values())

    # Initialize BM25 model
    bm25 = CustomTermScoring(product_descriptions)

    # Query to search
    query = "best noise-canceling headphones with long battery life"
    
    # Compute BM25 scores for the query
    scores = bm25.get_okapi_score(len(product_descriptions), query)

   

    #Rank and display results sorted by relevance (highest score first)
    ranked_results = sorted(zip(product_names, scores), key=lambda x: x[1], reverse=True)

    print("\n🔹 Search Results (Ranked by Relevance):\n")
    for rank, (product, score) in enumerate(ranked_results, start=1):
        print(f"{rank}. {product} - Score: {score:.4f}")
