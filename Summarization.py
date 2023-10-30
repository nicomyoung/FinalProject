import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from community import community_louvain
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
import json
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from collections import defaultdict
import matplotlib.pyplot as plt

openai.api_key_path = "C:\\Final Project\\api_key.txt"

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
)


# Initialize the adaptive concurrency parameters
INITIAL_CONCURRENT_REQUESTS = 10  # Starting conservative
MAX_CONCURRENT_REQUESTS = 50      # This is the upper limit of concurrent requests
MIN_CONCURRENT_REQUESTS = 5       # This is the lower limit of concurrent requests
RATE_LIMIT_ERRORS_THRESHOLD = 3   # If we encounter 3 rate limit errors, we decrease concurrency
INCREMENT_AFTER_SUCCESSFUL_REQUESTS = 50  # Increase concurrency after 50 successful requests without rate limit errors

class AdaptiveConcurrency:
    def __init__(self):
        self.concurrent_requests = INITIAL_CONCURRENT_REQUESTS
        self.rate_limit_errors = 0
        self.successful_requests = 0

    def encountered_rate_limit_error(self):
        self.rate_limit_errors += 1
        if self.rate_limit_errors >= RATE_LIMIT_ERRORS_THRESHOLD:
            self.concurrent_requests = max(MIN_CONCURRENT_REQUESTS, self.concurrent_requests - 5)
            self.rate_limit_errors = 0  # Reset the count

    def successful_request(self):
        self.successful_requests += 1
        if self.successful_requests >= INCREMENT_AFTER_SUCCESSFUL_REQUESTS:
            self.concurrent_requests = min(MAX_CONCURRENT_REQUESTS, self.concurrent_requests + 5)
            self.successful_requests = 0  # Reset the count

    def get_concurrency_level(self):
        return self.concurrent_requests

adaptive_concurrency = AdaptiveConcurrency()




class TextProcessor:
    def __init__(self, chunk_size=500):
        self.chunk_size = chunk_size

    def chunk_text(self, text):
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def compute_similarity(self, chunks):
        vectorizer = TfidfVectorizer().fit_transform(chunks)
        vectors = vectorizer.toarray()
        return cosine_similarity(vectors)

    def detect_communities(self, similarity_matrix):
        G = nx.from_numpy_array(similarity_matrix)
        partition = community_louvain.best_partition(G, resolution=1.0)
        return partition

    
    @lru_cache(maxsize=1000)
    def summarize_chunk(self, chunk):
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Provide 2-3 concise bullet points summarizing the following."},
                {"role": "user", "content": chunk}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            adaptive_concurrency.successful_request()  # No rate-limit error, count the success
            return response.choices[0].message['content'].strip()

        except openai.error.OpenAIError as e:
            if 'rate limit' in str(e).lower():
                adaptive_concurrency.encountered_rate_limit_error()
            raise e

    @lru_cache(maxsize=1000)
    def generate_title(self, chunks):
        concatenated_text = ' '.join(chunks)
        messages = [{"role": "system", "content": "You are a helpful assistant. Provide a title for the following content."},
                    {"role": "user", "content": concatenated_text}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message['content'].strip()

    def get_sentiments(self, texts):
        results = sentiment_analyzer(texts)
        sentiments = []
        for result in results:
            sentiments.append(result['score'] if result['label'] == 'POSITIVE' else -result['score'])
        return sentiments

    def process(self, text):
        chunks = self.chunk_text(text)
        similarity_matrix = self.compute_similarity(chunks)
        partition = self.detect_communities(similarity_matrix)
        
        communities = defaultdict(list)
        for idx, community_id in partition.items():
            communities[community_id].append(chunks[idx])

        summarized_communities = {}
        sentiments = {}
        with ThreadPoolExecutor() as executor:
            for community_id, community_chunks in communities.items():
                summarized_texts = list(executor.map(self.summarize_chunk, community_chunks))
                titles = list(executor.map(self.generate_title, community_chunks))

                sentiment_scores = self.get_sentiments(summarized_texts)
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                sentiments[community_id] = avg_sentiment
                
                title = titles[0]
                summarized_communities[title] = {
                    'summaries': summarized_texts,
                    'avg_sentiment': avg_sentiment
                }

        overall_sentiment = sum(sentiments.values()) / len(sentiments)
        return summarized_communities, overall_sentiment
    
   

class FileProcessor:
    def __init__(self, input_dir, output_dir, num_files=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_files = num_files
        os.makedirs(output_dir, exist_ok=True)
        self.text_processor = TextProcessor()

    def process_files(self):
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.json')]
        limit = len(files) if not self.num_files else min(self.num_files, len(files))
        
        while files[:limit]:
            with ThreadPoolExecutor(max_workers=adaptive_concurrency.get_concurrency_level()) as executor:
                executor.map(self._process_single_file, files[:limit])
            # slicing off the already processed files
            files = files[adaptive_concurrency.get_concurrency_level():]


    def _process_single_file(self, file_name):
        try:
            with open(os.path.join(self.input_dir, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = data[-1].get("complete", "") if isinstance(data, list) else ""
                    
            summarized_communities, overall_sentiment = self.text_processor.process(text)
            
            output_data = {
                'file_name': file_name,
                'overall_sentiment': overall_sentiment,
                'topics': summarized_communities
            }

            output_file_name = os.path.join(self.output_dir, f"output_{file_name}")
            with open(output_file_name, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
                    
        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")


if __name__ == "__main__":
    fp = FileProcessor("C:\\Final Project\\jsonTranscriptions", "C:\\Final Project\\jsonOutput", 5)
    fp.process_files()