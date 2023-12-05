import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from community import community_louvain
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
import json
from transformers import pipeline
#from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from collections import defaultdict
import matplotlib.pyplot as plt
import mysql.connector
from mysql.connector import Error
from nltk import sent_tokenize
import spacy
from spacy.matcher import Matcher

# WORK ON LOCAL SUMMARIZATION STUFF 
# GET SOME TYPE OF QUALITATIVE RESPONSE
# SHOW PARTICULAR OUTPUTS AND SPECIFICS
# PUTTING TOGETHER A WRITE-UP FOR POTENTIAL EMPLOYERS
# SOFTWARE DEVELOPER 
# ASR DOMAIN GENERALIZATION - GENRE is RADIO BROADCASTERS - TRAIN MODEL ON NYT WOULDNT WORK ON WSJ 
#   NOW ITS MORE ROBUST _ RADIO BROADCASTER IS NOT FRIENDS AT A RESTAURANT BACKGROUND NOISE, CLEAR SPEECH, ETC. CONVERSATION TOPIC CLIPS VS SCRIPTED KNOWLEDGE 


nlp = spacy.load("en_core_web_sm")

def db_connect():
    try:
        return mysql.connector.connect(
            host='localhost',
            database=os.getenv('MYSQL_DATABASE', 'chatr'),
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASS'),
        )
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None
    

openai.api_key_path = "C:\\Final Project\\api_key.txt"

cardinality_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
)


emotion_classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=2)

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

    def chunk_text(self, text, max_chunk_size=512):
        """
        This method creates chunks based on semantic similarity of sentences.
        Sentences are embedded and then grouped together if they are semantically similar,
        ensuring that chunks are as close to the max_chunk_size as possible without exceeding it.
        """
        # Tokenize the document into sentences
        sentences = sent_tokenize(text)
        sentence_embeddings = [nlp(sentence).vector for sentence in sentences]
        
        chunks = []
        current_chunk = []
        current_chunk_embedding = []
        
        for sentence, embedding in zip(sentences, sentence_embeddings):
            # If adding the sentence doesn't exceed the max chunk size, or if the chunk is empty (ensure at least one sentence per chunk)
            if len(current_chunk) == 0 or len(' '.join(current_chunk) + ' ' + sentence) <= max_chunk_size:
                current_chunk.append(sentence)
                current_chunk_embedding.append(embedding)
            else:
                # Compare the embedding of the current sentence to the average embedding of the current chunk
                avg_chunk_embedding = np.mean(current_chunk_embedding, axis=0)
                similarity = cosine_similarity([embedding], [avg_chunk_embedding])
                
                # If the sentence is semantically similar to the current chunk, add it to the current chunk
                if similarity > 0.7:  # This threshold can be tuned
                    current_chunk.append(sentence)
                    current_chunk_embedding.append(embedding)
                else:
                    # Otherwise, start a new chunk
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_chunk_embedding = [embedding]
        
        # Don't forget to add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

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
                {"role": "system", "content": "You are a helpful assistant. Provide concise bullet points summarizing the following, limited to three bullet points maximum, each no longer than two sentences. Avoid sponsorship plugs."},
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
        results = cardinality_analyzer(texts)
        sentiments = []
        for result in results:
            sentiments.append(result['score'] if result['label'] == 'POSITIVE' else -result['score'])
        return sentiments
    
    def extract_key_sentiments(self, text):
        """
        Analyze the sentiment of text, extracting the two most prominent emotions.
        Returns a string of the two key emotions.
        """
        # Analyze the entire text using the emotion classifier
        model_outputs = emotion_classifier(text)

        # Check if there are at least two emotions detected
        if len(model_outputs) > 0 and len(model_outputs[0]) >= 2:
            # Extract labels from the first two dictionaries
            first_emotion = model_outputs[0][0]["label"]
            second_emotion = model_outputs[0][1]["label"]
            return f'{first_emotion}, {second_emotion}'
        else:
            return model_outputs[0][0]["label"]



    def extract_aspects(self, doc):
        """
        Extract key aspects or topics from the text using NLP techniques.
        """
        aspects = []
        # Use spaCy's Matcher or Dependency Parser to extract aspects
        # Example: Extracting nouns and compound nouns as aspects
        for chunk in doc.noun_chunks:
            aspects.append(chunk.text)
        return aspects

    def analyze_aspect_sentiments(self, aspects, text):
        """
        Analyze sentiment for each aspect extracted from the text.
        """
        aspect_sentiments = {}
        for aspect in aspects:
            # Refine sentiment analysis for each aspect using the new model
            model_output = emotion_classifier(aspect)
            aspect_sentiments[aspect] = model_output[0]
        return aspect_sentiments


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
        #print("Summarized Communities, ", summarized_communities)

        overall_sentiment = sum(sentiments.values()) / len(sentiments)
        return summarized_communities, overall_sentiment
    




class FileProcessor:
   
    def __init__(self, num_files=None):
        self.num_files = num_files
        self.text_processor = TextProcessor()

    def get_transcript(self):
        connection = db_connect()
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT t.id, t.file_name, t.transcription
                FROM transcriptions t
                LEFT JOIN summaries s ON t.file_name = s.file_name
                WHERE s.id IS NULL
            """)
            return cursor.fetchall()
        except mysql.connector.Error as e:
            print(f"Error fetching transcriptions: {e}")
        finally:
            if connection:
                connection.close()


    def process_transcriptions(self):
        transcriptions = self.get_transcript()
        limit = len(transcriptions) if not self.num_files else min(self.num_files, len(transcriptions))
        
        # Using ThreadPoolExecutor to process in parallel
        with ThreadPoolExecutor(max_workers=adaptive_concurrency.get_concurrency_level()) as executor:
            for transcription in transcriptions[:limit]:
                executor.submit(self._process_single_transcription, transcription)
    
    def _process_single_transcription(self, transcription):
        file_name = transcription['file_name']
        text = transcription['transcription']
        print(f"Starting processing for file: {file_name}")

        summarized_communities, overall_sentiment = self.text_processor.process(text)
        summary_id = self.insert_summary(file_name, overall_sentiment)
        #print("Summary ID: ", summary_id)

        for title, details in summarized_communities.items():
            #print("Title: ", title)
            # Aggregate all bullet points into one text for sentiment analysis
            full_text = ' '.join(details['summaries'])
            key_sentiments = self.text_processor.extract_key_sentiments(full_text)
            sentiment_score = details['avg_sentiment']
            #print("Sentiment Score: ", sentiment_score)
            #print("Key Sentiments: ", key_sentiments)
            
            # Process each bullet point for storage, but use the aggregated sentiment
            for bullet_point in details['summaries']:
                self.insert_summary_details(summary_id, title, bullet_point, sentiment_score, key_sentiments)
        print(f"Finished processing file: {file_name}")

    def insert_summary(self, file_name, overall_sentiment):
        connection = db_connect()
        try:
            cursor = connection.cursor()
            cursor.execute("""
                INSERT INTO summaries (file_name, overall_sentiment)
                VALUES (%s, %s)
            """, (file_name, overall_sentiment))
            connection.commit()
            return cursor.lastrowid
        except mysql.connector.Error as e:
            print(f"Error while inserting summary: {e}")
        finally:
            if connection:
                connection.close()

    def insert_summary_details(self, summary_id, title, bullet_point, sentiment_score, key_sentiments):
        connection = db_connect()
        try:
            cursor = connection.cursor()
            cursor.execute("""
                INSERT INTO summary_details (summary_id, title, bullet_point, sentiment_score, key_sentiments)
                VALUES (%s, %s, %s, %s, %s)
            """, (summary_id, title, bullet_point, sentiment_score, key_sentiments))
            connection.commit()
        except mysql.connector.Error as e:
            print(f"Error while inserting summary detail: {e}")
        finally:
            if connection:
                connection.close()




if __name__ == "__main__":
    dp = FileProcessor(15)
    dp.process_transcriptions()
