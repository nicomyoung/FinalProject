import os
import json
from google.cloud import storage, speech_v1p1beta1 as speech
import mysql.connector
from mysql.connector import Error
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define constants
BUCKET_NAME = "fp-mp3-bucket"
LOCAL_DIRECTORY = "C:/Final Project/localAudios/wavFiles"


# Initialize Google Cloud clients
storage_client = storage.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
speech_client = speech.SpeechClient()




# Database Utility Functions
def db_connect():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database=os.getenv('MYSQL_DATABASE', 'chatr'),
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASSWORD')
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        raise e


def insert_transcription(file_name, transcription):
    try:
        connection = db_connect()
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO transcriptions (file_name, transcription)
                VALUES (%s, %s)
            """, (file_name, transcription))
            connection.commit()
    except Error as e:
        print(f"Error while inserting transcription: {e}")
    finally:
        if connection:
            connection.close()


def check_if_transcribed(file_name):
    try:
        connection = db_connect()
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM transcriptions WHERE file_name = %s
                )
            """, (file_name,))
            result = cursor.fetchone()
            return result[0]
    except Error as e:
        print(f"Error checking if transcription exists: {e}")
        return False
    finally:
        if connection:
            connection.close()

# Transcription and File Processing Classes
class AudioTranscription:
    def __init__(self):
        self.storage_client = storage.Client(project="sigma-outlook-400223") 
        self.speech_client = speech.SpeechClient()
        self.bucket_name = BUCKET_NAME
        

    def upload_files(self, local_directory):
        bucket = self.storage_client.bucket(self.bucket_name)
        for filename in os.listdir(local_directory):
            if filename.endswith(".wav"):
                blob = bucket.blob(filename)
                blob.upload_from_filename(os.path.join(local_directory, filename))
                print(f"File {filename} uploaded to {self.bucket_name}.")

    def transcribe_audio(self, gcs_uri):
        audio = speech.RecognitionAudio(uri=gcs_uri)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="en-US",
            audio_channel_count=2,
            enable_automatic_punctuation=True,
        )

        operation = self.speech_client.long_running_recognize(config=config, audio=audio)
        response = operation.result()

        complete_transcription = ""
        
        for result in response.results:
            # Concatenate all transcriptions from the audio file
            complete_transcription += result.alternatives[0].transcript + " "
        
        return gcs_uri.split('/')[-1], complete_transcription.strip()
    
    def process_single_file(self, file_name):
        """Processes a single audio file for transcription."""
        if not check_if_transcribed(file_name):
            gcs_uri = f"gs://{self.bucket_name}/{file_name}"
            file_name, complete_transcription = self.transcribe_audio(gcs_uri)
            insert_transcription(file_name, complete_transcription)
            print(f"Inserted transcription for {file_name} into the database.")
        else:
            print(f"Transcription for {file_name} already exists, skipping...")

    def process_bucket(self):
        """Processes all audio files in the bucket for transcription using multiple threads."""
        futures = []
        with ThreadPoolExecutor() as executor:
            for blob in self.storage_client.list_blobs(self.bucket_name):
                if blob.name.endswith(".wav"):
                    futures.append(executor.submit(self.process_single_file, blob.name))
        for future in as_completed(futures):
            try:
                future.result()  # This will re-raise any exception raised during execution
            except Exception as e:
                print(f"An error occurred: {e}")


# Main execution
if __name__ == "__main__":
    transcription_processor = AudioTranscription()
    # transcription_processor.upload_files(LOCAL_DIRECTORY)
    transcription_processor.process_bucket()
    # Optionally call transcription_processor.process_single_file("specific_file.wav") 
