import os
import json
from google.cloud import storage, speech_v1p1beta1 as speech
from google.cloud.exceptions import Conflict

# Set Google Cloud credentials and initialize clients
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Final Project/key.json"
storage_client = storage.Client(project="sigma-outlook-400223")  # specifying the project
speech_client = speech.SpeechClient()

# Configurations
bucket_name = "fp-mp3-bucket"  # specifying the bucket name
local_directory = "C:/Final Project/localAudios/wavFiles"
json_output_directory = "C:/Final Project/jsonTranscriptions"
if not os.path.exists(json_output_directory):
    os.makedirs(json_output_directory)

# 2. Upload WAV files to GCS
def upload_files():
    bucket = storage_client.bucket(bucket_name)
    for filename in os.listdir(local_directory):
        if filename.endswith(".wav"):
            blob = bucket.blob(filename)
            blob.upload_from_filename(os.path.join(local_directory, filename))
            print(f"File {filename} uploaded to {bucket_name}.")

def transcribe_audio(gcs_uri):
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="en-US",
        audio_channel_count = 2,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        enable_speaker_diarization=True,
        diarization_speaker_count=2  # might need to adjust this
    )

    operation = speech_client.long_running_recognize(config=config, audio=audio)
    response = operation.result()

    transcription_data = []
    complete_transcription = ""
    
    for result in response.results:
        for word_info in result.alternatives[0].words:
            word = word_info.word
            start_time = word_info.start_time.total_seconds()
            if hasattr(word_info, "speaker_tag"):
                speaker_tag = word_info.speaker_tag
            else:
                speaker_tag = None
            transcription_data.append({
                "start_time": start_time,
                "word": word,
                "speaker_tag": speaker_tag
            })
            complete_transcription += " " + word

    transcription_data.append({"complete": complete_transcription.strip()})
    
    # Extracting file name from the URI
    file_name = gcs_uri.split("/")[-1]
    json_filename = os.path.splitext(file_name)[0] + ".json"
    json_file_path = os.path.join(json_output_directory, json_filename)
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(transcription_data, json_file, ensure_ascii=False, indent=4)

def process_bucket():
    """ checks if the file exists before running the transcription """
    for blob in storage_client.list_blobs(bucket_name):
        if blob.name.endswith(".wav"):
            json_filename = os.path.splitext(blob.name)[0] + ".json"
            json_file_path = os.path.join(json_output_directory, json_filename)
            
            if not os.path.exists(json_file_path):
                gcs_uri = f"gs://{bucket_name}/{blob.name}"
                transcription_data = transcribe_audio(gcs_uri)
                # further processing of transcription_data if needed
                print(f"Processed and saved transcription for {blob.name}")
            else:
                print(f"Transcription for {blob.name} already exists, skipping...")


def process_single_file(file_name):
    gcs_uri = f"gs://{bucket_name}/{file_name}"
    transcription_data = transcribe_audio(gcs_uri)
    # save or further process transcription_data

# You can call process_bucket() for batch processing or process_single_file(file_name) for individual files.

# Main execution
if __name__ == "__main__":
   # upload_files()
    process_bucket() 
