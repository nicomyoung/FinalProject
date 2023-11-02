import os
import json
import openai
from concurrent.futures import ThreadPoolExecutor

openai.api_key_path = "C:\\Final Project\\api_key.txt"

class GenerativeAssistant:

    def __init__(self, summary_path):
        self.summary_path = summary_path
        self.summaries = {}

        # Check if the provided path is a directory or a single file
        if os.path.isdir(summary_path):
            self.files = [os.path.join(summary_path, f) for f in os.listdir(summary_path) if f.startswith('output_')]
        elif os.path.isfile(summary_path):
            self.files = [summary_path]
        else:
            raise ValueError(f"Provided path {summary_path} is neither a directory nor a file.")

    def generate_response(self, title, bullet_points):
        topic_content = f"{title}\n"
        for point in bullet_points:
            topic_content += f"{point}\n"
        
        messages = [
            {"role": "system", "content": "You are a helpful analyst. Further the discussion on the following topic with 2 insightful questions related to the topic at hand"},
            {"role": "user", "content": topic_content}
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            return (title, response.choices[0].message['content'].strip())
        except Exception as e:
            print(f"An error occurred while generating response for title: {title}. Error: {e}")
            return (title, None)

    def process_single_topic(self, title, bullet_points):
        _, response = self.generate_response(title, bullet_points)
        return response

    def process_single_file(self, file_path):
        responses = {}
        
        # Use ThreadPoolExecutor for concurrent processing of topics
        with ThreadPoolExecutor() as executor:
            future_to_title = {}
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for title, details in data['topics'].items():
                    bullet_points = details['summaries']
                    future = executor.submit(self.generate_response, title, bullet_points[0])  # using the first summary for now
                    future_to_title[future] = title

                for future in future_to_title:
                    title, response = future.result()
                    if response:
                        responses[title] = response
            except Exception as e:
                print(f"An error occurred while processing summary file: {file_path}. Error: {e}")

        return responses

    def process_all_files(self):
        for file in self.files:
            self.summaries[file] = self.process_single_file(file)

    def display_responses(self):
        for file, responses in self.summaries.items():
            print(f"\nFile: {file}\n{'=' * 40}")
            for title, response in responses.items():
                print(f"Title: {title}")
                print(f"Response: {response}\n")


if __name__ == "__main__":
    assistant = GenerativeAssistant("C:\\Final Project\\jsonOutput")

    # Uncomment below for single topic analysis
    # response = assistant.process_single_topic("Some Title", ["Point 1", "Point 2"])
    # print(response)

    # Uncomment below for single file analysis
    responses = assistant.process_single_file("C:\\Final Project\\jsonOutput\\output_podcast_20230417_nprpolitics_c38035b4-0e7c-4160-882b-25dc3a90a82c.json")
    for title, response in responses.items():
        print(f"Title: {title}")
        print(f"Response: {response}\n")

    # Uncomment below for directory analysis (all files)
    # assistant.process_all_files()
    # assistant.display_responses()
