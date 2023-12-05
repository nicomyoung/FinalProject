import os
import json
import openai
from concurrent.futures import ThreadPoolExecutor
import mysql.connector
from mysql.connector import Error

openai.api_key_path = "C:\\Final Project\\api_key.txt"

class GenerativeAssistant:

    def __init__(self, db_config):
        self.db_config = db_config

    def db_connect(self):
        try:
            return mysql.connector.connect(**self.db_config)
        except Error as e:
            print(f"Error connecting to MySQL database: {e}")
            return None

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

    def store_response(self, summary_detail_id, response):
        connection = self.db_connect()
        try:
            cursor = connection.cursor()
            query = "INSERT INTO generative_responses (summary_detail_id, response) VALUES (%s, %s)"
            cursor.execute(query, (summary_detail_id, response))
            connection.commit()
        except Error as e:
            print(f"Error while inserting generative response: {e}")
        finally:
            if connection:
                connection.close()

    def process_responses(self):
        connection = self.db_connect()
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT id, title, bullet_point FROM summary_details")
            summary_details = cursor.fetchall()

            for detail in summary_details:
                response = self.generate_response(detail['title'], [detail['bullet_point']])
                if response:
                    self.store_response(detail['id'], response[1])
        except Error as e:
            print(f"Error while fetching summary details: {e}")
        finally:
            if connection:
                connection.close()
    
    def process_file(self, file_name):
        connection = self.db_connect()
        try:
            cursor = connection.cursor(dictionary=True)
            query = """
                SELECT sd.id, sd.title, sd.bullet_point
                FROM summary_details sd
                JOIN summaries s ON sd.summary_id = s.id
                WHERE s.file_name = %s
            """
            cursor.execute(query, (file_name,))
            summary_details = cursor.fetchall()

            for detail in summary_details:
                response = self.generate_response(detail['title'], [detail['bullet_point']])
                if response:
                    self.store_response(detail['id'], response[1])
        except Error as e:
            print(f"Error while fetching summary details: {e}")
        finally:
            if connection:
                connection.close()
    
    
    def process_bullet_point(self, summary_detail_id):
        connection = self.db_connect()
        try:
            cursor = connection.cursor(dictionary=True)
            query = "SELECT title, bullet_point FROM summary_details WHERE id = %s"
            cursor.execute(query, (summary_detail_id,))
            detail = cursor.fetchone()
            response = self.generate_response(detail['title'], [detail['bullet_point']])
            if response:
                self.store_response(summary_detail_id, response[1])
        except Error as e:
            print(f"Error while processing bullet point: {e}")
        finally:
            if connection:
                connection.close()

    def process_title(self, title):
        connection = self.db_connect()
        try:
            cursor = connection.cursor(dictionary=True)
            query = "SELECT id, bullet_point FROM summary_details WHERE title = %s"
            cursor.execute(query, (title,))
            details = cursor.fetchall()
            bullet_points = [detail['bullet_point'] for detail in details]
            response = self.generate_response(title, bullet_points)
            if response:
                for detail in details:
                    self.store_response(detail['id'], response[1])
        except Error as e:
            print(f"Error while processing title: {e}")
        finally:
            if connection:
                connection.close()





if __name__ == "__main__":
    db_config = {
        host='localhost',
            database=os.getenv('MYSQL_DATABASE', 'chatr'),
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASS'),
    }
    assistant = GenerativeAssistant(db_config)
    
