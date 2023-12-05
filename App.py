from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource
import mysql.connector
from mysql.connector import Error
import os
import json
from Summarization import FileProcessor
from GenerativeAssistant import GenerativeAssistant

app = Flask(__name__)
CORS(app)
api = Api(app)

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


class SummaryList(Resource):
    def get(self):
        connection = db_connect()
        if not connection:
            return {"message": "Database connection failed"}, 500

        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT file_name FROM summaries")
            summaries = cursor.fetchall()
            return jsonify([summary['file_name'] for summary in summaries])
        except Error as e:
            print(f"Error fetching summaries: {e}")
            return {"message": "Failed to fetch summaries"}, 500
        finally:
            if connection:
                connection.close()

class Summary(Resource):
    def get(self, file_name):
        connection = db_connect()
        if not connection:
            return {"message": "Database connection failed"}, 500

        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM summaries WHERE file_name = %s", (file_name,))
            summary = cursor.fetchone()
            return jsonify(summary) if summary else {"message": "File not found"}, 404
        except Error as e:
            print(f"Error fetching summary: {e}")
            return {"message": "Failed to fetch summary"}, 500
        finally:
            if connection:
                connection.close()

class SummarizeTranscription(Resource):
    def post(self):
        uploaded_file = request.files['transcription']
        save_path = os.path.join("C:\\Final Project\\transcriptions", uploaded_file.filename)
        uploaded_file.save(save_path)

        # Process transcription and save summaries to the database
        fp = FileProcessor(num_files=1)
        fp.process_transcriptions()

        return jsonify({"message": "File processed successfully"})

class GenerateDiscussion(Resource):
    def post(self):
        title = request.json['title']
        summary_detail_id = request.json['summary_detail_id']

        assistant = GenerativeAssistant(db_config)

        # Generate discussion for the specified bullet point
        response = assistant.process_bullet_point(summary_detail_id)

        return jsonify({"response": response})

class SummaryHistory(Resource):
    def get(self):
        connection = db_connect()
        if not connection:
            return {"message": "Database connection failed"}, 500

        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT file_name, creation_time FROM summaries ORDER BY creation_time DESC")
            summaries = cursor.fetchall()
            return jsonify(summaries)
        finally:
            if connection:
                connection.close()

class SentimentGraphData(Resource):
    def get(self):
        connection = db_connect()
        if not connection:
            return {"message": "Database connection failed"}, 500

        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT file_name, overall_sentiment, creation_time FROM summaries ORDER BY creation_time")
            sentiment_data = cursor.fetchall()
            return jsonify(sentiment_data)
        finally:
            if connection:
                connection.close()

# Add endpoints to the API
api.add_resource(SummarizeTranscription, "/summarize")
api.add_resource(GenerateDiscussion, "/generate")
api.add_resource(SummaryList, "/summaries")
api.add_resource(Summary, "/summary/<string:file_name>")
api.add_resource(SummaryHistory, "/summary-history")
api.add_resource(SentimentGraphData, "/sentiment-graph-data")

if __name__ == "__main__":
    app.run(debug=True)
