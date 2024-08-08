__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
import hmac
import hashlib
import base64
from datetime import datetime, timedelta, timezone
from typing import List, Dict
import chromadb
from openai import OpenAI
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
from flask import Flask, request, abort, jsonify
from io import StringIO

# 常數定義
CHROMA_DB = 'Cofit211-cosine'
CHAT_MODEL_NAME = "gpt-4o-mini"
GCS_BUCKET_NAME = 'ian-line-bot-files'
GCS_FILE_PATH = 'messages-health.csv'

# Line Bot 設定
line_bot_api = LineBotApi(os.environ.get('CHANNEL_ACCESS_TOKEN'))
channel_secret = os.environ.get('CHANNEL_SECRET')
handler = WebhookHandler(channel_secret)

# OpenAI 設定
client = OpenAI(
    api_key = os.environ.get('OPENAI_API_KEY'),
)

# DeepInfra 設定
deepinfra_client = OpenAI(
    api_key = os.environ.get('DEEPINFRA_API_KEY'),
    base_url = "https://api.deepinfra.com/v1/openai",
)

app = Flask(__name__)

def get_embedding(text: str) -> List[float]:
    embeddings = deepinfra_client.embeddings.create(
        model="BAAI/bge-m3",
        input=text,
        encoding_format="float"
    )
    return embeddings.data[0].embedding

def configure_retriever():
    client = chromadb.PersistentClient(path='./')
    collection = client.get_collection(name=CHROMA_DB)
    return collection

def get_relevant_documents(query: str, collection, top_k: int = 3) -> List[Dict]:
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    documents = [
        {"page_content": doc, "metadata": meta} 
        for doc, meta in zip(results['documents'][0], results['metadatas'][0])
    ]
    return documents

def generate_response(messages: List[Dict]) -> str:
    try:
        chat_completion = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=messages,
            temperature=0
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API 錯誤: {str(e)}")
        return f"抱歉,發生了一個錯誤: {str(e)}"

def log_message_to_gcs(user_input, gpt_output, chat_room_id, blob):
    taipei_time = datetime.now(timezone.utc) + timedelta(hours=8)
    
    new_row = {
        "timestamp": taipei_time.strftime('%Y-%m-%d %H:%M:%S'),
        "chat_room_id": chat_room_id,
        "user_input": user_input,
        "gpt_output": gpt_output
    }
    
    if blob.exists():
        content = blob.download_as_text()
        df = pd.read_csv(StringIO(content))
    else:
        df = pd.DataFrame(columns=["timestamp", "chat_room_id", "user_input", "gpt_output"])
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    blob.upload_from_string(df.to_csv(index=False), 'text/csv')

def get_chat_history(chat_room_id, blob):
    if blob.exists():
        content = blob.download_as_text()
        df = pd.read_csv(StringIO(content))
        
        chat_history = df[df['chat_room_id'] == chat_room_id].tail(5)
        
        formatted_history = []
        for _, row in chat_history.iterrows():
            if row['gpt_output'] == '。':
                formatted_history = []
            else:
                formatted_history.append(f"時間: {row['timestamp']}")
                formatted_history.append(f"使用者: {row['user_input']}")
                formatted_history.append(f"AI: {row['gpt_output']}")
                formatted_history.append("---")
            
        formatted_history_embedding_use = []
        if not chat_history.empty:
            last_row = chat_history.iloc[-1]
            formatted_history_embedding_use.append(f"使用者: {last_row['user_input']}")
            formatted_history_embedding_use.append(f"AI: {last_row['gpt_output']}")
        
        return "\n".join(formatted_history), "\n".join(formatted_history_embedding_use)
    else:
        return "", ""

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_input = event.message.text
    chat_room_id = event.source.group_id if hasattr(event.source, 'group_id') else event.source.user_id

    client = storage.Client()
    bucket = client.get_bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_FILE_PATH)
    
    if user_input == 'reset':
        user_input = '。'
        GPT_output = '。'
    else:
        chat_history, chat_history_embedding_use = get_chat_history(chat_room_id, blob)
        
        collection = configure_retriever()
        
        cumulative_query = chat_history_embedding_use + '\n' + user_input
        relevant_docs = get_relevant_documents(cumulative_query, collection, top_k=3)
        context = relevant_docs[0]['page_content'] if relevant_docs else ""

        messages_for_ai = [
            {"role": "system", "content": f"你是一位專業的健康、醫療和飲食相關的諮詢師。用繁體中文回覆。請使用以下背景資訊來回答問題:\n\n{context}"},
            {"role": "user", "content": f"之前的對話紀錄：\n{chat_history}\n\n使用者的最新問題：{user_input}"}
        ]
        
        GPT_output = generate_response(messages_for_ai)
        
        # 添加 YouTube URL 到回覆中
        youtube_urls = []
        for doc in relevant_docs:
            if 'video_id' in doc['metadata']:
                video_id = doc['metadata']['video_id']
                youtube_url = f"https://www.youtube.com/watch?v={video_id}"
                youtube_urls.append(youtube_url)
        
        if youtube_urls:
            GPT_output += "\n\n推薦影片：\n" + "\n".join(youtube_urls)

    log_message_to_gcs(user_input, GPT_output, chat_room_id, blob)
    
    line_bot_api.reply_message(event.reply_token, TextSendMessage(GPT_output))

    return 'OK'

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))