from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
import json
import sqlite3
from datetime import datetime
import os

class MessageHistoryStorage:
    def __init__(self, storage_type="json"):
        self.storage_type = storage_type
        if storage_type == "sqlite":
            self._init_db()
            
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect('message_history.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS conversations
                    (id TEXT PRIMARY KEY, 
                     timestamp TEXT,
                     agent_type TEXT,
                     messages TEXT)''')
        conn.commit()
        conn.close()

    def save_history(self, message_history, agent_type, conversation_id=None):
        """Save message history using specified storage method"""
        if conversation_id is None:
            conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Convert message history to dict format
        messages_dict = messages_to_dict(message_history)
        
        if self.storage_type == "json":
            self._save_to_json(messages_dict, agent_type, conversation_id)
        elif self.storage_type == "sqlite":
            self._save_to_sqlite(messages_dict, agent_type, conversation_id)
        
    def _save_to_json(self, messages_dict, agent_type, conversation_id):
        """Save messages to JSON file"""
        os.makedirs('message_histories', exist_ok=True)
        filename = f"message_histories/{agent_type}_{conversation_id}.json"
        
        data = {
            "conversation_id": conversation_id,
            "agent_type": agent_type,
            "timestamp": datetime.now().isoformat(),
            "messages": messages_dict
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
    def _save_to_sqlite(self, messages_dict, agent_type, conversation_id):
        """Save messages to SQLite database"""
        conn = sqlite3.connect('message_history.db')
        c = conn.cursor()
        
        c.execute("""INSERT INTO conversations (id, timestamp, agent_type, messages) 
                    VALUES (?, ?, ?, ?)""",
                    (conversation_id,
                     datetime.now().isoformat(),
                     agent_type,
                     json.dumps(messages_dict)))
        
        conn.commit()
        conn.close()
        
    def load_history(self, conversation_id, agent_type=None):
        """Load message history from storage"""
        if self.storage_type == "json":
            filename = f"message_histories/{agent_type}_{conversation_id}.json"
            with open(filename, 'r') as f:
                data = json.load(f)
                return messages_from_dict(data["messages"])
        
        elif self.storage_type == "sqlite":
            conn = sqlite3.connect('message_history.db')
            c = conn.cursor()
            c.execute("SELECT messages FROM conversations WHERE id = ?", (conversation_id,))
            result = c.fetchone()
            conn.close()
            
            if result:
                return messages_from_dict(json.loads(result[0]))
            return None