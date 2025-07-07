from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from src.utils.data_loader import load_course_data
from src.tools.course_retriever import build_vector_store
from src.models.course_bot_chain import (
    build_recommendation_chain,
    build_interest_conversation_chain,
    build_extract_interest_chain
)
from src.utils.conversation_state import ConversationState
import os
import json

from langchain.memory import ConversationBufferWindowMemory

load_dotenv()
app = Flask(__name__)

docs = load_course_data()
vector_store = build_vector_store(docs)

user_states = {}
user_memories = {}

MAX_MEMORY_MESSAGES = 100
MIN_COUNSELOR_TURNS = 5

def get_user_id():
    return "user_001"

def get_user_state(user_id: str) -> ConversationState:
    if user_id not in user_states:
        user_states[user_id] = ConversationState()
    return user_states[user_id]

def get_user_memory(user_id: str) -> ConversationBufferWindowMemory:
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=MAX_MEMORY_MESSAGES
        )
    return user_memories[user_id]

def format_chat_history(memory):
    chat_history_str = ""
    for msg in memory.chat_memory.messages:
        role = "Student" if msg.type == "human" else "Bot"
        chat_history_str += f"{role}: {msg.content}\n"
    return chat_history_str

def should_offer_course_recommendation(user_state):
    return (user_state.interest_turns >= MIN_COUNSELOR_TURNS and 
            user_state.get_interests() and 
            not getattr(user_state, 'has_offered_recommendation', False))

def add_counselor_prompt(bot_response, user_state):
    if should_offer_course_recommendation(user_state):
        user_state.has_offered_recommendation = True
        return bot_response + "\n\nBased on our conversation, I've learned quite a bit about your interests. Would you like me to recommend some courses that might be perfect for you?"
    return bot_response

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/set_grade", methods=["POST"])
def set_grade():
    data = request.get_json()
    if not data or 'grade' not in data:
        return jsonify({"error": "Grade is required"}), 400

    try:
        grade = int(data.get("grade"))
        if not 8 <= grade <= 12:
            return jsonify({"error": "Invalid grade input. Please provide a grade between 8 and 12."}), 400
    except ValueError:
        return jsonify({"error": "Grade must be a valid number."}), 400

    user_id = get_user_id()
    user_state = get_user_state(user_id)
    user_state.set_grade(grade)
    return jsonify({"message": f"Grade set to {grade}"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message")
    if not message:
        return jsonify({"error": "Message is required"}), 400

    user_id = get_user_id()
    user_state = get_user_state(user_id)
    if not user_state.grade:
        return jsonify({"response": "Please set your grade first."}), 400

    memory = get_user_memory(user_id)
    memory.chat_memory.add_user_message(message)
    chat_history_str = format_chat_history(memory)

    user_message_lower = message.lower()
    
    if ("yes" in user_message_lower or "sure" in user_message_lower or "recommend" in user_message_lower) and getattr(user_state, 'has_offered_recommendation', False):
        user_state.has_offered_recommendation = False
        base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        recommendation_chain = build_recommendation_chain(base_retriever)
        credit_type = data.get("credit_type", "any")

        try:
            recommendation = recommendation_chain.invoke({
                "question": "recommend courses based on interests",
                "grade": user_state.grade,
                "interests": user_state.get_interests(),
                "credit_type": credit_type
            })
        except Exception as e:
            print(f"Error in recommendation chain: {e}")
            recommendation = "I'm having trouble finding course recommendations right now. Could you tell me more about what subjects interest you most?"

        memory.chat_memory.add_ai_message(recommendation)
        save_chat_log(user_id, memory.chat_memory.messages)
        return jsonify({"response": recommendation})

    course_keywords = [
        "course", "credit", "class", "subject", "recommend", "suggest", 
        "dual credit", "english", "science", "math", "history", "art",
        "elective", "graduation", "requirements", "what should i take"
    ]

    if any(keyword in user_message_lower for keyword in course_keywords):
        base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        recommendation_chain = build_recommendation_chain(base_retriever)
        credit_type = data.get("credit_type", "any")

        try:
            recommendation = recommendation_chain.invoke({
                "question": message,
                "grade": user_state.grade,
                "interests": user_state.get_interests(),
                "credit_type": credit_type
            })
        except Exception as e:
            print(f"Error in recommendation chain: {e}")
            recommendation = "I'm having trouble finding course recommendations right now. Could you be more specific about what type of courses you're looking for?"

        memory.chat_memory.add_ai_message(recommendation)
        save_chat_log(user_id, memory.chat_memory.messages)

        if "no relevant courses found" in recommendation.lower():
            return jsonify({
                "response": "I couldn't find specific courses matching your criteria. Would you like to try adjusting your grade level or exploring different subject areas?"
            })

        return jsonify({"response": recommendation})

    if not user_state.get_interests() or user_state.interest_turns < MIN_COUNSELOR_TURNS:
        user_state.interest_turns += 1
        interest_chain = build_interest_conversation_chain()
        
        try:
            bot_response = interest_chain.invoke({
                "grade": user_state.grade,
                "chat_history": chat_history_str,
                "user_input": message
            })
        except Exception as e:
            print(f"Error in interest conversation chain: {e}")
            bot_response = "I'd love to learn more about your interests. What subjects or activities do you enjoy?"

        updated_chat_history = chat_history_str + f"Student: {message}\nBot: {bot_response}\n"
        
        try:
            extract_chain = build_extract_interest_chain()
            extracted_interests = extract_chain.invoke({"chat_history": updated_chat_history})

            if extracted_interests and extracted_interests.lower() != "no clear interests yet.":
                user_state.add_interest(extracted_interests)
                print(f"Extracted interests: {extracted_interests}")
        except Exception as e:
            print(f"Error extracting interests: {e}")

        bot_response = add_counselor_prompt(bot_response, user_state)
        memory.chat_memory.add_ai_message(bot_response)
        save_chat_log(user_id, memory.chat_memory.messages)
        return jsonify({"response": bot_response})

    interest_chain = build_interest_conversation_chain()
    
    try:
        bot_response = interest_chain.invoke({
            "grade": user_state.grade,
            "chat_history": chat_history_str,
            "user_input": message
        })
    except Exception as e:
        print(f"Error in interest conversation: {e}")
        bot_response = "That's interesting! Tell me more about what you enjoy or what you're curious about."

    updated_chat_history = chat_history_str + f"Student: {message}\nBot: {bot_response}\n"
    
    try:
        extract_chain = build_extract_interest_chain()
        extracted_interests = extract_chain.invoke({"chat_history": updated_chat_history})
        
        if extracted_interests and extracted_interests.lower() != "no clear interests yet.":
            user_state.add_interest(extracted_interests)
            print(f"Additional interests extracted: {extracted_interests}")
    except Exception as e:
        print(f"Error extracting additional interests: {e}")

    bot_response = add_counselor_prompt(bot_response, user_state)
    memory.chat_memory.add_ai_message(bot_response)
    save_chat_log(user_id, memory.chat_memory.messages)
    return jsonify({"response": bot_response})

def save_chat_log(user_id, messages, path="chat_logs"):
    try:
        os.makedirs(path, exist_ok=True)
        simple_messages = []
        
        for msg in messages:
            simple_messages.append({
                "role": "user" if msg.type == "human" else "bot",
                "content": msg.content,
                "timestamp": msg.additional_kwargs.get("timestamp") if hasattr(msg, 'additional_kwargs') else None
            })
        
        with open(os.path.join(path, f"{user_id}_chat.json"), "w", encoding="utf-8") as f:
            json.dump(simple_messages, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error saving chat log: {e}")

@app.route("/get_chat_history", methods=["GET"])
def get_chat_history():
    user_id = get_user_id()
    memory = get_user_memory(user_id)
    
    messages = []
    for msg in memory.chat_memory.messages:
        messages.append({
            "role": "user" if msg.type == "human" else "bot",
            "content": msg.content
        })
    
    return jsonify({
        "messages": messages,
        "total_messages": len(messages),
        "max_messages": MAX_MEMORY_MESSAGES
    })

@app.route("/clear_history", methods=["POST"])
def clear_history():
    user_id = get_user_id()
    if user_id in user_memories:
        user_memories[user_id].clear()
    if user_id in user_states:
        user_states[user_id] = ConversationState()
    
    return jsonify({"message": "Chat history cleared successfully"})

@app.route("/get_user_info", methods=["GET"])
def get_user_info():
    user_id = get_user_id()
    user_state = get_user_state(user_id)
    memory = get_user_memory(user_id)
    
    return jsonify({
        "user_id": user_id,
        "grade": user_state.grade,
        "interests": user_state.get_interests(),
        "interest_turns": user_state.interest_turns,
        "message_count": len(memory.chat_memory.messages),
        "has_offered_recommendation": getattr(user_state, 'has_offered_recommendation', False)
    })

if __name__ == "__main__":
    app.run(debug=True)