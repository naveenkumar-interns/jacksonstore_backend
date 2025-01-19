from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
from dotenv import load_dotenv
from datetime import datetime
from utils import load_vectorstore

app = Flask(__name__)
CORS(app)
# Load environment variables
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

memory = ConversationBufferWindowMemory(return_messages=True, k=2)

def convert_to_json(data):
    result = []
    for item in data:
        # Filter out unnecessary keys from metadata
        filtered_metadata = {
            key: value for key, value in item.metadata.items()
            if key not in ["seq_num", "source", "handle"]
        }
        result.append(filtered_metadata)
    return result


def get_product_search(query):
    result = vectorstore.similarity_search(query=query,k=3)
    return convert_to_json(result)

def get_response(input_text):
    try:
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Jackson Hardware Store's knowledgeable AI assistant. Your role is to:
        1. Help customers easily navigate the product catalog and find the right hardware, tools, and equipment from our extensive inventory.
        2. Provide accurate information about product availability, pricing, and specifications.
        3. Recommend related or complementary products when appropriate.
        4. Maintain a friendly, polite, and professional tone that reflects our commitment to excellent customer service.
        5. Encourage further interaction to assist customers thoroughly.
        6. note important: Avoid using technical formatting like new line symbols, markdown symbols *, _, etc., or bullet points.
        7. Write in clear, plain text that feels conversational and easy to read.

        note: Provide friendly and professional responses that resemble natural human conversation.
        Deliver the response here in plain text without any formatting.

        Remember to maintain context from the conversation history: {history}
        """,
    ),
    (
        "human",
        "{input}"
    ),
])


        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
        )

        response = chain.invoke({"input": input_text})
        return response['text']
        
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise


def get_response_product_search(input_text,related_products):
    try:
        prompt = ChatPromptTemplate.from_messages([
        (
        "system",
        """You are Jackson Hardware Store's AI assistant. Your job is to:
        1. Help customers find the right tools, hardware, or equipment.
        2. Suggest relevant products based on customer needs and related items.
        3. Share key product details like brand, features, use cases, and availability.
        4. Recommend alternatives if a product is unavailable.
        5. note important: Avoid using technical formatting like new line symbols, markdown symbols *, _, etc., or bullet points.
        6. Write in clear, plain text that feels conversational and easy to read.

        note: Provide friendly and professional responses that resemble natural human conversation.
        Deliver the response here in plain text without any formatting.
        chat history: {history}
        """,
    ),
    ("human", "{input}"),
])

        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
        )
        query = f"user query : {input_text} and related products based on user query:{str(related_products)}"
        response = chain.invoke({"input": query})
        return response['text']
        
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise 


@app.route('/open-vectorstore', methods=['POST'])
def createvectorstore():
    from create_store import open_vectorstore
    open_vectorstore()
    print("vectorstore loaded successfully")
    return jsonify({"message": "Vectorstore created successfully"})


@app.route('/load-vectorstore', methods=['POST'])
def loadvectorstore():
    global vectorstore
    vectorstore = load_vectorstore(vectorstore_path="shopify_langchain_testing_vectorstore",index_name="products")
    print("vectorstore loaded successfully")
    return jsonify({"message": "Vectorstore loaded successfully"})

# Store chat history
chat_history = []

@app.route('/chat', methods=['POST'])
def chat():
    try:
        message = request.json
        message.update({
            'timestamp': datetime.now().isoformat(),
            'id': len(chat_history)
        })
        chat_history.append(message)
        

        related_products_for_query = get_product_search(message['content']) 

        ai_response = get_response(input_text = message['content'])
        response = {
            'content': ai_response,
            'sender': 'bot',
            'timestamp': datetime.now().isoformat(),
            'id': len(chat_history),
            'related_products_for_query':related_products_for_query
        }

        chat_history.append(response)
        
        return jsonify(response)
    
    except Exception as e:
        error_response = {
            'content': "I apologize, but I encountered an error. Please try again.",
            'sender': 'bot',
            'error': True,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(error_response), 500
    
@app.route('/chat-product-search', methods=['POST'])
def chat_product_search():
    try:
        message = request.json
        message.update({
            'timestamp': datetime.now().isoformat(),
            'id': len(chat_history)
        })
        chat_history.append(message)
        

        related_products_for_query = get_product_search(message['content'])
        

        ai_response = get_response_product_search(input_text = message['content'], related_products = related_products_for_query)
        
        response = {
            'content': ai_response,
            'sender': 'bot',
            'timestamp': datetime.now().isoformat(),
            'id': len(chat_history),
            'related_products_for_query':related_products_for_query

        }
        chat_history.append(response)
        
        return jsonify(response)
    
    except Exception as e:
        error_response = {
            'content': "I apologize, but I encountered an error. Please try again.",
            'sender': 'bot',
            'error': True,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(error_response), 500
    

@app.route('/chat-history', methods=['GET'])
def get_chat_history():
    return jsonify(chat_history)

@app.route('/clear-chat', methods=['POST'])
def clear_chat():
    try:
        chat_history.clear()
        memory.clear()
        return jsonify({"message": "Chat history cleared successfully"})
    except Exception as e:
        return jsonify({"error": "Failed to clear chat history"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
