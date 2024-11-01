import os
from huggingface_hub import InferenceClient
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import chainlit as cl
from typing import Optional
from langchain.memory import ConversationBufferMemory
from literalai import LiteralClient

# Load environment variables
load_dotenv()

# Initialize Literal AI Client
literal_client = LiteralClient(api_key=os.getenv("LITERAL_API_KEY"))

# Download NLTK data
nltk.download('vader_lexicon')

# Configure Hugging Face API
client = InferenceClient(
    "microsoft/Phi-3-mini-4k-instruct",
    token=os.getenv("HF_API_KEY"),
)

# Advanced sentiment analysis function
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity

    if sentiment_scores['compound'] >= 0.1:
        overall = 'positive'
    elif sentiment_scores['compound'] <= -0.1:
        overall = 'negative'
    else:
        overall = 'neutral'

    intensity = abs(sentiment_scores['compound'])

    return {
        'overall': overall,
        'intensity': intensity,
        'subjectivity': subjectivity,
        'scores': sentiment_scores
    }
    
SYSTEM_PROMPT_GENERAL = """
You're Ashley, a supportive AI friend focused solely on mental health. Your mission is to uplift and motivate users, creating a safe space for them to express their feelings. 

**Tone & Approach:**
- **Friendly Introduction:** Introduce yourself casually as Ashley only during the first chat.
- **Empathy & Light Humor:** Be kind and supportive, using light humor to help elevate the user’s mood without being overly casual.
- **Personalized Support:** Tailor your responses based on the user's emotional state, providing relatable advice for those feeling down or stressed.

**Goals & Guidelines:**
1. **Be Genuine:** Speak like a friend who genuinely cares, using relatable language.
2. **Focus on Mental Health:** Stay strictly on topic, addressing feelings and emotional challenges without veering into personal details about celebrities or sports.
3. **Motivational Encouragement:** Use encouraging words to uplift users when they express sadness or frustration, reminding them of their strengths and potential.
4. **Self-Exploration:** Encourage users to reflect on their thoughts and feelings with thoughtful questions.
5. **Practical Well-Being Tips:** Offer brief, actionable advice related to mental health, including self-care practices, sleep, and nutrition when appropriate.
6. **Positive Outlook:** Acknowledge challenges but guide users towards constructive solutions and self-discovery.

**Response Style:**
- **Concise & Impactful:** Keep responses brief yet meaningful, focusing on delivering clear support.
- **Sentiment-Driven Language:** Adjust your tone based on the user’s emotional state, aiming to provide comfort and encouragement.
- **No Meta-Commentary:** Provide direct, actionable advice without commentary about the conversation itself.

**Objective:** Your goal is to make users feel heard, uplifted, and empowered, fostering an environment of support and positivity centered around their mental health journey.
"""


# Set Chainlit Starters
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Daily motivation boost",
            message="I'm feeling a bit low today. Can you share some uplifting words or remind me why small steps matter?",
            icon="/public/coffee-cup.png",
        ),
        cl.Starter(
            label="Relaxation ideas",
            message="Could you suggest some calming activities to help me unwind after a tough day?",
            icon="/public/meditation.png",
        ),
        cl.Starter(
            label="Ways to handle stress",
            message="I'm feeling overwhelmed. Can you offer some simple ways to manage stress?",
            icon="/public/sneakers.png",
        ),
        cl.Starter(
            label="Finding positivity",
            message="I need a fresh perspective. Can you help me find something positive in my day?",
            icon="/public/idol.png",
        )
    ]


# Define LangChain Prompt Template
prompt_template = PromptTemplate(
    input_variables=["system_prompt", "user_input", "sentiment"],
    template="{system_prompt}\nUser's emotional state: {sentiment}\nUser: {user_input}\nAshley:"
)

# A simple dictionary to store user passwords (replace with a secure database in production)
user_passwords = {}

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    try:
        # Try to get the user from Literal AI
        user_info = literal_client.api.get_user(identifier=username)
        
        # User exists in Literal AI, check if we have a password for them
        if username in user_passwords:
            if user_passwords[username] == password:
                return cl.User(identifier=username, metadata={"role": "user", "info": user_info})
            else:
                print(f"Invalid password for user: {username}")
                return None
        else:
            # User exists in Literal AI but not in our password store
            # We'll treat this as a new user registration
            user_passwords[username] = password
            print(f"Registered existing Literal AI user: {username}")
            return cl.User(identifier=username, metadata={"role": "user", "info": user_info})
    except Exception as e:
        # User doesn't exist in Literal AI, create new user
        try:
            new_user_info = literal_client.api.create_user(identifier=username)
            user_passwords[username] = password
            print(f"Created new user: {username}")
            return cl.User(identifier=username, metadata={"role": "user", "info": new_user_info})
        except Exception as e:
            print(f"Error creating new user: {e}")
            return None

@cl.on_message
async def main(message: cl.Message):
    # Analyze sentiment
    sentiment_info = analyze_sentiment(message.content)
    sentiment_description = f"Sentiment: {sentiment_info['overall']}, Intensity: {sentiment_info['intensity']:.2f}, Subjectivity: {sentiment_info['subjectivity']:.2f}"

    # Prepare the prompt content with the system prompt, sentiment, and user input
    prompt_content = prompt_template.format(
        system_prompt=SYSTEM_PROMPT_GENERAL,
        user_input=message.content,
        sentiment=sentiment_description
    )

    response = ""
    msg = cl.Message(content="")  # Prepare to send the response
    await msg.send()

    for chunk in client.chat_completion(
        messages=[{"role": "user", "content": prompt_content}],
        max_tokens=500,
        stream=True,
    ):
        token = chunk.choices[0].delta.content
        
        if "Ashley:" in token:
            token = token.split("Ashley:")[1].strip()
        
        if token:
            response += token
            await msg.stream_token(token)

    await msg.update()
    
    
@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    try:
        memory = ConversationBufferMemory(return_messages=True)
        
        # Retrieve chat history from LiteralAI
        user_id = cl.user_session.get_user().identifier
        user_chats = literal_client.api.get_chats(identifier=user_id)

        for chat in user_chats:
            if chat["type"] == "user_message":
                memory.chat_memory.add_user_message(chat["content"])
            else:
                memory.chat_memory.add_ai_message(chat["content"])

        cl.user_session.set("memory", memory)
    except Exception as e:
        print(f"Error during chat resume: {e}")
