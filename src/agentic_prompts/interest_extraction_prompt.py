from langchain.prompts import ChatPromptTemplate

extract_interest_prompt = ChatPromptTemplate.from_template("""
You are an interest extraction system. Your task is to analyze student conversation data and identify clear academic or personal interests that can be used for educational course recommendations.

Here is the full conversation so far:

{chat_history}

Analysis Instructions:
1. Scan through all student messages in the conversation history
2. Identify interests that are explicitly mentioned or strongly implied through positive language
3. Focus on extractable interests that could map to educational courses or learning opportunities
4. Ignore casual mentions or negative statements about topics

Extract interests in these categories:
- Academic subjects (e.g., "mathematics", "biology", "history", "literature")
- Technical skills (e.g., "programming", "web development", "data analysis", "robotics")
- Creative pursuits (e.g., "music production", "digital art", "photography", "writing")
- Physical activities (e.g., "sports science", "fitness", "dance")
- Career interests (e.g., "medicine", "engineering", "business", "teaching")
- Hobbies with learning potential (e.g., "gaming", "cooking", "gardening")

Quality criteria for extraction:
- Only include interests mentioned with enthusiasm or positive sentiment
- Prioritize interests mentioned multiple times or elaborated upon
- Exclude topics mentioned only once in passing
- Exclude subjects mentioned negatively (e.g., "I hate math")

Output Format:
- Return a clean, comma-separated list of interests in lowercase
- Maximum 5-7 most relevant interests
- If no clear interests meet the criteria, respond with exactly: "No clear interests yet."

Example outputs:
- "programming, robotics, physics, game design"
- "creative writing, literature, psychology"
- "No clear interests yet."
""")