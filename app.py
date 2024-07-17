import os
from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from dotenv import load_dotenv
import openai
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure OpenAI with API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

client = OpenAI(api_key=api_key)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/learn', methods=['POST'])
def learn():
    data = request.get_json()
    concept = data['concept']
    profession = data['profession']
    complexity = int(data['complexity'])
    
    # Determine complexity level
    complexity_prompts = [
        "Explain {concept} to a {profession} in the simplest terms possible.",
        "Explain {concept} to a {profession} in simple terms.",
        "Explain {concept} to a {profession}.",
        "Explain {concept} to a {profession} with some technical details.",
        "Explain {concept} to a {profession} with detailed technical information."
    ]

    profession_prompt = " Also, let me know everything I should know about {concept} as a {profession}."
    # Append the additional string to each prompt
    complexity_prompts = [prompt + profession_prompt for prompt in complexity_prompts]
    
    prompt = complexity_prompts[complexity - 1].format(concept=concept, profession=profession)
    
    # Calculate the number of tokens in the prompt
    prompt_tokens = num_tokens_from_string(prompt, "cl100k_base")
    total_tokens = 4096  # Updated token limit for GPT-4o
    max_tokens = total_tokens - prompt_tokens
    
    def generate_response():
        response = client.chat.completions.create(
            model="gpt-4o",  # Use the latest GPT-4o model
            messages=[
                {"role": "system", "content": "You are good at explaining concepts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,  # Adjust temperature for creativity vs. determinism
            top_p=1,
            stream=True
        )
        for chunk in response:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content:
                yield chunk_content.encode('utf-8')

    return Response(stream_with_context(generate_response()), content_type='text/event-stream')

@app.route('/quiz', methods=['POST'])
def quiz():
    data = request.get_json()
    content = data.get('content', '')

    quiz_prompt = f"Based on the following content, generate an effective quiz question (not simple) in the first line, followed by four options for the quiz on the next four lines. Correct option must always be 4th option.\n\nContent:\n{content}\n\nQuiz Question:"

    def generate_quiz_response():
        response = client.chat.completions.create(
            model="gpt-4o",  # Use the latest GPT-4o model
            messages=[
                {"role": "system", "content": "You are an expert in generating quiz questions based on content."},
                {"role": "user", "content": quiz_prompt}
            ],
            max_tokens=200,
            temperature=0.7,  # Adjust temperature for creativity vs. determinism
            top_p=1,
            stream=True
        )
        for chunk in response:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content:
                yield chunk_content.encode('utf-8')

    full_response = ''
    for chunk in generate_quiz_response():
        full_response += chunk.decode('utf-8')

    quiz_data = full_response.strip().split('\n')
    question = quiz_data[0].strip()
    print("The question was: ", question)
    options = [opt.strip() for opt in quiz_data[2:6]]
    print("The options were: ", options)
    correct_answer = quiz_data[5].replace("Correct answer:", "").strip()
    print("The correct_answer was: ", correct_answer)

    quiz = {
        "question": question,
        "options": options,
        "correct_answer": correct_answer
    }

    return jsonify(quiz)

@app.route('/mindmap', methods=['POST'])
def mindmap():
    data = request.get_json()
    concept = data['concept']
    
    mindmap_prompt = f"For {concept}, please draw a visually appealing Graphviz graph to break down the concept into atomic level subtopics to facilitate easy, intuitive learning. Make each node clickable with a hyperlink that has the node's label as the value."

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use the latest GPT-4o model
            messages=[
                {"role": "system", "content": "You are an expert in creating mind maps using GraphViz."},
                {"role": "user", "content": mindmap_prompt}
            ],
            max_tokens=1000,
            temperature=0.7,
            top_p=1,
            stream=True
        )
        full_response = ''
        for chunk in response:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content:
                full_response += chunk_content

        mindmap = full_response.strip()
        return jsonify({"mindmap": mindmap})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
