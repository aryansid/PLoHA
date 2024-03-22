import fitz
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json 

client = OpenAI()

PROMPT = """
Given a segment from a medical textbook, identify the question, its corresponding answer, and the rationale. Ensure the answer and rationale are relevant to the question. If they don't match, respond with only 'Ignore'. Otherwise, format your response as 'Question: [found question], Answer: [found answer], Rationale: [found rationale]'. DO NOT RESPOND WITH ANYTHING ELSE. 
"""

def gpt(system, user, model="gpt-3.5-turbo-0125"):
    messages = [
        {
            "role": "system", 
            "content": system
        },
        {
            "role": "user", 
            "content": user
        }
    ]
    
    try: 
        response = client.chat.completions.create(
        model=model,
        messages=messages, 
        temperature=0.5,
        )
        
        gpt_response = response.choices[0].message.content.strip()

        return gpt_response
    except Exception as e: 
        print(f"Error in OpenAI API call: {e}")
        return None

page_ranges = [(68, 79), (109, 113), (156, 163), (194, 209), (246, 255), (291, 303), (331, 343), (365, 377)]
pages_to_read = []

for start, end in page_ranges:
    pages_to_read.extend(range(start-1, end))
    
textbook_data = [] 

textbook_path = "C:/Users/aryaa/Downloads/Kaplan Medical Prep Book.pdf"
with fitz.open(textbook_path) as doc:
    for page_num in pages_to_read:
        if page_num < len(doc):
            page = doc.load_page(page_num) 
            text = page.get_text() 
            textbook_data.append(text)  

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([textbook_data])

json_objects = []
for text in texts:  
  output = gpt(PROMPT, text)
  
  if output.strip().lower() != "ignore": 
    parts = output.split("Answer:")
    question = parts[0].replace("Question:", "").strip()
    answer, rationale = parts[1].split("Rationale:")
    answer = answer.strip()
    rationale = rationale.strip()
    
    json_obj = {"Question": question, "Answer": answer, "Rationale": rationale}
    json_objects.append(json_obj)
    
results_file_path = "textbook_extraction.json"
with open(results_file_path, 'w') as file:
    json.dump(json_objects, file, indent=4)


