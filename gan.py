from openai import OpenAI
import json 
import random

client = OpenAI()

generator_system = """
You are a specialist tasked with creating medical rationales for specific questions and answers. Your rationales should be medically accurate and reflect the nuance expected of a healthcare professional. Follow these guidelines to craft your response:
Guidelines: 

"""

discriminator_system = """
You are an expert at discerning the authenticity of medical rationales. Given a medical question and answer along with two potential rationales, your task is to determine which one was produced by a medical professional. Adhere to the following guidelines to accurately identify the correct rationale:
Guidelines:  

"""

generator_messages = [
        {
            "role": "system", 
            "content": generator_system
        },
    ]


discriminator_messages = [
        {
            "role": "system", 
            "content": discriminator_system
        },
    ]

def gpt(messages, model="gpt-3.5-turbo-0125"):
    messages = messages
    
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

with open("textbook_extraction.json", "r") as file: 
  data = json.load(file)
  
  for entry in data: 
    question = entry["Question"]
    answer = entry["Answer"]
    original_rationale = entry["Rationale"]
    
    generator_user = f"Question: {question}, Answer: {answer}"
    generator_messages.append({"role": "user", "content": generator_user})
    generated_rationale = gpt(generator_messages)
    generator_messages.append({"role": "assistant", "content": generated_rationale})
    
    rationales = ((original_rationale, 1), (generated_rationale, 2))
    random.shuffle(rationales)
    
    rationales = [f"Rationale {i}: {rationale}" for i, (rationale, _) in enumerate(rationales, 1)]
    discriminator_user = f"Question: {question}, Answer: {answer}, " + ", ".join(rationales) + " YOU MUST ONLY OUTPUT THE INDEX OF THE CORRECT RATIONALE. NO OTHER TEXT!"
    discriminator_messages.append({"role": "user", "content": discriminator_user})
    discriminator_choice = gpt(discriminator_messages)
    discriminator_messages.append({"role": "assistant", "content": discriminator_choice})
    
    predicted_index = rationales[int(discriminator_choice)-1][1]
    # Discriminator correct
    if predicted_index == 1: 
      discriminator_user_correct = f"Question: {question}, Answer: {answer}, " + ", ".join(rationales) + "Based on the information provided, you accurately identified rationale {discriminator_choice} as the one authored by a medical professional. Please explain clearly and concisely how you distinguished it as such."
      discriminator_messages.append({"role": "user", "content": discriminator_user_correct})
      discriminator_explaination = gpt(discriminator_messages)
      discriminator_messages.append({"role": "assistant", "content": discriminator_explaination})
      
      guidelines_index = generator_system.find("Guidelines:")
      pre_guidelines = generator_system[:guidelines_index]
      post_guidelines = generator_system[guidelines_index:]
      
      if post_guidelines.endswith("\n\n"):
          new_guidelines_section = post_guidelines + discriminator_explaination + "\n"
      elif post_guidelines.endswith("\n"):
          new_guidelines_section = post_guidelines + "\n" + discriminator_explaination + "\n"
      else:
          new_guidelines_section = post_guidelines + "\n\n" + discriminator_explaination + "\n"
        
      generator_system = pre_guidelines + new_guidelines_section
    else: 
      discriminator_user_wrong = f"Question: {question}, Answer: {answer}, " + ", ".join(rationales) + "With the information provided, you mistakenly identified rationale {discriminator_choice} as being authored by a medical professional. Please analyze this error and describe, in clear and concise terms, how you plan to avoid such a mistake in the future."
      discriminator_messages.append({"role": "user", "content": discriminator_user_wrong})
      discriminator_explaination = gpt(discriminator_messages)
      discriminator_messages.append({"role": "assistant", "content": discriminator_explaination})
      
      guidelines_index = discriminator_system.find("Guidelines:")
      pre_guidelines = discriminator_system[:guidelines_index]
      post_guidelines = discriminator_system[guidelines_index:]
      
      if post_guidelines.endswith("\n\n"):
          new_guidelines_section = post_guidelines + discriminator_explaination + "\n"
      elif post_guidelines.endswith("\n"):
          new_guidelines_section = post_guidelines + "\n" + discriminator_explaination + "\n"
      else:
          new_guidelines_section = post_guidelines + "\n\n" + discriminator_explaination + "\n"
        
      discriminator_system = pre_guidelines + new_guidelines_section
      
guidelines_text = generator_system.split("Guidelines:")[1]  
with open("few_shot_prompt.txt", "w") as file:
  file.write(guidelines_text)
