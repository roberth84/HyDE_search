import os
import json
import openai


class LLM_Model():
    
    def __init__(self):
        openai.api_key_path = '/home/jupyter/openai_key.txt'
        self.cache_name = 'llm_model_cache.json'
        self.cache = {}
        
        # retrieve from cache if it exists
        #if os.path.isfile(self.cache_name):
        #    with open(self.cache_name, 'r') as f:
        #        self.cache = json.load(f)

    def get_response(self, question):
        if question not in self.cache.keys():

            
            prompt = f"""I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\n\nQ: {question}\nA:
        """
            response_text = ''
            while response_text is '':
                response = openai.Completion.create(
                  model="text-davinci-003",
                  prompt=prompt,
                  temperature=0.7,
                  max_tokens=200,
                  top_p=1,
                  frequency_penalty=0.0,
                  presence_penalty=0.0,
                  stop=["\n"]
                )
                response_text = response['choices'][0]['text'].strip()
            self.cache[question] = response['choices'][0]['text'].strip()
            #with open(self.cache_name, 'w') as f: 
            #    json.dump(self.cache, f)
        else:
            print("question already in cache")
            
        return self.cache[question]