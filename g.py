import google.generativeai as genai

genai.configure(api_key="Replace YOUR_API_KEY") # Replace with your actual API key

models = genai.list_models()
for model in models:
    print(model.name)


