import google.generativeai as genai

genai.configure(api_key="AIzaSyD5oIp0iZ_P5e8ZHZhndqJF5SnjzqsjZoQ") # Replace YOUR_API_KEY

models = genai.list_models()
for model in models:
    print(model.name)


