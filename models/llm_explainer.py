import google.generativeai as genai

genai.configure(api_key="AIzaSyAQJA1-gq00Rj5IzOelaETaUij5zOrUK4w")

model = genai.GenerativeModel("gemini-2.5-flash")

def generate_llm_explanation(news, sentiment, impact, entities):

    prompt = f"""
You are an expert AI news analyst.

News: {news}

Sentiment: {sentiment}
Impact: {impact}
Entities: {entities}

Explain briefly why this impact level makes sense.
Then give 3 realistic outcomes.

Format:

Explanation:
...

Outcomes:
- ...
- ...
- ...
"""

    try:

        response = model.generate_content(prompt)

        return response.text.strip()

    except Exception as e:

        print("DEBUG:", e)

        return "Explanation unavailable"