import base64
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

def analyze_stool_images(image_paths):
    """
    Takes a list of image paths and returns AI analysis results for each.
    """
    # Convert images to base64 for API
    encoded_images = []
    for idx, path in enumerate(image_paths):
        with open(path, "rb") as f:
            encoded_images.append({
                "image_id": f"img_{idx+1}",
                "data": base64.b64encode(f.read()).decode()
            })

    # Build multi-image analysis prompt
    system_prompt = """
    You are a medical image assistant trained to classify human stool using the Bristol Stool Chart (types 1–7).
    Analyze EACH of the provided images and return an array of JSON objects, one per image, in this exact format:
    [
      {
        "image_id": "[unique identifier provided or index]",
        "bristol_type": [1-7 or null if unclear], 
        "color": "[basic color description or 'unknown']",
        "consistency": "[brief texture/shape description or 'unclear']",
        "notable_findings": "[mention visible abnormalities like blood, mucus, unusual color, or 'none']",
        "confidence": "[integer percentage confidence in classification]",
        "action_required": "[boolean: true if confidence <70 or bristol_type is null, otherwise false]"
      }
    ]
    Guidelines:
    - Use the Bristol Stool Chart as the primary reference.
    - Keep descriptions short and plain-language.
    - If you cannot confidently classify (confidence <70%), set bristol_type to null and action_required to true.
    - Do NOT give medical diagnoses or advice.
    """

    # Send to OpenAI
    user_prompt = "Analyze these stool images:\n"
    for img in encoded_images:
        user_prompt += f"Image ID: {img['image_id']}, Data: data:image/jpeg;base64,{img['data']}\n"

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # Parse response
    result = response.choices[0].message["content"]
    return result  # JSON string — store/parse as needed

# Example usage
if __name__ == "__main__":
    results = analyze_stool_images(["stool1.jpg", "stool2.jpg"])
    print(results)
