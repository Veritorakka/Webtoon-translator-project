import ollama
import os

def translate_and_provide_context(text_to_translate, source_language):
    # Hae käytettävä malli ympäristömuuttujasta
    model = os.getenv('OLLAMA_MODEL')

    # Tarkistetaan, onko ympäristömuuttuja asetettu
    if not model:
        raise ValueError("LLM model is not set! Please set the OLLAMA_MODEL environment variable when starting the Docker container.")

    # Muokataan promptia ottamaan huomioon lähdekieli
    response = ollama.chat(model=model, messages=[
      {
        'role': 'user',
        'content': f'''Translate the following text from {source_language} into English and provide small amount of context for the translation. The response should be structured as follows:
        - "Translation": <Provide the translation here>.
        - "Context": <Provide a brief context or explanation here>.
        
        Here is the text to translate:
        "{text_to_translate}"'''
      },
    ])
    
    # Haetaan vastaus
    full_response = response['message']['content']
    
    # Oletusarvoiksi tyhjät
    translation = ""
    context = ""

    # Tarkistetaan onko 'Translation' ja 'Context' osat mukana vastauksessa
    if 'Translation' in full_response and 'Context' in full_response:
        translation = full_response.split('Translation')[1].split('Context')[0].strip(": \n")
        context = full_response.split('Context')[1].strip(": \n")
    else:
        print("Vastaus ei ole odotetussa muodossa.")
        print(full_response)

    # Palautetaan erillään käännös ja konteksti
    return translation, context


