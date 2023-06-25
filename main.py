from dotenv import load_dotenv
from transformers import pipeline
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
import streamlit as st
import os
import poe
import logging
import sys

# load env variables
load_dotenv()

# set local variables
beat_duration = 8 # seconds
max_tokens = 100 # max tokens for the llm model
llm_model = 'gpt-3.5-turbo' # gpt model name

# print colored text
def print_colored_text(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m")

# image to text
def img2text(url):
    # use huggingface pipeline to convert image to text
    image_to_text = pipeline("image-to-text", model='Salesforce/blip-image-captioning-base')

    text = image_to_text(url)[0]['generated_text']
    print_colored_text("Image to text: " + text, 32)
    return text

# llm (gpt-3.5-turbo in this case) to create the beat text from photo description
def description2beat(scenario):
    # template for the prompt
    template = """
    You are a music producer. You have access to an AI that creates a beat based on a 
    description. I will give you a description of an image, and you will write me a 
    description of a beat that could act as background music, something fitting. There 
    should be no lyrics, just a description of the beat in less than 25 words.

    Image description: {scenario}
    Beat description:
    """

    # create the prompt with the input variables and the template
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    beat_llm = LLMChain(llm=ChatOpenAI(model_name=llm_model), prompt=prompt, verbose=True)
    
    # generate the beat text
    beat_text = beat_llm.predict(scenario=scenario, max_tokens=max_tokens)

    print_colored_text("Beat text generated: " + beat_text, 32)
    return beat_text

# the free version
def description2beatPoe(scenario):
    token = os.getenv("TOKEN")
    poe.logger.setLevel(logging.INFO)
    client = poe.Client(token)

    message = """ 
    You are a music producer. You have access to an AI that creates a beat based on a 
    description. I will give you a description of an image, and you will write me a 
    description of a beat that could act as background music, something fitting. There 
    should be no lyrics, just a description of the beat in less than 25 words.

    Image description: {}

    Beat description: 
    """
    message = message.format(scenario)
    
    for chunk in client.send_message("chinchilla", message):
        pass
    beat_text = chunk["text"]

    print_colored_text("Beat text generated: " + beat_text, 32)
    return beat_text

# text to beat
def text2beat(text): 
    # delete the previous sound.wav file
    try:
        os.remove('sound.wav')
    except:
        pass

    # the model can be choose from different sizes, 
    # but I used small because it's the fastest one to download
    model = MusicGen.get_pretrained('small')
    model.set_generation_params(duration=beat_duration)

    descriptions = [text]

    wav = model.generate(descriptions)  

    # Will save under sound.wav, with loudness normalization at -14 db LUFS.
    audio_write("sound", wav[0].cpu(), model.sample_rate, strategy="loudness")

def main():
    if len(sys.argv) == 1:
        print_colored_text("Using poe version", 32)
    elif sys.argv[1] == 'official':
        print_colored_text("Using official version", 32)
    elif sys.argv[1] == 'poe':
        print_colored_text("Using poe version", 32)
    # set page title and favicon
    st.set_page_config(page_title='image to beat', page_icon='🎵')

    # set page layout
    st.header('Turn your image into a background beat using AI')
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Delete the previously downloaded image if it exists
        if os.path.exists("downloaded.jpg"):
            os.remove("downloaded.jpg")

        bytes_data = uploaded_file.getvalue()
        # save the image to a file named 'downloaded.jpg'
        with open("downloaded.jpg", "wb") as file:
            file.write(bytes_data)

        # show the image
        st.image(uploaded_file, caption='Uploade    d Image.', use_column_width=True)

        # convert the image to text
        scenario = img2text("downloaded.jpg")

        # convert the text to beat description
        # if there's no argument, use the poe version
        if len(sys.argv) == 1:
            beat_description = description2beatPoe(scenario)
        elif sys.argv[1] == 'official':
            beat_description = description2beat(scenario)
        elif sys.argv[1] == 'poe':
            beat_description = description2beatPoe(scenario)

        # convert the beat description to beat
        text2beat(beat_description)

        # show the beat description
        with st.expander("scenario"):
            st.write(scenario)

        with st.expander("beat description"):
            st.write(beat_description)

        # show the beat
        st.audio('sound.wav')

if __name__ == "__main__":
    main()
