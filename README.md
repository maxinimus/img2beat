# Image to Beat

This is a Python script that allows you to turn your image into a background beat using AI. This is done by converting the image to text, then using that text to generate a description of a beat that would fit the image. That beat is then created using an AI model and output as a sound file.

## Requirements

- Python 3.6+
- audiocraft
- Langchain
- OpenAI API key
- poe

To install the required packages, run:

```python
pip install audiocraft langchain poe
```

## Usage

To use the script, run:

```python
streamlit run main.py
```

or

```python
streamlit run main.py poe
```

or

```python
streamlit run main.py official
```

When using `streamlit run main.py` or `streamlit run main.py poe`, the poe version will be used (reverse engineered), when using `streamlit run main.py official`, the official chat gpt api will be used, which will cost money. To run the official api, you need to include `OPENAI_API_KEY` in the .env file, and when running the poe version, you need to include `TOKEN` in the .env file, which can be found from the instructions [here](https://github.com/ading2210/poe-api).

Both versions do the same thing, they use 3 AI models to convert an image into image caption (description), to beat description, to sound.

## Credits

This script uses the following:

- audiocraft: https://github.com/facebookresearch/audiocraft
- Langchain: https://github.com/hwchase17/langchain
- poe: https://github.com/ading2210/poe-api
- blip-image-captioning-base: https://huggingface.co/Salesforce/blip-image-captioning-base
