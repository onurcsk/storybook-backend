from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoProcessor, TFBlipForConditionalGeneration
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
import os
from io import BytesIO
import hashlib
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings


app = FastAPI()

# Initialize Vertex AI
aiplatform.init(project=os.environ["GOOGLE_PROJECT_ID"], location=os.environ["GOOGLE_PROJECT_REGION"])

# Load the image captioning model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Load vector_db for prompt context (RAG)
# Initialize embeddings
embedding_function = VertexAIEmbeddings(project=os.environ["GOOGLE_PROJECT_ID"], model_name="textembedding-gecko@003")

# Load the vector database from the persisted directory with embeddings
persist_directory = "./raw_data/chroma_db"
vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

def create_context(genre, image_captions):
    # Setting up search query based on user provided genre/theme and image captions
    if genre:
        search_query = f"{genre}."
    if image_captions:
        search_query += " " + ", ".join(image_captions)
    # Define number of closest documents
    num_closest_docs = 3
    # Perform similarity search to retrieve relevant documents
    docs = vector_db.similarity_search(search_query, k=num_closest_docs)
    return docs


def generate_caption(image):
    inputs = processor(images=image, return_tensors="tf")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def hash_image(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    img_str = buffer.getvalue()
    return hashlib.md5(img_str).hexdigest()

def generate_story(genre, num_words, num_characters, reader_age, language, character_names, character_genders, image_captions):
    docs = create_context(genre, image_captions)
    text1 = "Write me a story."
    if genre:
        text1 = f"Write a story in the following theme or genre: {genre}"
    if reader_age:
        text1 += f" suitable for {reader_age}-year-olds"
    if language:
        text1 += f"in {language}."
    if num_words:
        text1 += f" with {num_words} words"
    if num_characters:
        text1 += f" and {num_characters} characters."
    else:
        text1 += "."

    if character_names or character_genders:
        characters_info = " The main characters are "
        if character_names and character_genders:
            characters_info += ", ".join([f"{name} ({gender})" if gender else name for name, gender in zip(character_names, character_genders)])
        else:
            characters_info += ", ".join(character_names if character_names else character_genders)
        text1 += characters_info + "."

    text1 += " The story should be engaging and didactic. It should have a clear introduction, development, and a clear ending."
    if image_captions:
        text1 += " The following captions should be integrated in the story to contribute to the story development: " + ", ".join(image_captions)

    if docs:
        text1 += f"Consider these three example stories as additional context to build the final story: {docs}."

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }
    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    # Generate story prompt
    model = GenerativeModel("gemini-1.5-flash-001")
    responses = model.generate_content([text1], generation_config=generation_config, safety_settings=safety_settings, stream=True)

    generated_story = ""
    for response in responses:
        generated_story += response.text

    return generated_story

@app.post("/generate_caption/")
async def generate_image_caption(file: UploadFile = File(...)):
    image = Image.open(file.file)
    caption = generate_caption(image)
    return JSONResponse(content={"caption": caption})

@app.post("/generate_story/")
async def generate_story_endpoint(
    genre: str = Form(None),
    num_words: int = Form(None),
    num_characters: int = Form(None),
    reader_age: int = Form(None),
    character_names: str = Form(None),
    character_genders: str = Form(None),
    image_captions: str = Form(None),
    language: str = Form(None)
):
    character_names_list = [name.strip() for name in character_names.split(",")] if character_names else []
    character_genders_list = [gender.strip() for gender in character_genders.split(",")] if character_genders else []

    image_captions = [caption.strip() for caption in image_captions.split(",")] if image_captions else []
    story = generate_story(genre, num_words, num_characters, reader_age, language, character_names_list, character_genders_list, image_captions)
    return JSONResponse(content={"story": story})

@app.get("/")
def hello():
    return {'greetings' : 'hello story lovers!'}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
