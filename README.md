# Storybook - Personalized AI Story Generator for Kids

Welcome to **Storybook**, a personalized AI story generation app designed to create unique stories based on your inputs and uploaded images. This application leverages advanced AI techniques to generate captivating stories for children, customized to their preferences and interests.

## Project Overview

Storybook is an innovative AI application that allows users to generate personalized stories by providing specific details such as genre, number of words, reader's age, language, and character details. The application integrates image captions from user-upoloaded photos to include those as elements in the story, making it more engaging and enjoyable. It is using `Salesforce/blip-image-captioning-large` model from Hugging Face. For the story generation we employ Retrieved Augmented Generation (RAG) together with Google's VertexAI generative model `gemini-1.5-flash-001` on a dataset consisting of around 900,000 stories especially meant for young children. This increases the accuracy and suitability for children of the generative model inferences.

## Benefits of the App

- **Personalized Stories**: Customize the genre, length, and language of the story to match the reader's preferences and age.
- **Character Customization**: Add characters with specific names and genders to make the story more relatable.
- **Turning your Real-Life into Stories**: Upload images to generate captions and incorporate them into the story, enriching the storytelling experience and making it super easy to use.
- **Downloadable PDFs**: Generate and download stories in PDF format for easy sharing and offline reading.

## How It Works

### User Inputs

1. **Story Details**: Enter the story genre, number of words, reader's age, and select the language.
2. **Character Details**: Add character names and select their genders.
3. **Image Upload**: Upload up to 5 images to generate captions that will be incorporated into the story.

### AI Story Generation Process

1. **Image Captioning**: Uploaded images are processed to generate descriptive captions using an API.
2. **Story Creation**: The provided inputs (genre, word count, age, language, character details, and captions) are sent to an API that generates a unique story.
3. **Retrieved Augmented Generation (RAG)**: The use of the RAG technique enhances the accuracy and reliability of the generative AI model with stories fetched from a dataset consisting of around 900,000 stories especially meant for Young Children (https://huggingface.co/datasets/ajibawa-2023/Children-Stories-Collection)
3. **PDF Generation**: The generated story is converted into a PDF document for easy downloading and sharing.

## Technical Overview

### AI models and techniques

#### Image Captioning

- **Model**: The image captioning is performed using the `Salesforce/blip-image-captioning-large` model.
- **Process**: Images are uploaded and passed to the FastAPI endpoint, where the model generates captions that describe the visual content. These captions are then incorporated into the story to enhance the narrative.

#### Retrieved Augmented Generation (RAG)

- **Embeddings**: Google Cloud's Vertex AI is used to create embeddings for a dataset of existing children stories, which allows for a similarity search based on the theme/genre and the image captions to retrieve relevant children stories as additional context.
- **Retrieval**: Chroma, from the `langchain_community.vectorstores` library, is utilized as the vector database to store and retrieve relevant text snippets based on the user inputs (genre, image captions).
- **Generative Model**: The story generation leverages Vertex AI Generative Models (`vertexai.preview.generative_models`) to create a coherent and contextually enriched story.
- **Process**:
  - **Retrieval**: Relevant text snippets and knowledge are retrieved from a large dataset using Vertex AI embeddings based on the user inputs.
  - **Augmentation**: The retrieved information is fed into the Vertex AI Generative Model to produce a coherent and contextually enriched story.
- **Advantages**: RAG allows for more accurate and contextually appropriate story generation by grounding the output in relevant data, ensuring that the generated stories are both creative and informative.

### Technology Stack for RAG

- **Database**: Chroma, a vector database, is used to store and retrieve relevant text snippets.
- **Generative Model**: Vertex AI Generative Models are used for generating the story content, taking into account the retrieved information to ensure coherence and relevance.
- **Integration**: The retrieval and generative components are seamlessly integrated using FastAPI, which handles the inputs from the frontend, processes them through the RAG pipeline, and returns the generated story.

### Technology Stack for the App

- **Frontend**: Built with Streamlit for an interactive and user-friendly interface.
- **Backend**: Powered by a FastAPI service hosted on a cloud platform.
- **Image Processing**: Utilizes the PIL (Pillow) library for image handling and hashing.
- **PDF Generation**: Uses `pdfkit` to convert HTML templates into PDF documents.
- **Session Management**: Maintained using Streamlit's session state.

### Session Management

- **Session State**: User inputs and generated data are stored in the session state to maintain continuity and allow for easy modification and re-generation of stories.
- **Story History**: The app maintains a history of the last five stories generated, allowing users to review and download their favorite stories.

## Installation and Usage

### Prerequisites

- Python 3.7+
- Streamlit
- PIL (Pillow)
- Requests
- PDFKit

## Conclusion

Storybook is a powerful tool for creating personalized and engaging stories for children. By combining user inputs with advanced AI techniques, it offers a unique and enjoyable storytelling experience. We hope you and your little ones enjoy the magical stories created with Storybook!

## Acknowledgements

This project was developed as part of the LeWagon Data Science & AI Final Project. Special thanks to the instructors and fellow students for their support and guidance.
**Authors: Sara Welter, Onurcan Sabri Kurt and Marius Tippkoetter**

---
