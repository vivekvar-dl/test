# LangChain with Gemini Demo App

A Streamlit application demonstrating various LangChain capabilities using Google's Gemini models.

## Features

- **Simple Chain**: Generate short stories based on a topic using Gemini
- **Sequential Chain**: Create stories with titles using multiple chained Gemini models
- **Conversation Memory**: Chat with Gemini AI that remembers context
- **Text Analysis**: Analyze text and extract key information 
- **Web Q&A**: Enter a URL and ask questions about the webpage content

## Requirements

- Python 3.7+
- Google AI API key

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/langchain-gemini-demo.git
cd langchain-gemini-demo
```

2. Create a virtual environment and activate it:
```
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Set up your Google API key as an environment variable (optional):
```
# Linux/Mac
export GOOGLE_API_KEY=your-api-key

# Windows
set GOOGLE_API_KEY=your-api-key
```

Alternatively, you can enter your API key directly in the app's sidebar.

## Running the App

Start the Streamlit app:
```
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

## Usage

1. Enter your Google API key in the sidebar if you haven't set it as an environment variable
2. Navigate between the different demo tabs to explore LangChain capabilities with Gemini
3. Follow the instructions in each tab to interact with the demos

## Technologies Used

- [LangChain](https://github.com/hwchase17/langchain): Framework for building applications with language models
- [Google Gemini](https://ai.google.dev/): Google's generative AI models
- [Streamlit](https://streamlit.io/): Web app framework for data science and machine learning 