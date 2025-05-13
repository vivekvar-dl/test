import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="LangChain Gemini Demo",
    page_icon="🦜",
    layout="wide",
)

# Sidebar with API key input
with st.sidebar:
    st.title("🦜️ LangChain Gemini Demo")
    
    google_api_key = st.text_input("Google API Key", type="password")
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    
    st.divider()
    st.markdown("# Demo Features")
    st.markdown("- Simple Chain")
    st.markdown("- Sequential Chain")
    st.markdown("- Conversation Memory")
    st.markdown("- Text Analysis")
    st.markdown("- Web Q&A (Text Only)")
    
    st.divider()
    st.caption("Created with LangChain and Google Gemini")

# Main content
st.title("🦜️🔗 LangChain with Gemini")
st.markdown("""
This app demonstrates various capabilities of LangChain with Google's Gemini models.
Choose a demo from the tabs below.
""")

# Check if API key is provided
if not google_api_key:
    st.warning("Please enter your Google API key in the sidebar to use the demos.")
    st.stop()

# Create tabs for different demos
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Simple Chain", 
    "Sequential Chain", 
    "Conversation Memory", 
    "Text Analysis", 
    "Web Q&A"
])

# Tab 1: Simple Chain
with tab1:
    st.header("Simple LLM Chain")
    st.markdown("""
    This demo shows a simple chain that takes a topic and generates a short story about it.
    """)
    
    topic = st.text_input("Enter a topic for a short story")
    
    if topic:
        # Create prompt template
        template = """Write a short story about {topic} in about 100 words."""
        prompt_template = PromptTemplate(input_variables=["topic"], template=template)
        
        # Create LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        
        # Create chain
        story_chain = LLMChain(llm=llm, prompt=prompt_template)
        
        # Run chain
        with st.spinner("Generating story..."):
            result = story_chain.invoke({"topic": topic})
            story = result.get("text", "")
        
        st.subheader("Generated Story:")
        st.write(story)
        
        # Explain the chain
        with st.expander("How does this work?"):
            st.markdown("""
            1. We created a prompt template with a variable for the topic
            2. We initialized a Gemini 2.0 Flash model with a temperature of 0.7 (more creative)
            3. We created a LLMChain that combines the prompt and the LLM
            4. When run, the chain:
               - Takes the input `topic`
               - Fills the prompt template
               - Sends the prompt to Gemini
               - Returns Gemini's response
            """)

# Tab 2: Sequential Chain
with tab2:
    st.header("Sequential Chain")
    st.markdown("""
    This demo shows a sequential chain that generates a short story and then creates a title for it.
    """)
    
    seq_topic = st.text_input("Enter a topic for the sequential chain")
    
    if seq_topic:
        # Create LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        
        # First, generate a story directly
        story_prompt = PromptTemplate.from_template("Write a very short story about {topic} in about 100 words.")
        
        with st.spinner("Generating story..."):
            story_chain = LLMChain(llm=llm, prompt=story_prompt)
            story_result = story_chain.invoke({"topic": seq_topic})
            story = story_result.get("text", "")
        
        # Then, generate a title
        title_prompt = PromptTemplate.from_template(
            """Create a catchy title for the following story:
            {story}
            
            Title:"""
        )
        
        with st.spinner("Generating title..."):
            title_chain = LLMChain(llm=llm, prompt=title_prompt)
            title_result = title_chain.invoke({"story": story})
            title = title_result.get("text", "")
        
        st.subheader(title)
        st.write(story)
        
        # Explain the chain
        with st.expander("How does this work?"):
            st.markdown("""
            1. We created two separate LLMChains with Gemini 2.0 Flash:
               - A story chain that takes a topic and generates a story
               - A title chain that takes a story and generates a title
            2. We run the chains in sequence manually
            3. The output of the first chain (story) becomes the input to the second chain
            4. The final result includes both the story and the title
            """)

# Tab 3: Conversation Memory
with tab3:
    st.header("Conversation with Memory")
    st.markdown("""
    This demo shows how LangChain can maintain conversation memory to have contextual conversations.
    """)
    
    # Initialize session state for memory
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()
    
    # Get user input
    user_input = st.text_input("You:", key="memory_input")
    
    # Display conversation history
    if st.session_state.memory.chat_memory.messages:
        st.subheader("Conversation History:")
        history = st.session_state.memory.load_memory_variables({})
        st.write(history["history"])
    
    # Process user input
    if user_input:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        conversation = ConversationChain(
            llm=llm, 
            memory=st.session_state.memory,
            verbose=True
        )
        
        with st.spinner("Thinking..."):
            response = conversation.predict(input=user_input)
        
        st.subheader("AI Assistant:")
        st.write(response)
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.memory = ConversationBufferMemory()
        st.success("Conversation cleared!")
        
    # Explain the chain
    with st.expander("How does this work?"):
        st.markdown("""
        1. We use ConversationBufferMemory to store the history of the conversation
        2. For each user message, we:
           - Add it to the conversation memory
           - Create a prompt that includes the conversation history
           - Send the prompt to the Gemini model
           - Add the model's response to the memory
        3. This allows the AI to maintain context across multiple interactions
        """)

# Tab 4: Text Analysis
with tab4:
    st.header("Text Analysis")
    st.markdown("""
    This demo shows how LangChain can analyze text input and extract information.
    """)
    
    # Text area for input
    text_input = st.text_area("Enter text to analyze:", height=200)
    
    if text_input:
        # Create template for analysis
        template = """
        Analyze the following text and provide:
        1. A brief summary (3-4 sentences)
        2. Key themes and topics
        3. Tone analysis (formal, informal, etc.)
        4. Five important keywords
        
        TEXT: {text}
        
        ANALYSIS:
        """
        prompt_template = PromptTemplate(input_variables=["text"], template=template)
        
        # Create LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        
        # Create chain
        analysis_chain = LLMChain(llm=llm, prompt=prompt_template)
        
        # Run chain
        with st.spinner("Analyzing text..."):
            result = analysis_chain.invoke({"text": text_input})
            analysis = result.get("text", "")
        
        st.subheader("Analysis Results:")
        st.write(analysis)
        
        # Explain the chain
        with st.expander("How does this work?"):
            st.markdown("""
            1. We take user-provided text as input
            2. We create a prompt template that asks for specific analysis points
            3. We use Gemini 2.0 Flash with temperature=0 for more factual analysis
            4. The LLM processes the text and returns structured analysis
            5. This demonstrates how LangChain can be used for text analysis tasks
            """)

# Tab 5: Web Q&A
with tab5:
    st.header("Web Question Answering (Text Only)")
    st.markdown("""
    This demo shows how LangChain can answer questions about web content using Gemini.
    """)
    
    # URL input
    url = st.text_input("Enter a URL to a webpage:")
    
    if url:
        try:
            with st.spinner("Fetching webpage content..."):
                # Fetch webpage content
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Extract text (basic extraction - could be improved)
                text = soup.get_text()
                # Clean up text a bit
                text = " ".join(text.split())
                
                # Show a preview of the extracted text
                with st.expander("Preview of extracted text"):
                    st.text(text[:1000] + "..." if len(text) > 1000 else text)
            
            st.success(f"Webpage content fetched! ({len(text)} characters)")
            
            # Question input
            question = st.text_input("Ask a question about the webpage:")
            
            if question:
                # Create template for Q&A
                template = """
                Use the following text from a webpage to answer the question.
                
                WEBPAGE CONTENT: {text}
                
                QUESTION: {question}
                
                ANSWER:
                """
                prompt_template = PromptTemplate(
                    input_variables=["text", "question"], 
                    template=template
                )
                
                # Create LLM and chain
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
                qa_chain = LLMChain(llm=llm, prompt=prompt_template)
                
                # Run chain
                with st.spinner("Generating answer..."):
                    result = qa_chain.invoke({
                        "text": text[:25000],  # Limiting text size
                        "question": question
                    })
                    answer = result.get("text", "")
                
                st.subheader("Answer:")
                st.write(answer)
                
        except Exception as e:
            st.error(f"Error processing webpage: {e}")
    
    # Explain the chain
    with st.expander("How does this work?"):
        st.markdown("""
        1. The webpage content is fetched and extracted using BeautifulSoup
        2. When a question is asked, the system:
           - Creates a prompt with the webpage content and the question
           - Sends this prompt to Gemini with a lower temperature for factual responses
           - Returns Gemini's answer based on the webpage content
        3. Note: This is a simplified version that processes text directly rather than using embeddings and vector search
        """)

# Footer
st.divider()
st.caption("Enter your Google API key in the sidebar to use the demos. No data is stored or logged.") 