import streamlit as st
from methods.big_mind_mapping import bmm
from methods.reverse_brainstorm import rb
from methods.role_storming import rs
from methods.scamper import sc
from methods.six_hats import sh
from methods.starburtsting import sb
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from typing import Any, List, Optional
from langchain_core.outputs import ChatGeneration, ChatResult

class LocalLLM(BaseChatModel):
    base_url: str
    api_key: str = "not-needed"
    
    def _generate(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[Any] = None, 
        **kwargs: Any
    ) -> ChatResult:
        import openai
        client = openai.Client(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        def convert_role(role: str) -> str:
            role_map = {"human": "user", "ai": "assistant"}
            return role_map.get(role, role)
        
        formatted_messages = [{"role": convert_role(m.type), "content": m.content} for m in messages]
        response = client.chat.completions.create(
            model="local-model",
            messages=formatted_messages,
            temperature=0.7
        )
        
        message = AIMessage(content=response.choices[0].message.content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "local"

# Make the layout wide for better display
st.set_page_config(page_title="🧠 Brainstormers", layout="wide")

# Sidebar for API key input
st.sidebar.title("⚙️ Settings")
api_type = st.sidebar.selectbox("Select API Type:", ["OpenAI", "Local API"])

if api_type == "OpenAI":
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    if api_key:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
else:
    base_url = st.sidebar.text_input("Enter Local API Base URL:", value="http://localhost:1234/v1")
    if base_url:
        llm = LocalLLM(base_url=base_url)

# Check if API settings are provided
if (api_type == "OpenAI" and api_key) or (api_type == "Local API" and base_url):
    st.title("🧠 Brainstormers")
    st.write("Welcome! Choose a brainstorming mode to start generating ideas for your project.")

    # Define the brainstorming modes, their corresponding functions, and descriptions
    modes = {
        "Big Mind Mapping": {
            "function": lambda query: bmm(query, llm),
            "description": "This involves creating a tree of ideas to explore the maximum amount of ideas in a very wide area. "
                           "This is perfect when you are lost and want to gather the maximum number of ideas."
        },
        "Reverse Brainstorming": {
            "function": lambda query: rb(query, llm),
            "description": "Instead of focusing on solutions, this technique involves identifying ways to cause a problem or "
                           "achieve the opposite effect. Perfect for spotting potential issues and coming up with innovative solutions."
        },
        "Role Storming": {
            "function": lambda query: rs(query, llm),
            "description": "Involves adopting the perspective of someone else to generate ideas. Great for gathering insights from different viewpoints."
        },
        "SCAMPER": {
            "function": lambda query: sc(query, llm),  # Pass `llm` to `sc`
            "description": "SCAMPER stands for Substitute, Combine, Adapt, Modify, Put to another use, Eliminate, and Reverse. "
                           "This method encourages thinking from multiple perspectives to generate diverse ideas."
        },
        "Six Thinking Hats": {
            "function": lambda query: sh(query, llm),
            "description": "This method, developed by Edward de Bono, looks at a problem from six different perspectives: "
                           "White (Data), Red (Emotions), Black (Risks), Yellow (Benefits), Green (Creativity), and Blue (Process management)."
        },
        "Starbursting": {
            "function": lambda query: sb(query, llm),
            "description": "Focuses on generating questions rather than answers using the 5 W's and 1 H (Who, What, Where, When, Why, How). "
                           "Ideal for comprehensive topic exploration."
        }
    }

    # Mode selection
    mode_choice = st.selectbox("Select a brainstorming mode:", list(modes.keys()))

    # Display the description of the selected mode
    if mode_choice:
        st.write(f"**Mode selected:** {mode_choice}")
        st.write(modes[mode_choice]["description"])  # Display mode description

        # User input for idea description
        user_query = st.text_area("Describe your idea in detail to get started:",
                                  "I want idea projects using LangChain that involves AI Agents and that solves social problems.")

        # Button to start the brainstorming process
        if st.button("Generate Ideas"):
            # Display a loading message
            with st.spinner("Generating ideas, please wait..."):
                # Call the function for the selected mode
                result = modes[mode_choice]["function"](user_query)
            
            # Display the result in markdown format
            st.markdown(result)
else:
    # Display a message asking for the API key
    st.title("🧠 Brainstormers")
    st.write("Unlock creative brainstorming methods like Big Mind Mapping, SCAMPER, and Role Storming to spark new ideas. Please configure your API settings in the sidebar to get started!")
