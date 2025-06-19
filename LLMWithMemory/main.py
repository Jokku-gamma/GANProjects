import streamlit as st
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers 

llm=CTransformers(
    model="llama-2-7b-chat.ggmlv3.q2_K.bin",
    model_type="llama",
    config={"max_new_tokens":256,"temperature":0.7}

)

temp="""
You are a friendly chatbot named Jokkubot that can have conversation with a human
{chat_hist}
Human:{human_input}
AI:
"""

prompt=PromptTemplate(
    input_variables=["chat_hist","human_input"],
    template=temp
)

st.set_page_config(page_title="JokkuBot conversational Chatbot",layout="centered")
st.title("JokkuBot")
st.caption("Powered by LLAMA 2 and langchain memeory")


if "messages" not in st.session_state:
    st.session_state.messages=[]
    st.session_state.messages.append({
        "role":"assistant",
        "content":"Hello There. I am Jokkubot. What can i do for you today?"
    })

if "conversation" not in st.session_state:
    st.session_state.memory=ConversationBufferMemory(memory_key="chat_hist")
    st.session_state.conversation=LLMChain(
        llm=llm,
        prompt=prompt,
        memory=st.session_state.memory,
        verbose=True
    )
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt=st.chat_input("What would love to discuss about")
if prompt:
    st.session_state.messages.append(
        {
        "role":"user",
        "content":prompt
        }
    )
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Jokkubot is thinking"):
            resp=st.session_state.conversation.run(prompt)
            st.markdown(resp)
            st.session_state.messages.append(
                {
                    "role":"assistant",
                    "content":resp
                }
            
            )
if st.button("Clear chat"):
    st.session_state.messages=[]
    st.session_state.memory.clear()
    st.rerun()