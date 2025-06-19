import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers


def getResponse(inp_text,n,style):
    path="./llama-2-7b-chat.ggmlv3.q2_K.bin"
    try:
        LLM=CTransformers(model=path,model_type="llama",config={'max_new_tokens':int(n),'temperature':0.7,'context_length':4096})

    except Exception as e:
        st.error(f"Error laoding CTransformers model :{e}")
        st.stop()

    template="""
    Write a blog for {blog_style} job profile for a topic {input_text}
    within {no_words} words.
"""
    prompt=PromptTemplate(input_variables=["blog_style","input_text","no_words"],template=template)
    resp=LLM(prompt.format(blog_style=blog_style,input_text=inp_text,no_words=no_words))
    print(resp)
    return resp


st.set_page_config(page_title="Generate Blogs",
                layout="centered",initial_sidebar_state="collapsed")
st.header("Generate Blogs")
inp_text=st.text_input("Enter the blog topic")
col1,col2=st.columns([5,5])
with col1:
    no_words=st.text_input('No of words')
with col2:
    blog_style=st.selectbox('Writing a blog for',('Researchers','Data Scientists','Common People'),index=0)
submit=st.button("Generate")

if submit:
    if not no_words.strip().isdigit() or int(no_words)<=0:
        st.error("Pls enter a valid positive number")
    elif not inp_text.strip():
        st.error("Pls enter a blog topic")
    else:
        with st.spinner("Generating blog"):
            st.write(getResponse(inp_text,no_words,blog_style))
