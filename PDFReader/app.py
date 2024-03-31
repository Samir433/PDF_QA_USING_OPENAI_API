import os
import io
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from constants import api_key
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


def main():
    # set the openai enviroment using api_key
    os.environ['OPENAI_API_KEY'] = api_key

    st.title("PDF Reader")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.write("PDF uploaded successfully!")

        # Read PDF contents
        pdf_contents = uploaded_file.read()

        pdf_file_like_object = io.BytesIO(pdf_contents)
        pdfreader = PdfReader(pdf_file_like_object)

        from typing_extensions import Concatenate
        # read text from pdf
        raw_text = ''
        for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
              raw_text += content

        # We need to split the text using Character Text Split such that it sshould not increse token size
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 800,
            chunk_overlap  = 200,
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)

        embeddings = OpenAIEmbeddings()
        document_search = FAISS.from_texts(texts, embeddings)

        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        def bot(query, i):
            if query:
                docs = document_search.similarity_search(query)
                response = chain.run(input_documents=docs, question=query)
                st.text_area("Chatbot Response:", value=response, key=f"response_ker_{i}")
                
                next_query = st.text_input("Enter your next question about the PDF : ", key=i+1)
                
                i += 1
                bot(query=next_query, i=i)

        i = 0
        query = st.text_input("Enter your question about the PDF : ", key=i)
        bot(query=query, i=i)


        i = 0

        # # Display number of pages
        st.write(f"Number of pages in PDF: {len(pdfreader.pages)}")

if __name__ == "__main__":
    main()
