import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel,AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from pymilvus import MilvusClient
import torch
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_milvus import Milvus 
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import T5EncoderModel
from utils.audio_utils import extract_audio_text
from utils.video_utils import extract_video_text
from utils.image_utils import extract_image_text
from utils.document_loaders import (
    process_logs,
    load_text_documents,
    load_word_documents,
    load_pdf_documents,
)



# Function to clear CUDA memory
def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        st.write("CUDA memory cleared successfully.")

# Initialize device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# st.write(f"Using device: {'GPU' if device.type == 'cuda' else 'CPU'}")

# Clear CUDA memory before processing
clear_cuda_memory()

# Initialize Milvus client
client = MilvusClient("milvus_database.db")
client.create_collection(
    collection_name="my_collection",
    dimension=768
)


# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def chunk_documents(data, chunk_size=1000, chunk_overlap=200):
    """Split documents into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "]
    )
    return splitter.split_documents(data)


def process_and_store_documents(documents):
    """Process and store documents in Milvus."""
    clear_cuda_memory()  # Clear memory before processing each batch
    chunks = chunk_documents(documents)
    st.write("Chunks generated:")
    st.write(chunks)
    embeddings=HuggingFaceEmbeddings()
    vectorstore = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,  # Pass the embeddings
        connection_args={"uri": "./milvus_database.db"},
        # drop_old=True,  # Drop old collection
    )

    st.success("Documents processed and stored in Milvus successfully!")

def app():
    st.title("Multimodal Document Processing with LangChain")
    st.subheader("Upload your files or query the Milvus database")

    # Switch between upload and query mode
    mode = st.radio("Choose mode", ["Upload Files", "Query"])

    if mode == "Upload Files":
        uploaded_file = st.file_uploader("Choose a file", type=["mp4", "mp3", "wav", "txt", "jpg", "jpeg", "png", "csv", "yaml", "json", "docx", "pdf"])

        if st.button("Process File"):
            if uploaded_file:
                file_type = uploaded_file.type
                st.write(f"Detected file type: {file_type}")

                documents = None

                if "audio" in file_type:
                    st.audio(uploaded_file, format="audio/wav")
                    st.write("Processing audio...")
                    text = extract_audio_text(uploaded_file)
                    documents = [Document(page_content=text, metadata={"source": "audio", "file_name": uploaded_file.name})]

                elif "video" in file_type:
                    st.video(uploaded_file)
                    st.write("Processing video...")
                    text = extract_video_text(uploaded_file)
                    documents = [Document(page_content=text, metadata={"source": "video", "file_name": uploaded_file.name})]

                elif "image" in file_type:
                    st.image(uploaded_file, caption="Uploaded Image")
                    st.write("Processing image...")
                    documents = extract_image_text(uploaded_file)

                elif "csv" in file_type or "yaml" in file_type or "json" in file_type:
                    st.write("Processing structured logs...")
                    documents = process_logs(uploaded_file, file_type, uploaded_file.name)

                elif "document" in file_type:
                    st.write("Processing Word document...")
                    documents = load_word_documents(uploaded_file)

                elif "pdf" in file_type:
                    st.write("Processing PDF document...")
                    documents = load_pdf_documents(uploaded_file)

                elif "text" in file_type:
                    st.write("Processing text document...")
                    documents = load_text_documents(uploaded_file)

                if documents:
                    st.write("Processing complete. Storing in Milvus...")
                    process_and_store_documents(documents)
                else:
                    st.error("Failed to process the document. Please check the file type.")

    elif mode == "Query":
        query = st.text_input("Enter your query:")
        if st.button("Search"):
            if query:
                st.write(f"Searching for: {query}")
                clear_cuda_memory()  # Clear CUDA memory before querying

                embeddings=HuggingFaceEmbeddings()
                
                vectorstore = Milvus(
                    embedding_function=embeddings,
                    connection_args={"uri": "./milvus_database.db"},
                    collection_name="LangChainCollection",
                )
                # Perform similarity search using the query embedding
                # docs = vectorstore.similarity_search(query, k=2)
                # docs = vectorstore.similarity_search_with_score(query,k=2)
                
                retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1})
                docs=retriever.invoke(query)
                
                st.write(docs)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                model_name = "google/flan-t5-small"

                # Load the tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    token="hf_token")

                # Load the model with proper arguments
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    device_map='auto',
                    torch_dtype=torch.float16,
                    token="hf_token",
                    load_in_8bit=True  # Enable 8-bit quantization
                    # load_in_4bit=True  # Enable 4-bit quantization
                )
                

                # Create a text-generation pipeline
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_new_tokens=512,
                    do_sample=True,
                    top_k=30,
                    num_return_sequences=10,
                    eos_token_id=tokenizer.eos_token_id
                )

                # Wrap the pipeline with HuggingFacePipeline for use with LangChain
                llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})
                
                # st.write(llm.predict('what is llms?'))
###-------------------------------------LLama2------------------------------------------------------------------------------------
                # model = "meta-llama/Llama-2-7b-chat-hf"
                # tokenizer = AutoTokenizer.from_pretrained(
                #             model,
                #             token="hf_token",)


                # model = AutoModelForCausalLM.from_pretrained(model,
                #                         device_map='auto',
                #                         torch_dtype=torch.float16,
                #                         token="hf_token",
                #                         # load_in_8bit=True,
                #                         load_in_4bit=True
                #                         )
                
                # pipe = pipeline("text-generation",
                #                 model=model,
                #                 tokenizer= tokenizer,
                #                 torch_dtype=torch.bfloat16,
                #                 device_map="auto",
                #                 max_new_tokens = 512,
                #                 do_sample=True,
                #                 top_k=30,
                #                 num_return_sequences=1,
                #                 eos_token_id=tokenizer.eos_token_id
                #                 )
                
                # llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0})

                
                
                
                # Prompt template
                PROMPT_TEMPLATE = """
                Human: You are an AI assistant and provide answers to questions by using fact-based and statistical information when possible.
                Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                <context>
                {context}
                </context>

                <question>
                {question}
                </question>

                The response should be specific and use statistics or numbers when possible.

                Assistant:"""

                prompt = PromptTemplate(
                    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
                )
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                # Invoke the RAG chain to generate a response
                res = rag_chain.invoke(query)
                
                # Display the response
                st.write("AI Assistant Response:")
                st.write(res)


if __name__ == "__main__":
    app()
