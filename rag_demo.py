# Simple RAG Demo using LangChain and OpenAI
# Import necessary libraries
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Initialize OpenAI client
client = OpenAI(
    api_key="xyz"  # Replace with your actual API key
)

# Step 1: Load PDF document
# This loads a PDF file as our knowledge source
loader = PyPDFLoader("D:\Aditya\e2enlp-2025\Aditya_Balu_PhD_Dissertation__Final_.pdf")  # Replace with your PDF file path
documents = loader.load()

print(f"Loaded {len(documents)} page(s) from PDF")

# Step 2: Split documents into chunks
# Splits the document into smaller pieces to fit within model context windows
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Maximum characters per chunk
    chunk_overlap=200,  # Overlap to maintain context between chunks
    length_function=len
)
chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks")

# Step 3: Create embeddings and store in vector database
# Creates vector representations of text chunks for semantic search
embeddings = OpenAIEmbeddings(
    api_key="xyz"  # Replace with your actual API key
)
# FAISS is an efficient similarity search library
vectorstore = FAISS.from_documents(chunks, embeddings)

print("Vector database created successfully")

# Step 4: Set up retriever
# The retriever will fetch relevant chunks based on the query
retriever = vectorstore.as_retriever(
    search_type="similarity",  # Use semantic similarity for search
    search_kwargs={"k": 3}  # Return top 3 most relevant chunks
)

# Step 5: Create prompt template
# This template will guide the model to use the retrieved context
prompt_template = """
Answer the question based only on the following context:

{context}

Question: {question}

If the information to answer the question is not present in the context, 
respond with "I don't have enough information to answer this question."

Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Step 6: Initialize the LLM
# Using the newer OpenAI API structure
llm = ChatOpenAI(
    api_key="xyz",  # Replace with your actual API key
    model_name="gpt-4o-mini",  # You can change to your preferred model
    temperature=0  # Set to 0 for more deterministic answers
)

# Step 7: Create the RAG pipeline
# This combines retrieval and generation into one chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" method: simply stuffs all retrieved documents into prompt
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

# Step 8: Query the system
# Test with a sample question
query = "What is the main topic of the document?"  # Replace with your test question
response = qa_chain.invoke({"query": query})

print("\nQuestion:", query)
print("Answer:", response["result"])

# Step 9: Interactive demo (optional)
# Let users ask questions until they type 'exit'
def interactive_demo():
    print("\n--- RAG Interactive Demo ---")
    print("Type 'exit' to end the demo")
    
    while True:
        user_query = input("\nEnter your question: ")
        if user_query.lower() == 'exit':
            break
            
        response = qa_chain.invoke({"query": user_query})
        print("Answer:", response["result"])
        
        # Optional: Display source chunks
        print("\nSource chunks (for reference):")
        docs = retriever.get_relevant_documents(user_query)
        for i, doc in enumerate(docs):
            print(f"Chunk {i+1}: {doc.page_content[:100]}...")

# Uncomment to run interactive demo
interactive_demo()