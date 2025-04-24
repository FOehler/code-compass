from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from helpers import identify_language, get_file_ending


class CodeCompass:
  def identify_language(folder_path: str):
    language = identify_language(folder_path)
    print(f"Detected language: {language}")
    return language

  def load_documents(folder_path: str, language: Language):
    """Loads files from the specified directory."""
    print(f"Loading files from: {folder_path}")

    loader = DirectoryLoader(
      folder_path,
      glob=f"**/*{get_file_ending(language)}",
      loader_cls=TextLoader,  # loader for text files
      loader_kwargs={"autodetect_encoding": True},
      show_progress=True,
    )
    try:
      documents = loader.load()
      if not documents:
        print(f"No '.{get_file_ending(language)}' files found in '{folder_path}'.")
        return []
      print(f"Loaded {len(documents)} files from '{folder_path}'.")
      return documents
    except Exception as e:
      print(f"Error loading files from '{folder_path}': {e}")
      return []

  def split_documents(documents, language):
    """Splits documents into chunks for improved emedding into the vector database"""
    print("Splitting documents for embedding...")
    csharp_splitter = RecursiveCharacterTextSplitter.from_language(
      language=language,
      chunk_size=1500,  # suggested chunk size and overlap by Gemini, look into optimization
      chunk_overlap=150,
    )
    document_chunks = csharp_splitter.split_documents(documents)
    print(f"Split into {len(document_chunks)} chunks.")
    return document_chunks

  def create_vector_store(document_chunks):
    """Creates a FAISS vector store from document chunks"""
    print("Preparing embedding model...")
    embeddings = HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2"
    )  # maybe make embedding model configurable
    try:
      vectorstore = FAISS.from_documents(document_chunks, embeddings)
      vectorstore.save_local("faiss_document_index")  # maybe also make configurable
      print("Vector store created and saved.")
      return vectorstore
    except Exception as e:
      print(f"Error creating vector store: {e}")
      return None

  def setup_rag_workflow(vectorstore):
    """Sets up the RAG workflow"""
    if vectorstore is None:
      print("Vector store not available. Cannot set up RAG chain.")
      return None

    print("Setting up RAG workflow...")
    try:
      llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash"
      )  # model could be made configurable
    except Exception as e:
      print(f"Error initializing Gemini LLM: {e}")
      print("Please ensure your GOOGLE_API_KEY environment variable is set correctly.")
      return None

    retriever = vectorstore.as_retriever(
      search_type="similarity",
      search_kwargs={"k": 5},  # retrieve most similar 5 chunks
    )

    prompt_template = """
      You are an AI assistant specialized in analyzing C# code.
      Answer the following question based *only* on the provided context:

      Context:
      {context}

      Question:
      {question}

      Answer:
      """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Define how to format the retrieved documents
    def format_documents(docs):
      return "\n\n".join(doc.page_content for doc in docs)

    # Create the RAG chain using LangChain Expression Language
    # 1. RunnableParallel allows parallel operations
    # - It assembles the context by taking the output from the retriever and merging it into a single string
    # - The question is passed through without any modification
    # 2. The output of runnable parallel is piped as an input to the prompt
    # 3. The prompt is fed into the llm
    # 4. The StrOutputParser extracts the string from the LLM
    rag_workflow = (
      RunnableParallel(
        {"context": retriever | format_documents, "question": RunnablePassthrough()}
      )
      | prompt
      | llm
      | StrOutputParser()
    )

    return rag_workflow
