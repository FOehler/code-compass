from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


class CodeCompass:
  def load_documents(folder_path: str):
    """Loads files from the specified directory."""
    print(f"Loading files from: {folder_path}")

    loader = DirectoryLoader(
      folder_path,
      glob="**/*.cs",  # for now assume csharp files
      loader_cls=TextLoader,  # loader for text files
      loader_kwargs={"autodetect_encoding": True},
      show_progress=True,
    )
    try:
      documents = loader.load()
      if not documents:
        print(f"No '.cs' files found in '{folder_path}'.")
        return []
      print(f"Loaded {len(documents)} files from '{folder_path}'.")
      return documents
    except Exception as e:
      print(f"Error loading files from '{folder_path}': {e}")
      return []

  def split_documents(documents):
    """Splits documents into chunks for improved emedding into the vector database"""
    print("Splitting documents for embedding...")
    csharp_splitter = RecursiveCharacterTextSplitter.from_language(  # again assume csharp for now
      language=Language.CSHARP,
      chunk_size=1500,  # suggested chunk size and overlap by Gemini, look into optimization
      chunk_overlap=150,
    )
    document_chunks = csharp_splitter.split_documents(documents)
    print(f"Split into {len(document_chunks)} chunks.")
    return document_chunks
