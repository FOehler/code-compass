from langchain_community.document_loaders import DirectoryLoader, TextLoader


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
