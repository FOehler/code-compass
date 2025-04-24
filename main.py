import os
import argparse
from dotenv import load_dotenv

from code_compass import CodeCompass
from helpers import get_language_from_string

DEFAULT_CODE_FOLDER = "../nodatime/src/NodaTime"  # example csharp library for testing

load_dotenv()


def main():
  ## Parse arguments
  parser = argparse.ArgumentParser(description="Query a codebase using RAG and Gemini.")
  parser.add_argument(
    "--code-dir",
    type=str,
    default=DEFAULT_CODE_FOLDER,
    help=f"Directory containing the code files (default: {DEFAULT_CODE_FOLDER})",
  )
  parser.add_argument(
    "--language",
    type=str,
    default=None,
    help="Override language used in the code files (default: Auto detect language)",
  )

  args = parser.parse_args()

  if not os.path.isdir(args.code_dir):
    print(f"Error: Code directory not found: {args.code_dir}")
    return

  ## Initialization logic
  print("\n--- Initializing Code Compass ---")
  if not args.language:
    language = CodeCompass.identify_language(args.code_dir)
  else:
    language = get_language_from_string(args.language)
    print(f"Language override active for {language}")
  if not language:
    print("Failed to identify language. Exiting.")
    return

  documents = CodeCompass.load_documents(args.code_dir, language)
  if documents:
    document_chunks = CodeCompass.split_documents(documents, language)
  else:
    print("No documents loaded. Exiting.")
    return

  vectorstore = CodeCompass.create_vector_store(document_chunks)
  if not vectorstore:
    print("Failed to initialize vector store. Exiting.")
    return

  rag_workflow = CodeCompass.setup_rag_workflow(vectorstore)
  if not rag_workflow:
    print("Failed to set up RAG workflow. Exiting.")
    return

  ## Query Loop
  print("\n--- Code Compass ---")
  print(
    "Enter your questions about the provided codebase. Type 'exit' or 'quit' to stop."
  )

  while True:
    try:
      query = input("\nQuestion: ")
      if query.lower() in ["exit", "quit"]:
        break
      if not query:
        continue

      print("Thinking...")

      # Invoke the model here
      answer = rag_workflow.invoke(query)

      print("\nAnswer:")
      print(answer)

    except KeyboardInterrupt:
      print("\nExiting...")
      break
    except Exception as e:
      print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
  main()
