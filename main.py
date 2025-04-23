import os
import argparse

DEFAULT_CODE_FOLDER = "../nodatime/src/NodaTime"  # example csharp library for testing


def main():
  ## Parse arguments
  parser = argparse.ArgumentParser(
    description="Query a C# codebase using RAG and Gemini."
  )
  parser.add_argument(
    "--code-dir",
    type=str,
    default=DEFAULT_CODE_FOLDER,
    help=f"Directory containing the C# code files (default: {DEFAULT_CODE_FOLDER})",
  )

  args = parser.parse_args()

  if not os.path.isdir(args.code_dir):
    print(f"Error: Code directory not found: {args.code_dir}")
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
      answer = "Dummy Answer"

      print("\nAnswer:")
      print(answer)

    except KeyboardInterrupt:
      print("\nExiting...")
      break
    except Exception as e:
      print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
  main()
