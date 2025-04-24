import os
from langchain_text_splitters import Language


def get_file_ending(language: Language):
  if language == Language.CSHARP:
    return ".cs"
  elif language == Language.JAVA:
    return ".java"
  elif language == Language.PYTHON:
    return ".py"
  elif language == Language.JS:
    return ".js"
  elif language == Language.MARKDOWN:
    return ".md"
  else:
    return ""


def identify_language(folder_path: str):
  try:
    for filename in os.listdir(folder_path):
      if filename.lower().endswith(".cs"):
        return Language.CSHARP
      if filename.lower().endswith(".java"):
        return Language.JAVA
      if filename.lower().endswith(".py"):
        return Language.PYTHON
      if filename.lower().endswith(".js"):
        return Language.JS
      if filename.lower().endswith(".md"):
        return Language.MARKDOWN
    return None
  except FileNotFoundError:
    print(f"Error: Folder not found at '{folder_path}")
    return None
  except NotADirectoryError:
    print(f"Error: '{folder_path}' is not a directory")
    return None


def get_language_from_string(language_string: str):
  if language_string == "csharp":
    return Language.CSHARP
  elif language_string == "java":
    return Language.JAVA
  elif language_string == "python":
    return Language.PYTHON
  elif language_string == "javascript":
    return Language.JS
  elif language_string == "markdown":
    return Language.MARKDOWN
  else:
    return None
