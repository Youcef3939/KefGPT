from pathlib import Path

def setup_folders():
    folders = [
        "data/pdfs",
        "data/vectors",
        "models/llm_models",
        "models/embeddings",
        "sessions/quizzes"
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {folder}")
    
    print("\n✅ Folder structure created!")
    print("\nNext steps:")
    print("1. Place your .gguf LLM model in models/llm_models/")
    print("2. Place your PDF files in data/pdfs/{course_name}/")
    print("3. Run: streamlit run app.py")

if __name__ == "__main__":
    setup_folders()