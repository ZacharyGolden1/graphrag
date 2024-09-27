from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.document_loaders import PyPDFLoader  # Assuming you have this class for loading PDFs
from langchain_community.document_loaders import TextLoader

import langchain.hub as hub
import os
import tempfile


CONCEPTS_TEXT_BOOK_PATH = "/Users/golden/Desktop/CMU/Y1/Spring21/21-127 Spring 2021"
ALGORITHMS_TEXT_BOOK_PATH = "/Users/golden/Desktop/Books/Textbooks/Algorithms (Erickson, Jeff) (z-lib.org).pdf"
PATH_TO_OBSIDIAN_NOTES = "/Users/zgolden/Desktop/Work"

EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Custom SBERT Embeddings
class SBERTEmbeddings(Embeddings):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def format_docs(docs):
    """Helper function to format documents"""
    return "\n\n".join(doc.page_content for doc in docs)

def concatenate_md_files(folder_path):
    concatenated_content = ""
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as md_file:
                        concatenated_content += md_file.read() + "\n\n"
                except Exception as e:
                    print(f"Error reading file {file_path}: {str(e)}")
    
    return concatenated_content

def main(query):
    # Create a temporary file and write the concatenated content to it
    with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False, suffix='.txt') as temp_file:    
        temp_file.write(concatenate_md_files(PATH_TO_OBSIDIAN_NOTES))
        temp_file_path = temp_file.name

    # Load the textbook
    # loader = PyPDFLoader(ALGORITHMS_TEXT_BOOK_PATH)
    loader = TextLoader(temp_file_path)
    pages = loader.load_and_split()

    # Create Chroma vectorstore with SBERT embeddings
    vectorstore = Chroma.from_documents(documents=pages, embedding=SBERTEmbeddings())

    # Set up the retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # Load the prompt from the hub
    prompt = hub.pull("rlm/rag-prompt")

    # Retrieve relevant documents based on the query
    retrieved_docs = retriever.get_relevant_documents(query)

    # Print the first retrieved document
    if retrieved_docs:
        print("\nTop Retrieved Documents:\n")
        iter = 0
        for document in retrieved_docs:
            iter += 1
            print(f"\n{iter}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ START DOCUMENT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(document.page_content)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END DOCUMENT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
            break
    print(f"returned {len(retrieved_docs)} docs")
    # Cleanup vectorstore
    vectorstore.delete_collection()

# Example usage
print("Ask a question about what I have been working on")
query = input()  # Input query from the user

main(query)