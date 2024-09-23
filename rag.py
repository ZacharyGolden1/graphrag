from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.document_loaders import PyPDFLoader  # Assuming you have this class for loading PDFs
import langchain.hub as hub

CONCEPTS_TEXT_BOOK_PATH = "/Users/golden/Desktop/CMU/Y1/Spring21/21-127 Spring 2021"
ALGORITHMS_TEXT_BOOK_PATH = "/Users/golden/Desktop/Books/Textbooks/Algorithms (Erickson, Jeff) (z-lib.org).pdf"
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

def main(query):
    # Load the textbook
    loader = PyPDFLoader(ALGORITHMS_TEXT_BOOK_PATH)
    pages = loader.load_and_split()
    docs = [item.page_content for item in pages]  # Get document content

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
        print("Top Retrieved Document:\n")
        for document in retrieved_docs:
            print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ START DOCUMENT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(document.page_content)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END DOCUMENT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    # Cleanup vectorstore
    vectorstore.delete_collection()

# Example usage
print("Ask a question about computer science algorithms")
query = input()  # Input query from the user

main(query)