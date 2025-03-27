from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd
import config

# Load environment variables
load_dotenv()

# Function to load and process the JSON data
def load_suggestions_data(file_path=config.SUGGESTIONS_DATA_PATH):
    return pd.read_json(file_path).to_dict(orient='records')

# Function to convert JSON data to documents for RAG
def create_documents_from_json(data):
    documents = []
    for employee_data in data:
        # Create a document for each employee
        employee_doc = f"Employé: {employee_data['employe']}\n"
        employee_doc += f"Évaluation: {employee_data['evaluation']}\n"
        employee_doc += f"Score: {employee_data['score']}\n"
        employee_doc += "Suggestions:\n"

        # Add each suggestion
        for i, suggestion in enumerate(employee_data['suggestions'], 1):
            employee_doc += f"  {i}. Type: {suggestion['type']}\n"
            employee_doc += f"     Contenu: {suggestion['content']}\n"
            employee_doc += f"     Source: {suggestion['source']}\n"

        documents.append(Document(page_content=employee_doc, metadata={"employe": employee_data['employe']}))

    return documents

# Function to setup the RAG pipeline
def setup_rag_pipeline():
    # Load the data
    data = load_suggestions_data()

    # Create documents
    documents = create_documents_from_json(data)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Create prompt template
    template = """
    Vous êtes un assistant RH qui aide à analyser les données d'évaluation des employés et leurs suggestions.
    Utilisez les informations contextuelles suivantes pour répondre à la question de l'utilisateur.
    Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas.
    
    Contexte: {context}
    
    Question: {question}
    
    Réponse:
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    # Setup LLM
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)
    
    # Create chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Main function to run the RAG application
def main():
    # Setup the RAG pipeline
    rag_chain = setup_rag_pipeline()
    
    print("\n=== Système de Questions-Réponses sur les Recommandations des Employés ===")
    print("Tapez 'exit' pour quitter.\n")

    while True:
        question = input("\nVotre question: ")
        if question.lower() == 'exit':
            break

        # Get response
        response = rag_chain.invoke(question)
        print("\nRéponse:")
        print(response)

if __name__ == "__main__":
    main()
