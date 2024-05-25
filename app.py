import os, re
from llama_parse import LlamaParse
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_core.runnables import RunnablePassthrough
import pickle
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant

from flask import Flask, render_template, request, redirect, session
from werkzeug.utils import secure_filename


groq_apikey = "" 
os.environ['LLAMA_CLOUD_API_KEY'] = ""
app = Flask(__name__)
api_key = os.getenv("OPENAI_API_KEY")

vectorstore = None
conversation_chain = None

if not os.path.exists('uploads'):
    os.makedirs('uploads')
    
UPLOAD_FOLDER = os.path.join('uploads')

load_chat_engine = True

app.secret_key = 'super secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.environ['OPENAI_API_KEY'] = api_key

datafile = "./data/datafile.pkl"

def get_documents(pdf_path): 
    parser = LlamaParse(result_type="markdown") # "reports/Report PDF 2.pdf" 
    if os.path.exists(datafile):
        with open(datafile, "rb") as f:
            parsed_documents = pickle.load(f)

    else:
        llama_parse_documents = parser.load_data(pdf_path)
        with open(datafile, "wb") as f:
            pickle.dump(llama_parse_documents, f)
        parsed_documents = llama_parse_documents
    return parsed_documents

def get_markdown_text(pdf_path):
    llama_parse_documents = get_documents(pdf_path) # get_documents()
    with open('data/output.md', 'a', encoding="utf-8") as f:
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')

    markdown_path = "data/output.md"
    loader = UnstructuredMarkdownLoader(markdown_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks


def get_vectorstore(docs):
    #docs = get_documents()
    embeddings = FastEmbedEmbeddings()
    index_name = "langchain-test-index"
    
    qdrant = Qdrant.from_documents(
        docs,
        embeddings,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name="my_documents",
    )
    
    return qdrant, embeddings

# with langchain
def Chain(llm_name, vectorstore):
    
    #vectorstore, embed_model = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={'k':15})
    custom_prompt_template = """
    You are a General Medical Practitioner. 
    You will be provided medical reports analyse the medical reports and answer user questions in detailed manner. 
    Answer only if you know the answer. Do not answer the question if you dont know the answer.
    
    context - {context}
    question - {question}
    
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    
    prompt = PromptTemplate.from_template(template=custom_prompt_template)
    if llm_name == "mistral":
        chat = ChatGroq(temperature=0, api_key=groq_apikey, model_name="mixtral-8x7b-32768")
    elif llm_name == "llama3":
        chat = ChatGroq(temperature=0, api_key=groq_apikey, model_name="llama3-8b-8192")
    elif llm_name == "gpt":
        chat = ChatOpenAI(model = "gpt-4-turbo", temperature=0)
    
    chain = LLMChain(llm=chat, prompt=prompt)
    
    rag_chain = (
        {'context': retriever, "question": RunnablePassthrough()} |
        chain
    )
    
    return rag_chain

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_document():
    
    global load_chat_engine
    
    pdf_docs = request.files['pdf_docs']
    if pdf_docs:
        load_chat_engine = True
    
    filename = secure_filename(pdf_docs.filename)
    pdf_docs.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    session['pdf_path'] = pdf_path
    return redirect('/chat') 

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    chat_history=[]
    global load_chat_engine
    pdf_path = session.get('pdf_path')
    
    if load_chat_engine == True:
        
        docs = get_markdown_text(pdf_path)
        app.vectorstore, embed_model = get_vectorstore(docs)
    
        os.remove('data/output.md')
        os.remove(datafile)
        load_chat_engine = False
        
    try:
        if request.method == "POST":
            user_question = request.form['user_question']
            llm_name = request.form.get('models')
            try:
                
                llm_chain = Chain(llm_name, app.vectorstore)
                conversational_chain = llm_chain
                
                response = conversational_chain.invoke(user_question)['text']
              #  print(f"response 1 = {response} \n")
                response = re.sub(r'\n', '<br>', response)
              #  print(f"response 2 = {response} \n")
                pattern = r'\*\*(.*?)\*\*'
                response = re.sub(pattern, r'<b>\1</b>', response)
              #  print(f"response 3 = {response} \n")
                chat_history.append(user_question)
                chat_history.append(response)
                
            except Exception as e:
                print(f"Exception 1 {e}")
                
    except Exception as e:
        print(f'Exception 2 - {e}')

    return render_template('chat.html', chat_history=chat_history)


if __name__ == '__main__':
    app.run(debug=True, port=8080) 