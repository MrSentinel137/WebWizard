from shiny import App, ui

import json, re
from Utils.DuckDuckGo import DuckDuckGoSearchResults
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.load import dumps, loads
from dotenv import load_dotenv

load_dotenv()

#### Functions
def generate_prompt(query):
    return f"""
        You are an intelligent assistant that helps users create search queries based on their input. When a user provides a query, your task is to identify relevant search queries that can be used for web searches. If the user’s query is too long or complex, break it down into multiple simpler search queries. 
        
        Your output should always be in the following JSON format:
        {{
        "search_query": ["search1", "search2", ...]
        }}
        Make sure to maintain the specified format strictly and provide clear and concise search queries.
        User query: "{query}"
        """

def extract_data(input_str):
    # Regular expression to find snippets and links
    pattern = r'snippet:\s*(.*?);?\s*title:\s*(.*?);?\s*link:\s*(https?://[^\s]+)'
    
    matches = re.findall(pattern, input_str)
    data_list = []
    
    for match in matches:
        snippet, title, link = match
        data_list.append({
            "snippet": snippet.strip(),
            "title" : title.strip(),
            "link": link.strip()
        })
    
    return data_list


def SearchLoader(query, num_results,llm):
    messages = [
        (
            "system",
            "You are a helpful assistant that returns output in json format",
        ),
        ("human", generate_prompt(query)),
    ]

    results = llm.invoke(messages).content
    js = json.loads(results)
    src_query = js["search_query"]

    web_search = DuckDuckGoSearchResults(num_results=num_results)

    src_info = []
    documents = []

    for qry in src_query:
        res = web_search.invoke(qry)
        data = extract_data(res)

        src_info.append({
            "search" : qry,
            "urls" : [doc["link"] for doc in data]
        })

        cc = ""
        for d in data:
            rr = d["snippet"]
            cc = cc + rr + "\n"
        
        document = Document(
            page_content=cc,
        )
        documents.append(document)

    return documents, src_info

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

### User Interface
models = ["llama-3.1-70b-versatile", "mixtral-8x7b-32768", "llama3-8b-8192", "llama3-70b-8192", "llama3-groq-70b-8192-tool-use-preview"]
modes = ["RAG Fusion", "Multi Query RAG"]

app_ui = ui.page_fillable(
    ui.card(  
        ui.card_header("Web Wizard"),
        ui.layout_sidebar(  
            ui.sidebar("Parameters",
                        ui.input_select("models", "Select a LLM model", choices=models),
                        ui.input_slider("temp", "Adjust the temperature", min=0.00, max=1.00,step=0.05, value=0.00),
                        ui.input_slider("n_results", "Number of Search Results", min=5, max=25, value=10),
                        ui.input_select("rag_type", "Select type of retrieval", choices=modes),
                        bg="#f8f8f8"),  
            ui.chat_ui("chat", width='min(750px, 100%)', height='auto', fill=True),  
        ),
    )  
)

# welcome message
welcome = ui.markdown(
    """
    ##### **Web Wizard**

    Welcome to **Web Wizard**, your intelligent companion for navigating the vast world of the internet! 🌐✨
    """
)

messages = [welcome]
history = []

def server(input, output, session):
    ui.card_header("Card with sidebar")
    
    chat = ui.Chat(id="chat", messages=messages)

    # Define a callback to run when the user submits a message
    @chat.on_user_submit
    async def _():
        # Get the user's input
        user_prompt = chat.user_input()
        user_prompt = f"{user_prompt}\n\nMake the output as detailed as possible and output should be in markdown format"
        
        # Access the input values
        selected_model = input.models()
        temperature = input.temp()
        n_results = input.n_results()
        search_type = input.rag_type()

        llm = ChatGroq(temperature=temperature, 
                model_name=selected_model
            )

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Load and preproccess documents
        documents, src_info = SearchLoader(user_prompt, n_results, llm)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        chat_history = []

        # RAG
        system_template = """Answer the following question based on this context:
        Make sure the output is in mardown format.
        {context}

        Also Take into consideration the Chat History:
        {chat_history}

        Question: {question}
        """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )

        if search_type == "RAG Fusion":
            # RAG-Fusion
            template = """"You are a helpful assistant designed to generate multiple search queries based on a single input query. 
            Please generate four different search queries related to the following question: {question}"""

            prompt_rag_fusion = ChatPromptTemplate.from_template(template)

            generate_queries = (
                prompt_rag_fusion
                | llm
                | StrOutputParser() 
                | (lambda x: x.split("\n"))
            )

            retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
            
            final_rag_chain = (
                {"context": retrieval_chain_rag_fusion, 
                "question": itemgetter("question"),
                "chat_history" : itemgetter("chat_history")} 
                | qa_prompt
                | llm
                | StrOutputParser()
            )

        else:
            # Multi Query
            template = """You are an AI language model assistant. 
            Your task is to create five different variations of the given user question to retrieve relevant 
            documents from a vector database. By offering multiple perspectives on the user question, your goal 
            is to help the user mitigate the limitations of distance-based similarity searches. 
            Present these alternative questions separated by newlines. Original 
            question: {question}"""

            prompt_perspectives = ChatPromptTemplate.from_template(template)
            
            generate_queries = (
                prompt_perspectives
                | llm
                | StrOutputParser() 
                | (lambda x: x.split("\n"))
            )

            retrieval_chain = generate_queries | retriever.map() | get_unique_union
            
            final_rag_chain = (
                {"context": retrieval_chain, 
                "question": itemgetter("question"),
                "chat_history" : itemgetter("chat_history")} 
                | qa_prompt
                | llm
                | StrOutputParser()
            )


        res = final_rag_chain.invoke({"question":user_prompt, "chat_history" : chat_history})

        history.append(HumanMessage(content=user_prompt))
        history.append(AIMessage(content=res))

        # Append a response to the chat
        await chat.append_message(
            ui.markdown(res)
        )


app = App(app_ui, server)
