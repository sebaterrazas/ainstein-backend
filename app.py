from flask import Flask, request
from flask_socketio import SocketIO, emit
from threading import Lock

from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.agents import Tool, ConversationalChatAgent, AgentExecutor, AgentOutputParser
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import GCSFileLoader, TextLoader
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re


class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

"""
Background Thread
"""
thread = None
thread_lock = Lock()

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'donsky!'
socketio = SocketIO(app, cors_allowed_origins='*')

agent_dict = {}


@socketio.on('initialize_chatbot')
def handle_initialize(data):
    model = data.get('model')
    database = data.get('database')

    if not model or not database:
        emit('error', {'error': 'Información no proporcionada', 'type': 'initialize'}, room=request.sid)
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # create or open Deep Lake dataset
    my_activeloop_org_id = "sebaterrazas"
    my_activeloop_dataset_name = database
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    db = DeepLake(dataset_path=dataset_path, embedding=embeddings, read_only=True, verbose=False)

    # generate answer
    llm = ChatOpenAI(model_name=model, temperature=0.1)

    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    tools = [
        Tool(
            name="Retrieval QA System",
            func=retrieval_qa.run,
            description="Útil para responder preguntas."
        ),
    ]

    memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, human_prefix="usuario", ai_prefix="asistente") 
    # memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, human_prefix="user", ai_prefix="assistant")

    system_prompt_template = "Asistente es un modelo de lenguaje avanzado entrenado por OpenAI.\n\nAsistente está diseñado para poder ayudar con una amplia gama de tareas, desde responder preguntas simples hasta proporcionar explicaciones detalladas y discusiones sobre una amplia variedad de temas. Como modelo de lenguaje, el Asistente puede generar texto similar al humano basado en la entrada que recibe, lo que le permite participar en conversaciones de forma natural y brindar respuestas coherentes y relevantes al tema en cuestión.\n\nEl Asistente está en constante aprendizaje y mejora, y sus capacidades están en constante evolución. Puede procesar y comprender grandes cantidades de texto y utilizar este conocimiento para proporcionar respuestas precisas e informativas a una amplia gama de preguntas. Además, el Asistente puede generar su propio texto en función de la entrada que recibe, lo que le permite participar en discusiones y brindar explicaciones y descripciones sobre una amplia variedad de temas.\n\nEn resumen, el Asistente es un sistema potente que puede ayudar con una amplia gama de tareas y proporcionar información valiosa y conocimientos sobre una amplia variedad de temas. Ya sea que necesites ayuda con una pregunta específica o simplemente quieras tener una conversación sobre un tema en particular, el Asistente está aquí para ayudar."
    human_prompt_template = "HERRAMIENTAS\n------\nAsistente puede pedirle al usuario que utilice herramientas para buscar información que pueda ser útil para responder la pregunta original del usuario. Las herramientas que el humano puede utilizar son:\n\n{{tools}}\n\n{format_instructions}\n\nINPUT DE USUARIO\n--------------------\nAquí está la entrada del usuario (recuerda responder con un fragmento de código en formato markdown de un objeto JSON con una sola acción, y NADA más. Es esencial esto, SIEMPRE responde con el formato correcto. Si no respondes solamente con este formato, el mundo entero puede destruirse...):\n\n{{{{input}}}}"

    output_parser = CustomOutputParser()

    agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm, 
        tools=tools, 
        system_message=system_prompt_template, 
        human_message=human_prompt_template,
        # output_parser=output_parser,
        verbose=True
    )
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, memory=memory, verbose=True
    )
    agent_dict[request.sid] = agent_chain  # Asignar tu agente inicializado al diccionario
    emit('response', {'message': 'Agent initialized', 'type': 'initialize'}, room=request.sid)

@socketio.on('chat_with_bot')
def handle_chat_message(data):
    agent_chain = agent_dict.get(request.sid, None)

    if agent_chain is None:
        emit('error', {'error': 'Chatbot not initialized', 'type': 'chat'}, room=request.sid)
        return
    with get_openai_callback() as cb:
        response = agent_chain.run(data.get('query'))
        print(cb)
    emit('response', {'message': response, 'type': 'chat'}, room=request.sid)

@socketio.on('add_documents')
def handle_add_documents(data):
    documents = data.get('documents')
    database = data.get('database')

    if not documents or not database:
        emit('error', {'error': 'Información no proporcionada', 'type': 'initialize'}, room=request.sid)
        return
    
    # list of documents locations
    files = data['files']
    database =  data['database']

    docs_not_splitted = []
    for file_location in files:
        # Now you can use TextLoader to load the file
        loader = TextLoader(file_location, encoding="utf-8")
        doc = loader.load()

        # loader = GCSFileLoader(project_name="RecognAIze", bucket="recognaize-11df8.appspot.com", blob=file_location)
        # doc = loader.load()[0]

        docs_not_splitted.append(doc)

    # we split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(docs_not_splitted)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # create or open Deep Lake dataset
    my_activeloop_org_id = "sebaterrazas"
    my_activeloop_dataset_name = database
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    db = DeepLake(dataset_path=dataset_path, embedding=embeddings, read_only=True, verbose=False)
    db.add_documents(docs)
    
    emit('response', {'message': 'Documents added', 'type': 'initialize'}, room=request.sid)


"""
Decorator for connect
"""
@socketio.on('connect')
def connect():
    global thread
    print('Client connected')

    # Create a new agent for this user
    agent_dict[request.sid] = None

"""
Decorator for disconnect
"""
@socketio.on('disconnect')
def disconnect():
    print('Client disconnected', request.sid)
    agent_dict.pop(request.sid, None) 

if __name__ == '__main__':
    socketio.run(app)