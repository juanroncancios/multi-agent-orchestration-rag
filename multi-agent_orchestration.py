import os
import json
import sys
import hashlib
from types import ModuleType
from dotenv import load_dotenv

# --- Monkey patch para omitir el DLL bloqueado de xxhash ---
class FakeXXHash:
    def __init__(self, data=b''):
        self._h = hashlib.sha256(data)
    def update(self, data):
        self._h.update(data)
    def digest(self):
        return self._h.digest()[:8]
    def hexdigest(self):
        return self._h.hexdigest()[:16]
    def intdigest(self):
        return int(self._h.hexdigest()[:16], 16)

fake_xxhash = ModuleType("xxhash")
fake_xxhash.xxh64 = lambda data=b'', **kwargs: FakeXXHash(data)
fake_xxhash.xxh3_64 = lambda data=b'', **kwargs: FakeXXHash(data)
fake_xxhash.xxh128 = lambda data=b'', **kwargs: FakeXXHash(data)
fake_xxhash.xxh3_128 = lambda data=b'', **kwargs: FakeXXHash(data)
sys.modules["xxhash"] = fake_xxhash
# --------------------------------------------------------

# 1. Cargar variables de entorno (.env)
load_dotenv()

# Resto de tus imports (langchain, etc.) abajo...

# Verificar API Keys obligatorias
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Falta configurar OPENAI_API_KEY en el archivo .env")

# 2. Imports de LangChain y Langfuse
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler

# Inicializar CallbackHandler de Langfuse para trazabilidad
langfuse_handler = CallbackHandler()


# PASO 1: Carga de Documentos y Creación de Vector Stores (RAG)
print("⏳ Cargando documentos y creando bases vectoriales...")

# Configuración del splitter para asegurar mínimo 50 chunks por dominio
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,      # Tamaño pequeño para partir el texto en suficientes chunks
    chunk_overlap=30     # Traslape entre fragmentos
)

embeddings = OpenAIEmbeddings()

def build_vectorstore(file_path):
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)
    print(f" -> Archivo {file_path}: Generados {len(chunks)} chunks.")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# Crear Retrievers para cada departamento
hr_retriever = build_vectorstore("data-hr_docs-policy_hr.txt")
tech_retriever = build_vectorstore("data-tech_docs-policy_tech.txt")
finance_retriever = build_vectorstore("data-finance_docs-policy_finance.txt")


# PASO 2: Definición de Agentes Especialistas (RAG Chains)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_prompt_template = (
    "Eres un asistente especializado en {domain} para la empresa SaaS.\n"
    "Responde la consulta del usuario basándote ÚNICAMENTE en el siguiente contexto:\n\n"
    "{context}\n\n"
    "Si la respuesta no está en el contexto, di amablemente que no tienes esa información."
)

def create_domain_chain(domain_name, retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template.format(domain=domain_name, context="{context}")),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

hr_chain = create_domain_chain("Recursos Humanos", hr_retriever)
tech_chain = create_domain_chain("Soporte Técnico e Infraestructura", tech_retriever)
finance_chain = create_domain_chain("Finanzas y Facturación", finance_retriever)


# PASO 3: Agente Orquestador (Intent Classifier Router)

router_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Clasifica la solicitud del usuario en exactamente UNA de estas tres categorías: 'hr', 'tech', o 'finance'.\n"
     "Responde ÚNICAMENTE con una de esas tres palabras en minúsculas, sin puntuación ni texto adicional.\n\n"
     "Ejemplos:\n"
     "- Necesito pedir mis vacaciones -> hr\n"
     "- No puedo conectar la VPN -> tech\n"
     "- ¿Cuándo me reembolsan mi factura? -> finance"),
    ("human", "{input}")
])

router_chain = router_prompt | llm

def route_and_execute(user_query):
    """Enruta la consulta al agente correspondiente y ejecuta la RAG Chain."""
    # 1. Clasificación de intención por el Orquestador
    intent_response = router_chain.invoke({"input": user_query})
    intent = intent_response.content.strip().lower()
    
    print(f"\n🔍 Consulta: '{user_query}'")
    print(f"🎯 Intención detectada por el Orquestador: [{intent.upper()}]")

    # 2. Enrutamiento condicional al agente especializado
    if "hr" in intent:
        response = hr_chain.invoke({"input": user_query}, config={"callbacks": [langfuse_handler]})
    elif "tech" in intent:
        response = tech_chain.invoke({"input": user_query}, config={"callbacks": [langfuse_handler]})
    elif "finance" in intent:
        response = finance_chain.invoke({"input": user_query}, config={"callbacks": [langfuse_handler]})
    else:
        # Fallback de seguridad en caso de clasificación ambigua
        response = hr_chain.invoke({"input": user_query}, config={"callbacks": [langfuse_handler]})

    return response["answer"]


# PASO 4: Carga y Ejecución del Golden Dataset (test_queries.json)

test_queries = [
    {"query": "¿Cuántos días de vacaciones tengo al año?", "expected_intent": "hr"},
    {"query": "¿Cuál es la política de licencias por matrimonio?", "expected_intent": "hr"},
    {"query": "¿Cómo solicito el reembolso del curso que hice?", "expected_intent": "hr"},
    {"query": "Mi VPN no conecta desde esta mañana, ¿qué hago?", "expected_intent": "tech"},
    {"query": "Olvidé mi contraseña de la cuenta corporativa", "expected_intent": "tech"},
    {"query": "¿Cuáles son las reglas para las contraseñas de las cuentas?", "expected_intent": "tech"},
    {"query": "¿Cuál es el límite diario para gastos de alimentación en viajes?", "expected_intent": "finance"},
    {"query": "¿Cuándo se emiten las facturas de suscripción?", "expected_intent": "finance"},
    {"query": "¿Qué descuento hay por pago anual anticipado?", "expected_intent": "finance"},
    {"query": "¿Me pueden reembolsar una cerveza que me tomé en un viaje de trabajo?", "expected_intent": "finance"}
]

if __name__ == "__main__":
    print("\n🚀 Iniciando pruebas del Sistema Multi-Agente con Trazabilidad en Langfuse...\n")
    
    for idx, test in enumerate(test_queries, 1):
        print(f"--- Test #{idx} ---")
        respuesta = route_and_execute(test["query"])
        print(f"🤖 Respuesta: {respuesta}\n")

    print("✅ ¡Ejecución finalizada! Revisa tu dashboard en Langfuse para ver las trazas capturadas.")