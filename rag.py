# -*- coding: utf-8 -*-
# Asegúrate de tener instaladas las bibliotecas necesarias:
# pip install langchain langchain_community langchain_core psycopg2-binary sentence-transformers faiss-cpu torch
# Si tienes una GPU compatible con FAISS y PyTorch, puedes instalar faiss-gpu
# pip install faiss-gpu

import multiprocessing
import os
import psycopg2 # Para la conexión con PostgreSQL

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import LlamaCppEmbeddings

# --- 1. Configuración del LLM (Modelo de Lenguaje Grande) ---
# Reemplaza '/ruta/a/tu/modelo/local.gguf' con la ruta real a tu modelo LlamaCpp
# Esta variable DEBE ser definida por ti.
# Ejemplo: local_model_path = "/home/user/models/llama-2-7b-chat.Q4_K_M.gguf"
local_model_path = "/app/models/gemma-3-1b-it-q4_0.gguf"

if not os.path.exists(local_model_path):
    print(f"Error: El archivo del modelo no se encuentra en '{local_model_path}'.")
    print("Por favor, descarga un modelo compatible con LlamaCpp (formato GGUF) y actualiza la variable 'local_model_path'.")
    exit()

try:
    llm = ChatLlamaCpp(
        temperature=0.5,
        model_path=local_model_path,
        n_ctx=4096,  # Aumentado para potencialmente más contexto, ajusta según tu modelo y RAM
        n_gpu_layers=16, # Ajusta según tu GPU y modelo. Si no tienes GPU o no es compatible, pon 0.
        n_batch=300,
        max_tokens=250, # La respuesta del LLM tendrá como máximo estos tokens
        n_threads=multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1,
        repeat_penalty=1.5,
        top_p=0.5,
        verbose=True, # Muestra información detallada del proceso del LLM
        # f16_kv=True, # Descomenta si tu modelo y hardware lo soportan para mejor rendimiento
    )
except Exception as e:
    print(f"Error al inicializar ChatLlamaCpp: {e}")
    print("Asegúrate de que la ruta al modelo es correcta y que LlamaCpp está bien configurado.")
    exit()

# --- 2. Configuración y Carga de Datos desde PostgreSQL ---
# Modifica estos detalles para que coincidan con tu configuración de PostgreSQL
DB_CONFIG = {
    "dbname": "ragdb",    # <--- !!! MODIFICA ESTA LÍNEA !!!
    "user": "raguser",        # <--- !!! MODIFICA ESTA LÍNEA !!!
    "password": "ragpassword",   # <--- !!! MODIFICA ESTA LÍNEA !!!
    "host": "localhost",           # O la IP/hostname de tu servidor PostgreSQL
    "port": "5432"                 # Puerto estándar de PostgreSQL
}

# Modifica esta consulta para extraer el texto que quieres que el RAG utilice
# Asegúrate de seleccionar una columna con un ID único si es posible para los metadatos
SQL_QUERY = "SELECT id, contenido_texto FROM tu_tabla_de_conocimiento;" # <--- !!! MODIFICA ESTA LÍNEA !!!

def cargar_documentos_desde_postgres(config_db, consulta_sql):
    """
    Se conecta a PostgreSQL, ejecuta una consulta y devuelve los resultados
    como una lista de objetos Document de LangChain.
    """
    documentos = []
    try:
        conn = psycopg2.connect(**config_db)
        cursor = conn.cursor()
        cursor.execute(consulta_sql)
        filas = cursor.fetchall()
        
        if not filas:
            print("Advertencia: No se encontraron datos en PostgreSQL con la consulta proporcionada.")
            print(f"Consulta: {consulta_sql}")
            return []

        for fila_idx, fila in enumerate(filas):
            # Asume que la primera columna es un ID y la segunda el contenido textual.
            # Ajusta los índices si tu consulta devuelve las columnas en un orden diferente.
            doc_id = str(fila[0]) if len(fila) > 0 else f"fila_{fila_idx}"
            contenido = str(fila[1]) if len(fila) > 1 else str(fila[0]) # Toma el segundo o el primero si solo hay uno

            metadatos = {"fuente": f"postgresql_tabla_X_id_{doc_id}"} # Personaliza tus metadatos
            documentos.append(Document(page_content=contenido, metadata=metadatos))
            
        cursor.close()
        conn.close()
        print(f"Se cargaron {len(documentos)} documentos desde PostgreSQL.")
    except psycopg2.Error as e:
        print(f"Error al conectar o consultar PostgreSQL: {e}")
        print("Verifica la configuración de tu base de datos (DB_CONFIG) y la consulta (SQL_QUERY).")
        return [] # Devuelve lista vacía en caso de error para no detener todo
    except IndexError:
        print(f"Error de índice al procesar las filas de la base de datos. Verifica que tu SQL_QUERY ({SQL_QUERY}) devuelve las columnas esperadas (ID, contenido).")
        return []
    return documentos

# --- 3. División de Texto ---
def dividir_documentos(documentos):
    """
    Divide los documentos cargados en fragmentos más pequeños.
    """
    if not documentos:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Tamaño del fragmento en caracteres
        chunk_overlap=200,  # Superposición entre fragmentos para mantener contexto
        length_function=len,
    )
    fragmentos = text_splitter.split_documents(documentos)
    print(f"Se dividieron los documentos en {len(fragmentos)} fragmentos.")
    return fragmentos

# --- 4. Modelo de Embeddings ---
# Usaremos un modelo de HuggingFace para crear los embeddings (vectores) de los textos.
# "sentence-transformers/all-MiniLM-L6-v2" es ligero y bueno para empezar.
# --- 4. Modelo de Embeddings (usando LlamaCppEmbeddings para GGUF) ---
embedding_model_gguf_path = "/app/models/nomic-embed-text-v1.5.Q3_K_S.gguf" # Ruta a tu modelo GGUF
print(f"Cargando modelo de embeddings GGUF desde: {embedding_model_gguf_path}...")
try:
    embeddings = LlamaCppEmbeddings(
        model_path=embedding_model_gguf_path,
        # Puedes añadir otros parámetros de LlamaCpp aquí si son necesarios
        # y compatibles con tu modelo de embedding GGUF.
        # Por ejemplo:
        # n_gpu_layers=0,  # Ajusta si quieres usar GPU y tu llama.cpp build lo soporta para embeddings
        # n_batch=512,     # Tamaño del batch para procesar embeddings (default suele ser 512)
        # n_ctx=512,       # Contexto del modelo de embeddings (default suele ser 512)
                           # Asegúrate de que este valor sea adecuado para el modelo Nomic.
                           # Nomic Embed v1.5 puede manejar hasta 8192 tokens, pero para embeddings
                           # usualmente se usan contextos más cortos. Verifica la documentación del GGUF específico.
        verbose=False      # Puedes ponerlo en True para más detalle durante la carga/uso
    )
    print("Modelo de embeddings GGUF cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo de embeddings GGUF: {e}")
    print(f"Asegúrate de que la ruta '{embedding_model_gguf_path}' es correcta y el archivo GGUF es válido.")
    print("También verifica que tu build de llama-cpp-python es compatible.")
    exit()


# --- 5. Vector Store (Almacén de Vectores) ---
# Usaremos FAISS, una biblioteca eficiente para búsqueda de similitud.
# El índice FAISS se puede guardar y cargar para evitar reprocesar los embeddings cada vez.
FAISS_INDEX_PATH = "faiss_index_postgres_data"

def crear_o_cargar_vectorstore(fragmentos_documentos, embeddings_model):
    """
    Crea un nuevo VectorStore FAISS si no existe uno guardado, o carga uno existente.
    """
    if os.path.exists(FAISS_INDEX_PATH) and os.listdir(FAISS_INDEX_PATH): # Verifica que la carpeta exista y no esté vacía
        print(f"Cargando VectorStore FAISS desde '{FAISS_INDEX_PATH}'...")
        try:
            vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
            print("VectorStore FAISS cargado.")
            # Verificar si el vectorstore cargado tiene documentos
            if not vectorstore.index_to_docstore_id: # Chequeo simple
                 print("Advertencia: El VectorStore cargado parece estar vacío. Se reconstruirá.")
                 raise FileNotFoundError # Forzar reconstrucción
        except Exception as e:
            print(f"No se pudo cargar el VectorStore FAISS (o estaba vacío): {e}. Se creará uno nuevo.")
            if not fragmentos_documentos:
                print("Error: No hay fragmentos de documentos para crear un nuevo VectorStore.")
                return None
            print("Creando nuevo VectorStore FAISS...")
            vectorstore = FAISS.from_documents(fragmentos_documentos, embeddings_model)
            vectorstore.save_local(FAISS_INDEX_PATH)
            print(f"Nuevo VectorStore FAISS creado y guardado en '{FAISS_INDEX_PATH}'.")
    else:
        if not fragmentos_documentos:
            print("Error: No hay fragmentos de documentos para crear un nuevo VectorStore y no hay uno guardado.")
            return None
        print(f"No se encontró VectorStore FAISS en '{FAISS_INDEX_PATH}'. Creando uno nuevo...")
        vectorstore = FAISS.from_documents(fragmentos_documentos, embeddings_model)
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"Nuevo VectorStore FAISS creado y guardado en '{FAISS_INDEX_PATH}'.")
    return vectorstore

# --- 6. Prompt Template Personalizado ---
# Este prompt guía al LLM sobre cómo usar el contexto y cómo responder.
prompt_template_str = """Eres un asistente de IA conversacional y servicial llamado 'PostgresRAG'. Tu tarea es responder a las preguntas del usuario basándote ÚNICAMENTE en el contexto proporcionado.
Si la información no está en el contexto, debes decir explícitamente "Basándome en la información que tengo, no puedo responder a esa pregunta".
No inventes respuestas ni utilices conocimiento externo.
Responde en español de forma concisa y en menos de 250 caracteres.

Contexto:
{context}

Pregunta: {question}

Respuesta útil:"""

PROMPT = PromptTemplate(
    template=prompt_template_str, input_variables=["context", "question"]
)

# --- 7. Cadena de Recuperación y Preguntas (RAG Chain) ---
def crear_cadena_rag(llm_model, vector_store, prompt_template):
    """
    Crea la cadena RetrievalQA que une el recuperador y el LLM.
    """
    if vector_store is None:
        print("Error: El VectorStore no está inicializado. No se puede crear la cadena RAG.")
        return None

    retriever = vector_store.as_retriever(
        search_type="similarity", # Otros tipos: "mmr" (Maximal Marginal Relevance)
        search_kwargs={"k": 3}    # Número de documentos relevantes a recuperar
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff", # "stuff" es bueno si el contexto cabe en el prompt.
                            # Otros: "map_reduce", "refine" para contextos más grandes.
        retriever=retriever,
        return_source_documents=True, # Para ver qué documentos se usaron
        chain_type_kwargs={"prompt": prompt_template} # Aplicamos nuestro prompt personalizado
    )
    print("Cadena RAG creada.")
    return qa_chain

# --- Flujo Principal del Script ---
def main():
    print("Iniciando el proceso RAG...")

    # Cargar documentos desde PostgreSQL
    documentos_db = cargar_documentos_desde_postgres(DB_CONFIG, SQL_QUERY)
    if not documentos_db:
        print("No se cargaron documentos desde la base de datos. El RAG no funcionará sin datos.")
        print("Por favor, verifica tu configuración de DB_CONFIG, SQL_QUERY y que la tabla tenga datos.")
        return

    # Dividir documentos
    fragmentos = dividir_documentos(documentos_db)
    if not fragmentos:
        print("No se generaron fragmentos de texto. El RAG no funcionará.")
        return

    # Crear o cargar VectorStore
    # Pasamos los fragmentos solo si necesitamos crear uno nuevo.
    # Si se carga uno existente, los fragmentos originales no son estrictamente necesarios en este punto
    # (a menos que quieras lógica para re-indexar si los fragmentos cambian).
    vectorstore = crear_o_cargar_vectorstore(fragmentos, embeddings)
    if vectorstore is None:
        print("Fallo al inicializar el VectorStore. Terminando.")
        return

    # Crear la cadena RAG
    rag_chain = crear_cadena_rag(llm, vectorstore, PROMPT)
    if rag_chain is None:
        print("Fallo al crear la cadena RAG. Terminando.")
        return

    print("\n--- Chatbot RAG con PostgreSQL Iniciado ---")
    print("Escribe 'salir' para terminar el chat.")

    while True:
        pregunta_usuario = input("\nTú: ")
        if pregunta_usuario.lower() == 'salir':
            print("Chatbot terminado. ¡Hasta luego!")
            break
        if not pregunta_usuario.strip():
            continue

        print("Procesando tu pregunta...")
        try:
            # El chain espera un diccionario con la clave "query"
            resultado = rag_chain.invoke({"query": pregunta_usuario})
            
            print("\nPostgresRAG:", resultado["result"])
            
            if resultado.get("source_documents"):
                print("\n  Documentos fuente consultados:")
                for idx, doc in enumerate(resultado["source_documents"]):
                    # Mostrar solo una parte del contenido para brevedad
                    contenido_breve = doc.page_content[:150].replace('\n', ' ') + "..."
                    print(f"    {idx+1}. Fuente: {doc.metadata.get('fuente', 'N/A')}, Contenido: '{contenido_breve}'")
        except Exception as e:
            print(f"Error durante la invocación de la cadena RAG: {e}")

if __name__ == "__main__":
    # Validaciones iniciales antes de llamar a main()
    if local_model_path == "/ruta/a/tu/modelo/local.gguf":
        print("Error Crítico: Debes configurar 'local_model_path' en el script.")
        print("Apunta esta variable al archivo GGUF de tu modelo LlamaCpp.")
    elif DB_CONFIG["dbname"] == "tu_basededatos" or SQL_QUERY == "SELECT id, contenido_texto FROM tu_tabla_de_conocimiento;":
        print("Advertencia: Parece que no has modificado la configuración de la base de datos (DB_CONFIG) o la consulta SQL (SQL_QUERY).")
        print("Por favor, actualízalas con tus datos reales para que el script funcione correctamente.")
        # Podrías decidir salir aquí o permitir que el usuario continúe bajo su propio riesgo
        continuar = input("¿Deseas continuar con la configuración por defecto (probablemente fallará)? (s/N): ")
        if continuar.lower() != 's':
            exit()
        main()
    else:
        main()

