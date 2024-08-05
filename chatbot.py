from flask import Flask, request, jsonify, render_template, Response
import llama_cpp
import psycopg2
from typing import Optional, Any, List
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.postgres import PGVectorStore

llama_cpp.llama_backend_init(numa=False)

app = Flask(__name__)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

llama = None
query_engine = None

db_name = "vector_db"
host = "localhost"
port = "5432"
user = "<user>"  # Replace with your user
password = "<password>" # Replace with your password
conn = psycopg2.connect(
    dbname="postgres",
    host=host,
    password=password,
    port=port,
    user=user,
)
conn.autocommit = True

with conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")
    c.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id SERIAL PRIMARY KEY,
        role TEXT NOT NULL,
        content TEXT NOT NULL
    )
    ''')

vector_store = PGVectorStore.from_params(
    database=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
    table_name="llama2_paper",
    embed_dim=384,
)

class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores

def load_llm():
    global llama
    global query_engine
    if llama is None:
        llama = LlamaCPP(
            model_url="https://huggingface.co/SanctumAI/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/meta-llama-3-8b-instruct.Q8_0.gguf",
            max_new_tokens=256,
            context_window= 8192,
            generate_kwargs={},
            model_kwargs={
                "n_gpu_layers": 34
                },
            verbose=True,
        )

    loader = PyMuPDFReader()
    documents = loader.load(file_path="./llama2.pdf")

    text_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64,
    )

    text_chunks = []

    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text = text_chunk,
        )
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    vector_store.add(nodes)

    retriever = VectorDBRetriever(
        vector_store, embed_model, query_mode="default", similarity_top_k=2
    )

    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llama, streaming=True)

model = "gpt-3.5-turbo"
# chat_history = []

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/stream', methods=['POST'])
def stream():
    user_query = request.json['user_query']
    with conn.cursor() as c:
        c.execute("INSERT INTO chat_history (role, content) VALUES (%s, %s)", ('user', user_query))

    def generate():
        streaming_response = query_engine.query(user_query)
        text = ""
        for chunk in streaming_response.response_gen:
            text += chunk
            yield text
        with conn.cursor() as c:
            c.execute("INSERT INTO chat_history (role, content) VALUES (%s, %s)", ('assistant', text))

    return Response(generate(), content_type='text/event-stream')

@app.route('/history', methods=['GET'])
def history():
    with conn.cursor() as c:
        c.execute("SELECT role, content FROM chat_history")
        rows = c.fetchall()
    return jsonify([{'role': row[0], 'content': row[1]} for row in rows])

if __name__ == '__main__':
    load_llm()
    app.run(port=5000, debug=False)


