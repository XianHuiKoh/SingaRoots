import streamlit as st

from utility import check_password

st.set_page_config(page_title="SingaRoots's Methodology", layout="wide")

# Do not continue if check_password is not True.
if not check_password():
    st.stop()

st.title("🛠️ Methodology", anchor=False)
st.write(
    """
    The SingaRoots platform uses an Retrieval-Augmented Generation (RAG) architecture to process user queries and generate concise, \
    factual profiles. This methodology ensures that all responses are grounded in authoritative source material while remaining \
    highly relevant and easy to understand.
    """
)

st.markdown("---")

st.image(
    "flowchart/flowchart.drawio.png",
    caption="Flowchart on Retrieval-Augmented Generation (RAG) System Architecture",
)


st.header("🗂️ Data Preparation & Vector Database Setup", anchor=False)
st.markdown(
    """
    SingaRoots utilizes a vector database built on **Chroma** for persistent storage and retrieval. All documents are processed using the following models:
    
    * **Embedding Model:** **OpenAI `text-embedding-3-small`** is used to generate vector representations of all document chunks.
    * **Generation & Compression LLM:** **OpenAI `gpt-4o-mini`** is employed for document compression (reranking) and final answer/profile generation.
    
    The entire system is partitioned into two distinct Chroma collections: `singaroots_historical_figures` and `singaroots_ich`, ensuring that queries within one domain do not retrieve irrelevant content from the other.
    """
)

st.subheader("🧩 Vector Database Ingestion Process", anchor=False)
st.markdown(
    """
    The process of building the VectorDB is a critical, one-time operation that transforms raw source material into a queryable knowledge graph.

    1.  **Source Loading & Entity Identification:** Raw Markdown files from Infopedia and Roots.sg are loaded, and their entity type (Figure or ICH) is identified.
    2.  **Metadata Enrichment:** Pre-curated structured data (like dates, categories, and reference numbers) from static dictionaries are loaded and assigned to each article.
    3.  **Chunking (Two-Stage Pipeline):** This is a critical step ensuring both thematic integrity and manageable size:
        * **Stage 1 (Semantic Split):** A **`MarkdownHeaderTextSplitter`** divides the document based on its section headers (e.g., `# Introduction`, `## Early Life`), ensuring chunks are thematically coherent.
        * **Stage 2 (Size Control):** The resulting thematic chunks are then passed to a **`RecursiveCharacterTextSplitter`** to break down any overly long sections into smaller pieces, ensuring no single chunk exceeds the embedding model's capacity or dilutes context with excess length.
    4.  **Final Document Creation:** The system combines the chunk, its surrounding section headers, and all the pre-curated metadata into a final Document Object, ready for embedding.
    5.  **Vectorization & Persistence:** The Document Objects are vectorized using **`text-embedding-3-small`** and stored persistently in their respective **Chroma collections** (Figures or ICH).
    """
)

st.markdown("---")

st.header("🔬 RAG Implementation per Data Domain", anchor=False)
st.write(
    """
    The RAG architecture is applied separately to each data domain to support the two distinct retrieval modes.
    """
)

st.subheader("🗿 Historical Figures RAG Chain", anchor=False)
st.markdown(
    """
    This chain uses data from the `singaroots_historical_figures` collection and is optimized for biographical and historical context.
    
    #### 🔍 Retrieval Mode A: Question-Based Retrieval (Directed QA)

    This mode is designed for high-precision, low-latency answering of user-defined questions about historical figures.
    
    The chain follows a conditional, multi-step process:

    1.  **Initial Retrieval (Vector Search):** The **User Query** is embedded and used to retrieve the top **k=75** documents based on semantic similarity (High Recall).
    2.  **Contextual Compression (Reranking):** A **`ContextualCompressionRetriever`** utilizing **`LLMListwiseRerank`** (powered by `gpt-4o-mini`) evaluates the relevance of the retrieved chunks *against the original User Query*. It compresses the context down to the most highly relevant **k=20** chunks (High Precision), discarding noise.
    3.  **Entity Extraction:** The `gpt-4o-mini` LLM attempts to extract canonical entity names (Historical Figures) from the User Query.
    4.  **Conditional Retrieval & Generation:**
        * **If entities are found (Entity-Based Retrieval):** A second retrieval is performed using **metadata filtering** on the extracted entity names, retrieving the top **k=50** most relevant chunks for those figures. `gpt-4o-mini` synthesizes a concise, factual answer based on **all** relevant retrieved chunks (both reranked and entity-filtered) and the User Query.
        * **If no entities are found (General Query):** The system proceeds with the **k=20** reranked chunks from Step 2. `gpt-4o-mini` synthesizes a concise, factual answer based **only** on these reranked chunks and the User Query.
    """
)

st.markdown(
    """
    #### 📇 Retrieval Mode B: Exploratory Retrieval (Structured Profiling)

    This mode is optimized for comprehensive data extraction for a selected figure to generate an infographic-style profile.
    
    The chain follows a dedicated two-step process:

    1.  **Metadata-Filtered Retrieval (Comprehensive Recall):** The retriever is passed the **Entity Name** and uses **metadata filtering** (`person='[selected name]'`) to fetch all chunks associated with the selected figure. A maximum **k=50** (maximum number of chunks per figure is 49) is used. This ensures **Comprehensive Recall** of the entire document without the risk of an LLM reranker filtering out potentially important facts.
    2.  **Structured Generation (JSON Schema):** The entire retrieved context is passed to the `gpt-4o-mini` model, which is configured to output a structured JSON response conforming to the specific **Pydantic Schema for Historical Figures**.
        
        The schema dictates the extraction of data points such as:
        * Full name, Birth and death details, Occupation or roles
        * Key contributions, Historical significance
        * Related figures and their relationships
    """
)

st.subheader("🎭 ICH Elements RAG Chain", anchor=False)
st.markdown(
    """
    This chain uses data from the `singaroots_ich` collection and is optimized for cultural practices, origins, and transmission efforts.

    #### 🔍 Retrieval Mode A: Question-Based Retrieval (Directed QA)

    This mode uses an adapted high-precision, conditional retrieval chain tuned for ICH queries:
    
    1.  **Initial Retrieval (Vector Search):** The **User Query** is embedded and used to retrieve the top **k=75** documents based on semantic similarity (High Recall).
    2.  **Contextual Compression (Reranking):** `LLMListwiseRerank` evaluates the relevance of the retrieved chunks *against the original User Query* and compresses the context down to the most relevant **k=20** chunks.
    3.  **Entity Extraction:** The `gpt-4o-mini` LLM attempts to extract canonical entity names (ICH Elements) from the User Query.
    4.  **Conditional Retrieval & Generation:**
        * **If entities are found (Entity-Based Retrieval):** A second retrieval is performed using **metadata filtering** on the extracted entity names, retrieving the top **k=35** most relevant chunks for those elements. `gpt-4o-mini` synthesizes a concise, factual answer based on **all** relevant retrieved chunks (both reranked and entity-filtered) and the User Query.
        * **If no entities are found (General Query):** The system proceeds with the **k=20** reranked chunks from Step 2. `gpt-4o-mini` synthesizes a concise, factual answer based **only** on these reranked chunks and the User Query.
    """
)

st.markdown(
    """
    #### 📇 Retrieval Mode B: Exploratory Retrieval (Structured Profiling)

    This mode uses the same process as the Figures mode but utilizes a different Pydantic schema to capture cultural data points.

    1.  **Metadata-Filtered Retrieval (Comprehensive Recall):** The retriever is passed the **Entity Name** and uses **metadata filtering** (`element='[selected element]'`) to fetch all chunks associated with the selected ICH element. A maximum **k=35** (maximum number of chunks per ICH element is 34) is used. This ensures **Comprehensive Recall** of the document.
    2.  **Structured Generation (JSON Schema):** The entire retrieved context is passed to the `gpt-4o-mini` model, which is configured to output a structured JSON response conforming to the specific **Pydantic Schema for ICH Elements**.

        The schema dictates the extraction of data points such as:
        * Element name, Reference number, Date of inclusion
        * Origins and historical evolution, Description of element
        * Associated practices or rituals
        * Transmission and safeguarding efforts
    """
)
