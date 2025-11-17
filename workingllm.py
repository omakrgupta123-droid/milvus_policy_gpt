import os
import time
import json
import boto3
import tempfile
import streamlit as st
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from pymilvus import connections, Collection
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================
load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION_US")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID_PolicyGPT")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY_PolicyGPT")

# Milvus Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
MILVUS_DB = os.getenv("MILVUS_DB")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# S3 Configuration
BUCKET_NAME = "edme-apps-data-dev"
PREFIX = "policygpt/"

# ============================================================
# AWS CLIENTS INITIALIZATION
# ============================================================
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

try:
    bucket_location = s3.get_bucket_location(Bucket=BUCKET_NAME)["LocationConstraint"]
    if bucket_location:
        AWS_REGION_NAME = bucket_location
except Exception:
    pass

textract = boto3.client(
    "textract",
    region_name=AWS_REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# MILVUS CONNECTION
# ============================================================
try:
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        user=MILVUS_USER,
        password=MILVUS_PASSWORD,
        db_name=MILVUS_DB
    )
    collection = Collection(MILVUS_COLLECTION)
    st.toast("✅ Connected to Milvus")
    time.sleep(1)
except Exception:
    st.toast("❌ Milvus Connection Failed", icon="⚠️")
    st.stop()

# ============================================================
# FETCH DOCUMENTS FROM MILVUS (FOR DROPDOWN)
# ============================================================
try:
    results = collection.query(
        expr="document_name != ''",
        output_fields=["document_name"],
        limit=10000
    )
    all_doc_names = sorted(list(set([r["document_name"] for r in results])))

    # Fallback using S3 path if Milvus query empty
    if not all_doc_names:
        s3_paths = [f"s3://{BUCKET_NAME}/{PREFIX}"]
        fallback_docs = collection.query(
            expr=f"s3_path in {s3_paths}",
            output_fields=["document_name"],
            limit=10000
        )
        all_doc_names = sorted(list(set([r["document_name"] for r in fallback_docs])))

except Exception as e:
    st.sidebar.warning(f"⚠️ Could not fetch document names: {e}")
    all_doc_names = []

# ============================================================
# STREAMLIT UI - SIDEBAR DOCUMENT DROPDOWN
# ============================================================
st.sidebar.subheader("📄 Select or Upload Document")

# ============================================================
# NEW FEATURE: COMPARISON MODE TOGGLE
# ============================================================
st.sidebar.divider()
comparison_mode = st.sidebar.checkbox("🔍 Enable Comparison Mode", value=False)

if comparison_mode:
    st.sidebar.info("Select multiple documents to compare")
    selected_docs = st.sidebar.multiselect(
        "Choose documents to compare:",
        all_doc_names,
        default=[]
    )
else:
    selected_doc = st.sidebar.selectbox(
        "Choose document to query:",
        ["All Documents"] + all_doc_names
    )

uploaded_file = st.sidebar.file_uploader("📤 Upload PDF", type=["pdf"])

# ============================================================
# CONVERSATION MEMORY MANAGEMENT
# ============================================================
st.sidebar.divider()
st.sidebar.subheader("💾 Conversation Memory")

# Display memory stats
if "chat_history" in st.session_state and st.session_state.chat_history:
    total_messages = len(st.session_state.chat_history)
    st.sidebar.info(f"📊 Total messages: {total_messages}")
    
    # Clear conversation button
    if st.sidebar.button("🗑️ Clear Conversation History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_all_textract_pages(job_id):
    all_blocks = []
    next_token = None
    while True:
        if next_token:
            response = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
        else:
            response = textract.get_document_analysis(JobId=job_id)
        all_blocks.extend(response.get("Blocks", []))
        next_token = response.get("NextToken", None)
        if not next_token:
            break
        time.sleep(1)
    return all_blocks


def extract_complete_text(all_blocks):
    page_numbers = sorted(set(b["Page"] for b in all_blocks if "Page" in b))
    lines_by_page, words_by_page, cell_text_by_page = {}, {}, {}

    for block in all_blocks:
        page = block.get("Page", 0)
        block_type = block.get("BlockType")
        text = block.get("Text", "").strip()
        if block_type == "LINE" and text:
            lines_by_page.setdefault(page, []).append({
                "text": text,
                "geometry": block.get("Geometry", {}),
                "id": block.get("Id")
            })
        elif block_type == "WORD" and text:
            words_by_page.setdefault(page, []).append({
                "text": text,
                "geometry": block.get("Geometry", {}),
                "id": block.get("Id")
            })
        elif block_type == "CELL" and text:
            cell_text_by_page.setdefault(page, []).append(text)

    full_text_parts = []
    for page in page_numbers:
        page_content = []
        if page in lines_by_page:
            lines = lines_by_page[page]
            lines.sort(key=lambda x: x["geometry"].get("BoundingBox", {}).get("Top", 0))
            page_content.extend([line["text"] for line in lines])
        elif page in words_by_page:
            words = words_by_page[page]
            words.sort(key=lambda x: (
                x["geometry"].get("BoundingBox", {}).get("Top", 0),
                x["geometry"].get("BoundingBox", {}).get("Left", 0)
            ))
            if words:
                current_line = [words[0]["text"]]
                current_top = words[0]["geometry"].get("BoundingBox", {}).get("Top", 0)
                for word in words[1:]:
                    word_top = word["geometry"].get("BoundingBox", {}).get("Top", 0)
                    if abs(word_top - current_top) < 0.01:
                        current_line.append(word["text"])
                    else:
                        page_content.append(" ".join(current_line))
                        current_line = [word["text"]]
                        current_top = word_top
                if current_line:
                    page_content.append(" ".join(current_line))
        if page in cell_text_by_page:
            page_content.extend(cell_text_by_page[page])
        if page_content:
            full_text_parts.append(f"--- PAGE {page} ---\n" + "\n".join(page_content))
    return "\n".join(full_text_parts), page_numbers


def generate_embedding(text):
    try:
        max_length = 8000
        if len(text) > max_length:
            text = text[:max_length]
        body = json.dumps({"inputText": text})
        response = bedrock_runtime.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response['body'].read())
        return response_body.get('embedding')
    except Exception:
        return None


def chunk_text(text, chunk_size=3000, overlap=300):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


def store_in_milvus(collection, embeddings_data):
    collection.insert(embeddings_data)
    collection.flush()


def build_conversation_context(chat_history, max_turns=5):
    """
    Build conversation context from chat history for LLM memory
    Limits to last max_turns exchanges to avoid token overflow
    """
    if not chat_history:
        return ""
    
    # Get last N exchanges (each exchange = user + assistant pair)
    recent_history = chat_history[-(max_turns * 2):]
    
    conversation_text = "Previous conversation:\n"
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_text += f"{role}: {msg['content']}\n"
    
    return conversation_text


def query_with_gpt4(user_query, collection, top_k=20):
    try:
        query_embedding = generate_embedding(user_query)
        if not query_embedding:
            return "Failed to generate embedding."

        # Apply document filter if one is selected
        search_expr = None
        if selected_doc != "All Documents":
            search_expr = f"document_name == '{selected_doc}'"

        search_params = {"metric_type": "L2", "params": {"nprobe": 35}}
        # search_params = {"metric_type": "COSINE", "params": {"nprobe": 50}}
        results = collection.search(
            data=[query_embedding],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=search_expr,
            output_fields=["text", "document_name", "page_number", "s3_path"]
        )

        context_parts = []
        for hits in results:
            for hit in hits:
                context_parts.append(
                    f"[Document: {hit.entity.get('document_name')}, Page: {hit.entity.get('page_number')}]\n{hit.entity.get('text')}"
                )

        if not context_parts:
            return "No relevant information found."

        context = "\n\n".join(context_parts)
        
        # Build conversation history context
        conversation_context = build_conversation_context(st.session_state.chat_history)
        
        system_prompt = """You are an AI assistant that answers based only on the provided document context. 
        You have access to previous conversation history to maintain context and provide coherent responses.
        Use the conversation history to understand follow-up questions and references to previous answers."""
        
        user_prompt = f"{conversation_context}\n\nDocument Context:\n{context}\n\nCurrent Question: {user_query}"

        # Build messages array with conversation history for better context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation history to messages (last 3 exchanges)
        recent_history = st.session_state.chat_history[-(6):]  # 3 exchanges = 6 messages
        for msg in recent_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current query with context
        messages.append({"role": "user", "content": user_prompt})

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error querying: {e}"


# ============================================================
# NEW FUNCTION: EXTRACT ANSWERS FROM EACH DOCUMENT INDIVIDUALLY
# ============================================================
def extract_answer_from_document(user_query, collection, doc_name, top_k=10):
    """
    Extract answer from a specific document and return structured result
    """
    try:
        query_embedding = generate_embedding(user_query)
        if not query_embedding:
            return None

        # Search only in the specific document
        search_expr = f"document_name == '{doc_name}'"
        search_params = {"metric_type": "L2", "params": {"nprobe": 35}}
        
        results = collection.search(
            data=[query_embedding],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=search_expr,
            output_fields=["text", "document_name", "page_number", "s3_path"]
        )

        # Collect context from this document
        context_parts = []
        page_numbers = set()
        
        for hits in results:
            for hit in hits:
                context_parts.append(hit.entity.get('text'))
                page_numbers.add(hit.entity.get('page_number'))

        if not context_parts:
            return None

        context = "\n\n".join(context_parts)
        
        # Ask GPT to extract specific answer from this document
        system_prompt = """You are an AI assistant that extracts specific information from document context.
        Extract the relevant answer to the user's question. Be concise and specific.
        Focus on extracting key values, numbers, dates, or specific details that directly answer the question.
        If the document contains the answer, provide it clearly. If not, respond with 'NOT FOUND'."""
        
        user_prompt = f"Document: {doc_name}\n\nContext:\n{context}\n\nQuestion: {user_query}\n\nExtract the specific answer:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,  # Lower temperature for more precise extraction
            max_tokens=500
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Return structured result if answer found
        if answer and answer.upper() != "NOT FOUND":
            return {
                "document_name": doc_name,
                "answer": answer,
                "pages": sorted(list(page_numbers)),
                "found": True
            }
        else:
            return None

    except Exception as e:
        return None


# ============================================================
# UPDATED FUNCTION: COMPARE MULTIPLE DOCUMENTS WITH INDIVIDUAL EXTRACTION
# ============================================================
def compare_documents(user_query, collection, doc_names, top_k=25):
    """
    Retrieves context from multiple specified documents and generates a comparison response.
    Now extracts answers from ALL documents that contain relevant information.
    """
    try:
        # First, extract answers from each document individually
        document_results = []
        
        with st.status("🔍 Searching across all selected documents...", expanded=True) as status:
            for idx, doc_name in enumerate(doc_names):
                st.write(f"📄 Analyzing: {doc_name}")
                result = extract_answer_from_document(user_query, collection, doc_name, top_k=15)
                if result:
                    document_results.append(result)
                time.sleep(0.1)  # Small delay for UI feedback
            
            status.update(label=f"✅ Found answers in {len(document_results)} out of {len(doc_names)} documents", state="complete")

        # If no results found in any document
        if not document_results:
            return "❌ No relevant information found in any of the selected documents."

        # Create a structured summary showing results from each document
        summary_parts = []
        summary_parts.append(f"## 📊 Results Summary")
        summary_parts.append(f"**Found in {len(document_results)} out of {len(doc_names)} documents**\n")
        
        # Display results in a structured format
        for idx, result in enumerate(document_results, 1):
            summary_parts.append(f"### {idx}. 📄 {result['document_name']}")
            summary_parts.append(f"**Pages:** {', '.join(map(str, result['pages']))}")
            summary_parts.append(f"**Answer:** {result['answer']}\n")
            summary_parts.append("---\n")

        # Now create comprehensive comparison context for detailed analysis
        query_embedding = generate_embedding(user_query)
        if not query_embedding:
            return "\n".join(summary_parts)

        doc_filter = " or ".join([f"document_name == '{doc}'" for doc in doc_names])
        search_expr = f"({doc_filter})"

        search_params = {"metric_type": "L2", "params": {"nprobe": 100}}
        results = collection.search(
            data=[query_embedding],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=search_expr,
            output_fields=["text", "document_name", "page_number", "s3_path"]
        )

        # Organize context by document
        doc_contexts = {doc: [] for doc in doc_names}
        for hits in results:
            for hit in hits:
                doc_name = hit.entity.get('document_name')
                if doc_name in doc_contexts:
                    doc_contexts[doc_name].append(
                        f"[Page: {hit.entity.get('page_number')}]\n{hit.entity.get('text')}"
                    )

        # Build context for comparison
        comparison_context = ""
        for doc_name, contexts in doc_contexts.items():
            if contexts:
                comparison_context += f"\n\n{'='*60}\nDOCUMENT: {doc_name}\n{'='*60}\n"
                comparison_context += "\n\n".join(contexts[:5])

        # Build conversation history context
        conversation_context = build_conversation_context(st.session_state.chat_history)

        system_prompt = """You are an AI assistant specialized in comparing multiple documents. 
    You have been provided with extracted answers from each document. Your task is to create a **comparative analysis** based on the user's query.

    **Key Instructions:**
    - **Adjust the level of detail** in your response according to the user’s request:
        1. If the user asks for a **detailed analysis**, provide a **comprehensive comparison** that:
            - Highlights the **key differences** in values/information across the documents.
            - Identifies any **similarities** or **patterns** that emerge.
            - Points out **unique aspects** of each document.
            - Provides **insights** into the variations and their possible impact.
            
        2. If the user asks for a **quick summary** or **overview** of the differences, provide a **concise comparison** that:
            - Summarizes the **major differences** in the documents.
            - Briefly touches on any **key trends** or **patterns**.
            - Highlights **unique findings** without going into excessive detail.

    **Example Queries:**
    - **Detailed Request:** “Can you give a thorough comparison of these answers and highlight any trends?” 
        - **Response:** A detailed, in-depth comparison with insights into trends, differences, and unique aspects.
    
    - **Concise Request:** “Can you just summarize the key differences between these answers?”
        - **Response:** A brief summary of the most important differences without extra details or trends.

    **Additional Notes:**
    - Always ensure the response **aligns with the user’s query**, whether they are looking for a **detailed analysis** or a **short summary**.
    - If the query is **open-ended**, assume a **concise summary** is preferred unless the user explicitly asks for more details.
    - If the query is **specific** or requests **insightful analysis**, provide a more **detailed response**.

    The extracted answers from each document have already been provided. Use this context to maintain consistency in your analysis, adapting your level of detail to the user's needs."""

        
        user_prompt = f"{conversation_context}\n\nQuery: {user_query}\n\n{comparison_context}\n\nProvide a detailed comparative analysis:"

        # Build messages array with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation history
        recent_history = st.session_state.chat_history[-(6):]
        for msg in recent_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current query
        messages.append({"role": "user", "content": user_prompt})

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        
        # Combine the structured summary with detailed analysis
        detailed_analysis = response.choices[0].message.content
        
        final_response = "\n".join(summary_parts) + "\n\n## 🔬 Detailed Comparative Analysis\n\n" + detailed_analysis
        
        return final_response

    except Exception as e:
        return f"Error comparing documents: {e}"


# ============================================================
# STREAMLIT MAIN APP
# ============================================================
st.title("📄 PolicyGPT - Document Query System")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================
# DOCUMENT UPLOAD AND PROCESSING
# ============================================================
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    s3_key = f"{PREFIX}{uploaded_file.name.replace(' ', '_')}"

    with st.spinner("Uploading PDF..."):
        s3.upload_file(temp_path, BUCKET_NAME, s3_key)
    st.sidebar.success("✅ Uploaded successfully")

    if st.button("Extract and Process Document", use_container_width=True):
        with st.spinner("Processing document..."):
            response = textract.start_document_analysis(
                DocumentLocation={"S3Object": {"Bucket": BUCKET_NAME, "Name": s3_key}},
                FeatureTypes=["TABLES", "FORMS"]
            )
            job_id = response["JobId"]

            while True:
                job_status = textract.get_document_analysis(JobId=job_id)
                status = job_status.get("JobStatus")
                if status == "SUCCEEDED":
                    break
                elif status == "FAILED":
                    st.error("Textract job failed.")
                    st.stop()
                time.sleep(5)

            all_blocks = get_all_textract_pages(job_id)
            full_text, page_numbers = extract_complete_text(all_blocks)
            page_texts = full_text.split("--- PAGE")

            all_chunks_data = []
            sequence_no, chunk_id = 1, 1

            for page_text in page_texts:
                if not page_text.strip():
                    continue
                try:
                    page_num_str = page_text.split("---")[0].strip()
                    page_number = int(page_num_str) if page_num_str.isdigit() else 1
                    page_content = page_text.split("---", 1)[-1].strip()
                except:
                    page_number = 1
                    page_content = page_text.strip()

                chunks = chunk_text(page_content)
                for chunk in chunks:
                    if chunk.strip():
                        all_chunks_data.append({
                            "text": chunk,
                            "page_number": page_number,
                            "chunk_id": chunk_id,
                            "sequenceno": sequence_no
                        })
                        chunk_id += 1
                        sequence_no += 1

            embeddings_list = []
            for chunk_data in all_chunks_data:
                embedding = generate_embedding(chunk_data["text"])
                if embedding:
                    embeddings_list.append({
                        "vector": embedding,
                        "text": chunk_data["text"],
                        "page_number": chunk_data["page_number"],
                        "chunk_id": chunk_data["chunk_id"],
                        "sequenceno": chunk_data["sequenceno"]
                    })

            vectors = [e["vector"] for e in embeddings_list]
            texts = [e["text"] for e in embeddings_list]
            page_nums = [e["page_number"] for e in embeddings_list]
            chunk_ids = [e["chunk_id"] for e in embeddings_list]
            seqs = [e["sequenceno"] for e in embeddings_list]
            s3_paths = [f"s3://{BUCKET_NAME}/{s3_key}"] * len(embeddings_list)
            doc_names = [uploaded_file.name] * len(embeddings_list)
            timestamps = [time.time()] * len(embeddings_list)

            milvus_data = [
                vectors, seqs, page_nums, chunk_ids, texts, s3_paths, doc_names, timestamps
            ]
            store_in_milvus(collection, milvus_data)

        st.success("✅ Document processed and stored successfully.")
        os.remove(temp_path)

# ============================================================
# CHAT INTERFACE
# ============================================================

# Display warning if in comparison mode with insufficient documents
if comparison_mode and len(selected_docs) < 2:
    st.warning("⚠️ Please select at least 2 documents to compare")

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Chat input at the bottom
user_query = st.chat_input("Ask a question about your documents...")
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    with st.spinner("Generating answer..."):
        if comparison_mode and len(selected_docs) >= 2:
            answer = compare_documents(user_query, collection, selected_docs)
        else:
            answer = query_with_gpt4(user_query, collection)
    
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()