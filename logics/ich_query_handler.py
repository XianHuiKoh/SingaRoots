from dotenv import load_dotenv
import os
import streamlit as st
import json

from langchain_openai import ChatOpenAI
from langchain_classic.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMListwiseRerank
from helper_functions.llm import get_vectordb, ICH_INFO_DICT
from rapidfuzz import process

if "OPENAI_API_KEY" in os.environ or load_dotenv(".env"):
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
elif st.secrets.get("OPENAI_API_KEY"):
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
else:
    raise ValueError("OPENAI_API_KEY not found in environment or secrets.")


LLM_GPT = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed=42)
VECTORDB = get_vectordb(load_documents=False, is_figures=False)


CANONICAL_ELEMENTS = ICH_INFO_DICT.keys()

TOP_K_INITIAL_CANDIDATES = 75
TOP_K_RERANK = 20
TOP_K_ENTITY = 35
# Max num of chunks is 34 (Chingay)
TOP_K_INFOGRAPHIC = 35

try:
    RERANKER_COMPRESSOR = LLMListwiseRerank.from_llm(LLM_GPT, top_n=TOP_K_RERANK)
except ImportError as e:
    print(f"Reranker failed to load: {e}")
    RERANKER_COMPRESSOR = None


class EntityAnalysis(BaseModel):
    all_candidates: list[str] = Field(
        description="Canonical ICH Element names from query/context."
    )


def create_entity_extraction_runnable():
    """
    Creates entity chain to extract entities from query and context chunks
    """
    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Expert entity extraction for Singapore ICH. Extract canonical names ONLY, correct typos, lowercase, no broad categories.
                """,
            ),
            (
                "human",
                """
                Query: {query}
                
                Context: {context}
                """,
            ),
        ]
    )
    entity_extractor = LLM_GPT.with_structured_output(schema=EntityAnalysis)
    return extraction_prompt | entity_extractor


def fuzzy_match_entities(entities: list[str], threshold=80) -> list[str]:
    """
    Gets a list of matched entities using the list of full ICH element names
    :param entities: List of entities extracted
    :param threshold: Set to 85. Anything above or equal to 85 will be considered a match
    :return: List of names of entities matched
    """
    matched_entities = []
    for e in entities:
        match, score, *_ = process.extractOne(e, CANONICAL_ELEMENTS)
        if score >= threshold:
            matched_entities.append(match)
    return list(set(matched_entities))


def handle_final_llm_call_for_general_queries(query: str, context: str) -> str:
    """
    Triggers LLM call to answer general queries, i.e. queries with no entity matches found
    :param query: Query from user
    :param context: Context from reranked chunks
    :return: Response from LLM
    """

    system_instruction = """
        You are a retrieval-augmented assistant for Singapore ICH.
        Your answer MUST be based **ONLY** on the provided context.

        ### MANDATORY FORMATTING AND CONSTRAINTS
        1. **Output Structure:** Your response must be composed of exactly two parts, presented consecutively with **NEWLINE** between them: 
        1) The synthesized Answer (cohesive paragraph format) and 
        2) The Source Aggregation. 
        **DO NOT ADD A SEPARATOR BETWEEN THE ANSWER AND SOURCE CITATION.**
        2. **Knowledge Gap Constraint:** If a fact required to answer the question is **not explicitly present** in the context, you must state: 
        "The sources do not contain sufficient information to answer this specific detail." 
        Do not guess or infer. 
        **If you use this statement, DO NOT include the second part of the output (the Source Aggregation).**

        ### CITING INSTRUCTION
        1. Provide your complete, synthesized answer first. Do **NOT** embed any citations (e.g., [Source: X]) within the main body of the answer.
        2. The aggregated sources MUST be **FORMATTED EXACTLY** as: "[Sources: Source Name 1; Source Name 2]". 
        **You must only include unique source names and consolidate any duplicates (e.g., list 'Singapore Infopedia' only once).**
    """
    GENERAL_QA_PROMPT = PromptTemplate.from_template(
        """
        Answer the user's question based on context.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
    )

    response = LLM_GPT.invoke(
        [
            ("system", system_instruction),
            ("human", GENERAL_QA_PROMPT.format(context=context, question=query)),
        ]
    )
    return response.content


def handle_final_llm_call_with_disambiguation(
    query: str, context_text_final: str
) -> str:
    """
    Triggers LLM call to answer queries with entity matches found
    :param query: Query from user
    :param context: Context from reranked chunks + entity-filtered chunks
    :return: Response from LLM
    """

    system_instruction = """
        You are a retrieval-augmented assistant for Singapore ICH.
        Your answer MUST be based **ONLY** on the provided context.

        ### MANDATORY FORMATTING AND CONSTRAINTS
        1. **Output Structure:** Your response must be composed of exactly two parts, presented consecutively with **NEWLINE** between them: 
        1) The synthesized Answer (cohesive paragraph format) and 
        2) The Source Aggregation. 
        **DO NOT ADD A SEPARATOR BETWEEN THE ANSWER AND SOURCE CITATION.**
        2. **Knowledge Gap Constraint:** If a fact required to answer the question is **not explicitly present** in the context, you must state: 
        "The sources do not contain sufficient information to answer this specific detail." 
        Do not guess or infer. **If you use this statement, DO NOT include the second part of the output (the Source Aggregation).**

        ### CITING INSTRUCTION
        1. Provide your complete, synthesized answer first. Do **NOT** embed any citations (e.g., [Source: X]) within the main body of the answer.
        2. The aggregated sources MUST be **FORMATTED EXACTLY** as: "[Sources: Source Name 1; Source Name 2]". 
        **You must only include unique source names and consolidate any duplicates (e.g., list 'Singapore Infopedia' only once).**
    """

    QA_PROMPT = PromptTemplate.from_template(
        """
        Answer for each ICH element separately.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
    )

    response = LLM_GPT.invoke(
        [
            ("system", system_instruction),
            ("human", QA_PROMPT.format(context=context_text_final, question=query)),
        ]
    )
    return response.content


class QueryRelevance(BaseModel):
    is_relevant: bool = Field(
        description="True if the query is about Singapore's Intangible Cultural Heritage (ICH) elements, False otherwise."
    )
    reasoning: str = Field(description="Brief explanation for the classification.")


def check_query_relevance(query: str) -> bool:
    """
    Checks if a user query is relevant to Singapore's Intangible Cultural Heritage (ICH).
    :param query: Query from user
    :return: True = Query is relevant, False = Query is not relevant
    """
    router_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Analyze the user's query. Determine STRICTLY if the query is about one or more elements of 
                Singapore's Intangible Cultural Heritage (ICH), such as traditions, practices, foods, 
                arts, festivals, or social customs (e.g., hawker culture, Chingay, dikir barat, kebaya). 
                
                Focus on the nature, history, practice, or significance of the cultural element in the 
                Singapore context. 
                
                If the query is about specific people, general history, government, or non-ICH topics (e.g., 
                'What is the current GDP?', 'Who is the Prime Minister?'), it is NOT relevant.
                """,
            ),
            ("human", "Query: {query}"),
        ]
    )

    router_llm = LLM_GPT.with_structured_output(schema=QueryRelevance)
    router_chain = router_prompt | router_llm

    try:
        classification = router_chain.invoke({"query": query})
        return classification.is_relevant
    except Exception as e:
        print(e)
        # Failsafe - assume relevant if the router LLM fails to respond
        return True


def get_search_response(query: str) -> str:
    """
    Gets query from user and retrieves the necessary context chunks
    before getting response from LLM
    :param query: Query from user
    :return: Response from LLM
    """
    if not check_query_relevance(query):
        return """
            Sorry, I am an expert in Singapore's Intangible Cultural Heritage (ICH) elements and can only answer questions 
            related to their traditions, practices, foods, arts, and social customs within the Singapore context. 
            Please ask a relevant question.
        """

    # High Recall
    base_retriever = VECTORDB.as_retriever(
        search_kwargs={"k": TOP_K_INITIAL_CANDIDATES}
    )

    # Rerank documents
    if RERANKER_COMPRESSOR is None:
        # Fallback to simple retrieval if reranker failed to load
        reranked_chunks = base_retriever.invoke(query)[:TOP_K_RERANK]
    else:
        reranking_retriever = ContextualCompressionRetriever(
            base_compressor=RERANKER_COMPRESSOR,
            base_retriever=base_retriever,
        )
        reranked_chunks = reranking_retriever.invoke(query)

    # Extract entities from the reranked chunks (list of Document objects)
    entity_chain = create_entity_extraction_runnable()
    extracted = entity_chain.invoke({"query": query, "context": reranked_chunks})
    entities_raw = extracted.all_candidates

    # Fuzzy match to canonical names
    entities = fuzzy_match_entities([e.lower() for e in entities_raw])

    if entities:
        # Retrieve a large set filtered by the detected entities
        combined_retriever = VECTORDB.as_retriever(
            search_kwargs={"k": TOP_K_ENTITY, "filter": {"element": {"$in": entities}}}
        )
        entity_chunks = combined_retriever.invoke(query)

        all_chunks = reranked_chunks + entity_chunks
        unique_chunks = list(
            {chunk.page_content: chunk for chunk in all_chunks}.values()
        )

        context_text_final = "\n\n---\n\n".join([c.page_content for c in unique_chunks])
        return handle_final_llm_call_with_disambiguation(query, context_text_final)
    else:
        # Use the high-precision reranked chunks for the final answer
        context_chunks = reranked_chunks
        context_text_final = "\n\n---\n\n".join(
            [c.page_content for c in context_chunks]
        )
        return handle_final_llm_call_for_general_queries(query, context_text_final)


class InfographicData(BaseModel):
    ich_name: str = Field(
        description="The common name of the ICH element (e.g., 'Hawker Culture')."
    )
    ref_no: str = Field(
        description="The unique reference number or identifier, if available. Default: Not available in sources."
    )
    date_of_inclusion: str = Field(
        description="The date or year it was included into the Singapore's ICH inventory. Default: Not available in sources."
    )
    chinese_or_other_name: str = Field(
        description="Other names in different languages (e.g., Chinese, Tamil, Malay). Default: Not available in sources."
    )
    communities_involved: str = Field(
        description="The communities involved with the practice. Default: Not available in sources."
    )
    geographic_location: str = Field(
        description="Where the practice is mainly performed or associated (e.g., specific hawker centres, temples, regions). Default: Not available in sources."
    )
    origins_and_historical_evolution: str = Field(
        description="A concise summary of its origins and how it developed historically in Singapore (50-75 words). Default: Not available in sources."
    )
    description_of_practice: str = Field(
        description="A detailed description of the practice, custom, or art form itself (75-100 words). Default: Not available in sources."
    )
    associated_practices_and_rituals: list[str] = Field(
        description="A list of 3-5 related events, practices, or rituals. Use 'Not available in sources.' if no information is found."
    )
    transmission_and_safeguarding_efforts: list[str] = Field(
        description="A list of 3-5 key efforts, programs, or ways the heritage is passed down and protected. Use 'Not available in sources.' if no information is found."
    )
    current_significance_in_sg: str = Field(
        description="A summary of the current cultural or social importance of the ICH element in Singapore today (50-75 words). Default: Not available in sources."
    )


def get_element_infographics(element_name: str) -> str:
    """
    Takes the element name selected by the user on the frontend and gets the details of the
    element for the infographic view.
    :param element_name: String of element name as shown on the frontend
    :return: A JSON string output
    """
    system_instruction = """
        You are a cultural knowledge assistant creating a structured summary of a Singapore Intangible Cultural Heritage (ICH) element. 
        Your answer MUST be based **EXCLUSIVELY** on the provided context from Singapore Infopedia and Roots.sg. 
        **CRITICAL INSTRUCTION**: For any field where the information is not explicitly present or cannot be reliably inferred, 
        you MUST set the value to "Not available in sources." Do not fabricate or speculate.
    """

    INFOGRAPHIC_PROMPT = PromptTemplate.from_template(
        f"{system_instruction}\n\n"
        """
        ### Retrieved Context:
        {context}

        ### Target ICH Element:
        {question}

        Please extract the information requested in the schema using only the context provided.

        **CRITICAL: Output the raw JSON object that conforms to the schema. \
        Do not use any markdown formatting (e.g., ```json) around the output.**
        """
    )

    structured_llm = LLM_GPT.with_structured_output(
        schema=InfographicData, include_raw=False
    )

    element_chunks = VECTORDB.as_retriever(
        search_kwargs={
            "k": TOP_K_INFOGRAPHIC,
            "filter": {"element": {"$in": [element_name.lower()]}},
        }
    ).invoke(element_name)

    context_text = "\n\n---\n\n".join([c.page_content for c in element_chunks])

    final_prompt = INFOGRAPHIC_PROMPT.format(
        context=context_text, question=element_name
    )

    pydantic_output: InfographicData = structured_llm.invoke(final_prompt)

    return json.dumps(pydantic_output.model_dump(), indent=4)
