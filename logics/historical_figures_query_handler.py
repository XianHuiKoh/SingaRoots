from dotenv import load_dotenv
import os
import streamlit as st
import json

from langchain_openai import ChatOpenAI
from langchain_classic.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMListwiseRerank
from helper_functions.llm import get_vectordb, FIGURE_INFO_DICT
from rapidfuzz import process


if "OPENAI_API_KEY" in os.environ or load_dotenv(".env"):
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
elif st.secrets.get("OPENAI_API_KEY"):
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
else:
    raise ValueError("OPENAI_API_KEY not found in environment or secrets.")


LLM_GPT = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed=42)
VECTORDB = get_vectordb(load_documents=False, is_figures=True)

CANONICAL_FIGURES = FIGURE_INFO_DICT.keys()

TOP_K_INITIAL_CANDIDATES = 75
TOP_K_RERANK = 20
TOP_K_ENTITY = 50
# The max number of chunks is 49 (Stamford Raffles)
TOP_K_INFOGRAPHIC = 50

try:
    RERANKER_COMPRESSOR = LLMListwiseRerank.from_llm(LLM_GPT, top_n=TOP_K_RERANK)
except ImportError as e:
    print(f"Reranker failed to load: {e}")
    RERANKER_COMPRESSOR = None


class EntityAnalysis(BaseModel):
    all_candidates: list[str] = Field(
        description="Canonical Singapore historical figures' names from query/context."
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
                Expert entity extraction for Singapore historical figures. Extract all canonical names, 
                correct any typos based on the canonical list, convert names to lowercase, 
                and return only the names of people. Do not include broad categories or non-person entities.
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


def fuzzy_match_entities(entities: list[str], threshold=85) -> list[str]:
    """
    Gets a list of matched entities using the list of proper full names
    :param entities: List of entities extracted
    :param threshold: Set to 85. Anything above or equal to 85 will be considered a match
    :return: List of names of entities matched
    """
    matched_entities = []
    for e in entities:
        match, score, *_ = process.extractOne(e, CANONICAL_FIGURES)
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

    canonical_list_str = ", ".join(CANONICAL_FIGURES)

    system_instruction = f"""
        You are an expert retrieval-augmented historian specializing in Singapore's historical figures.
        Your answer MUST be based **ONLY** on the provided context.

        ### MANDATORY FORMATTING AND CONSTRAINTS
        1. **Output Structure:** Your response must be composed of exactly two parts, presented consecutively with **NEWLINE**
        between them: 
        1) The synthesized Answer (cohesive paragraph format) and 
        2) The Source Aggregation. 
        **DO NOT ADD A SEPARATOR BETWEEN THE ANSWER AND SOURCE CITATION.**
        2. **Knowledge Gap Constraint:** If a fact required to answer the question is **not explicitly present** in the context, you must state: 
        "The sources do not contain sufficient information to answer this specific detail." 
        Do not guess or infer. 
        **If you use this statement, DO NOT include the second part of the output (the Source Aggregation).**
        3. **CRITICAL CONSTRAINT (Identification):** If the question asks to identify a historical figure, the answer **MUST** be a person from the following list: {canonical_list_str}. 
        Do not mention any figure not in this list.

        ### HISTORICAL ACCURACY INSTRUCTIONS
        * **Synthesize and Condense:** Provide a complete, concise, and direct synthesis of the relevant context. 
        **Start the synthesized Answer directly with the subject or the answer to the main verb of the question.**
        * **Prioritize Chronology:** For time-based questions, strictly synthesize the dates and events found within the context.
        * **Portfolio Precision & Tenure Mapping:** When citing appointments, use the **most precise and complete portfolio title**, and map it to its specific dates or date ranges (e.g., *Minister for Defence (1965–1967)*).
        * **Date Conflict Resolution:** If two retrieved documents offer conflicting information, **prioritize the chunk with the most specific date (DD-MMM-YYYY)**. If dates are equally specific, state the conflict clearly.

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
        You are an expert retrieval-augmented historian specializing in Singapore's historical figures.
        Your answer MUST be based **ONLY** on the provided context.

        ### MANDATORY FORMATTING AND CONSTRAINTS
        1. **Output Structure:** Your response must be composed of exactly two parts, presented consecutively with NEWLINE between them: 
        1) The synthesized Answer (cohesive paragraph format or clearly separated sections) and 
        2) The Source Aggregation. 
        **DO NOT ADD A SEPARATOR BETWEEN THE ANSWER AND SOURCE CITATION.**
        2. **Knowledge Gap Constraint:** If a fact required to answer the question is **not explicitly present** in the context, you must state: 
        "The sources do not contain sufficient information to answer this specific detail." 
        Do not guess or infer. **If you use this statement, DO NOT include the second part of the output (the Source Aggregation).**
        3. **Disambiguation Requirement:** For comparative or multi-figure questions (e.g., "compare Raffles and Farquhar"), 
        **structure the answer using distinct paragraphs or sections for each figure** to ensure clear separation and prevent confusion.

        ### HISTORICAL ACCURACY INSTRUCTIONS
        * **Synthesize and Condense:** Provide a complete, concise, and direct synthesis of the relevant context. 
        **Start the synthesized Answer directly with the subject or the answer to the main verb of the question.**
        * **Prioritize Chronology:** For time-based questions, strictly synthesize the dates and events found within the context.
        * **Portfolio Precision & Tenure Mapping:** When citing appointments, use the **most precise and complete portfolio title**, and map it to its specific dates or date ranges (e.g., *Minister for Defence (1965-1967)*).
        * **Date Conflict Resolution:** If two retrieved documents offer conflicting information, **prioritize the chunk with the most specific date (DD-MMM-YYYY)**. If dates are equally specific, state the conflict clearly.

        ### CITING INSTRUCTION
        1. Provide your complete, synthesized answer first. Do **NOT** embed any citations (e.g., [Source: X]) within the main body of the answer.
        2. The aggregated sources MUST be **FORMATTED EXACTLY** as: "[Sources: Source Name 1; Source Name 2]". 
        **You must only include unique source names and consolidate any duplicates (e.g., list 'Singapore Infopedia' only once).**
    """

    QA_PROMPT = PromptTemplate.from_template(
        """
        Answer for each historical figure separately if the question requires it.

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
        description="True if the query is about Singapore historical figures, False otherwise."
    )
    reasoning: str = Field(description="Brief explanation for the classification.")


def check_query_relevance(query: str) -> bool:
    """
    Checks if the query is relevant to Singapore historical figures
    :param query: Query from user
    :return: True = Query is relevant, False = Query is not relevant
    """
    router_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Analyze the user's query. Determine STRICTLY if the query is about one or more historical figures related to Singapore. 
                Focus on people, biography, roles, and contributions in the Singapore context. 
                If the query is too general (e.g., 'World War 2') or about non-human topics, it is NOT relevant.
                """,
            ),
            ("human", f"Query: {query}"),
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
            Sorry, I am an expert in Singapore historical figures and can only answer questions related to their biographies,\
            roles, and contributions within the Singapore context. Please ask a relevant question.
        """

    # High Recall
    base_retriever = VECTORDB.as_retriever(
        search_kwargs={"k": TOP_K_INITIAL_CANDIDATES}
    )

    #  Rerank documents
    if RERANKER_COMPRESSOR is None:
        # Fallback to simple retrieval if reranker failed to load
        reranked_chunks = base_retriever.invoke(query)[:TOP_K_RERANK]
    else:
        reranking_retriever = ContextualCompressionRetriever(
            base_compressor=RERANKER_COMPRESSOR,
            base_retriever=base_retriever,
        )
        reranked_chunks = reranking_retriever.invoke(query)

    # Extract figures from the reranked chunks (list of Document objects)
    entity_chain = create_entity_extraction_runnable()
    extracted = entity_chain.invoke({"query": query, "context": reranked_chunks})
    entities_raw = extracted.all_candidates

    # Fuzzy match to get the name of the figures as stored in the vectordb
    entities = fuzzy_match_entities([e.lower() for e in entities_raw])

    if entities:
        # Retrieve a large set filtered by the detected entities
        combined_retriever = VECTORDB.as_retriever(
            search_kwargs={"k": TOP_K_ENTITY, "filter": {"person": {"$in": entities}}}
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


class RelatedFigure(BaseModel):
    name: str = Field(description="The person's name")
    relationship: str = Field(description="The relationship between figure and person.")


class InfographicData(BaseModel):
    full_name: str = Field(description="The person's full name.")
    birth_date: str = Field(
        description="Date of birth in DD-MMM-YYYY, MMM-YYYY, or YYYY format."
    )
    death_date: str = Field(
        description="Date of death in DD-MMM-YYYY, MMM-YYYY, or YYYY format."
    )
    birth_place: str = Field(description="Place of birth.")
    death_place: str = Field(description="Place of death.")
    occupation_or_roles: str = Field(description="Main occupation or key roles held.")
    known_for: str = Field(
        description="A concise summary of what the person is primarily known for."
    )
    key_contributions: list[str] = Field(
        description="List of 3-5 major contributions or achievements."
    )
    historical_significance: str = Field(
        description="Summary of the person's historical impact on Singapore."
    )
    biography: str = Field(description="A short, factual biography (100-150 words).")
    related_figures: list[RelatedFigure] = Field(
        description="List of related figures, each with a 'name' and 'relationship'."
    )


def get_figure_infographics(figure_name: str) -> str:
    """
    Takes the figure name selected by the user on the frontend and gets the details of the
    figure for the infographic view.
    :param figure_name: String of figure name as shown on the frontend
    :return: A JSON string output
    """
    system_instruction = """
        You are a historical knowledge assistant creating structured information about Singapore historical figures.
        Your knowledge comes exclusively from the retrieved context (Singapore Infopedia and Roots.sg).
        You must never fabricate facts. Always provide neutral, factual, and concise answers.
    """

    INFOGRAPHIC_PROMPT = PromptTemplate.from_template(
        f"{system_instruction}\n\n"
        """
        ### Retrieved Context:
        {context}

        ### Target Figure:
        {question}

        ### Instructions:
        Using only the context provided, generate a structured summary of the person.
        For related persons, prioritise persons that are important to Singapore.
        If any field is missing, set its value to "Not available in sources."
        Dates should be in either DD-MMM-YYYY, MMM-YYYY, or YYYY.

        **CRITICAL: Output the raw JSON object that conforms to the schema. \
        Do not use any markdown formatting (e.g., ```json) around the output.**
        """
    )

    structured_llm = LLM_GPT.with_structured_output(
        schema=InfographicData, include_raw=False
    )

    figure_chunks = VECTORDB.as_retriever(
        search_kwargs={
            "k": TOP_K_INFOGRAPHIC,
            # Some names have a '.' in them.
            "filter": {"person": {"$in": [figure_name.lower().replace(".", "")]}},
        }
    ).invoke(figure_name)

    context_text = "\n\n---\n\n".join([c.page_content for c in figure_chunks])

    final_prompt = INFOGRAPHIC_PROMPT.format(context=context_text, question=figure_name)

    pydantic_output: InfographicData = structured_llm.invoke(final_prompt)

    return json.dumps(pydantic_output.model_dump(), indent=4)
