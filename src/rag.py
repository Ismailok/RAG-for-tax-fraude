import json

from time import time

from openai import OpenAI

import ingest


client = OpenAI()
index = ingest.load_index()


def search(query):
    boost = {
        'type_d_article': 0.22,
        'numéro_de_l_article_ou_de_la_loi': 0.499,
        'description_ou_texte_complet': 1.97,
        'mots_clés_ou_sujets_abordés': 2.60,
        'date_de_publication': 0.80,
        'source': 0.42
    }


    results = index.search(
        query=query, filter_dict={}, boost_dict=boost, num_results=10
    )

    return results


prompt_template = """
You are an expert assistant in French tax law. Answer the QUESTION using only the information provided in the CONTEXT, 
which is sourced from official legal texts and databases about tax fraud in France. Provide accurate, factual answers that comply with French law. 
If the QUESTION goes beyond the information available in the CONTEXT, state that you cannot answer without additional information.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

entry_template = """
type_d_article : {type_d_article}
numéro_de_l_article_ou_de_la_loi : {numéro_de_l_article_ou_de_la_loi}
description_ou_texte_complet : {description_ou_texte_complet}  
mots_clés_ou_sujets_abordés : {mots_clés_ou_sujets_abordés}
date_de_publication : {date_de_publication}
source : {source}
""".strip()


def build_prompt(query, search_results):
    context = ""

    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


def llm(prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    token_stats = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return answer, token_stats


evaluation_prompt_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


def evaluate_relevance(question, answer):
    prompt = evaluation_prompt_template.format(question=question, answer=answer)
    evaluation, tokens = llm(prompt, model="gpt-4o-mini")

    try:
        json_eval = json.loads(evaluation)
        return json_eval, tokens
    except json.JSONDecodeError:
        result = {"Relevance": "UNKNOWN", "Explanation": "Failed to parse evaluation"}
        return result, tokens


def calculate_openai_cost(model, tokens):
    openai_cost = 0

    if model == "gpt-4o-mini":
        openai_cost = (
            tokens["prompt_tokens"] * 0.00015 + tokens["completion_tokens"] * 0.0006
        ) / 1000
    else:
        print("Model not recognized. OpenAI cost calculation failed.")

    return openai_cost


def rag(query, model="gpt-4o-mini"):
    t0 = time()

    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer, token_stats = llm(prompt, model=model)

    relevance, rel_token_stats = evaluate_relevance(query, answer)

    t1 = time()
    took = t1 - t0

    openai_cost_rag = calculate_openai_cost(model, token_stats)
    openai_cost_eval = calculate_openai_cost(model, rel_token_stats)

    openai_cost = openai_cost_rag + openai_cost_eval

    answer_data = {
        "answer": answer,
        "model_used": model,
        "response_time": took,
        "relevance": relevance.get("Relevance", "UNKNOWN"),
        "relevance_explanation": relevance.get(
            "Explanation", "Failed to parse evaluation"
        ),
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "eval_prompt_tokens": rel_token_stats["prompt_tokens"],
        "eval_completion_tokens": rel_token_stats["completion_tokens"],
        "eval_total_tokens": rel_token_stats["total_tokens"],
        "openai_cost": openai_cost,
    }

    return answer_data
