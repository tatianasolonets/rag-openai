import asyncio
import os
import pandas as pd
from llama_index.evaluation import generate_question_context_pairs, RetrieverEvaluator, FaithfulnessEvaluator, RelevancyEvaluator, BatchEvalRunner
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI

# Add your API key here
os.environ['OPENAI_API_KEY'] = 'xxxx'
# Use gpt-4 model
llm = OpenAI(model="gpt-4")
# Read the documents for RAG
documents = SimpleDirectoryReader("./data/").load_data()

# chunk_size = 512
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)
vector_index = VectorStoreIndex(nodes)
query_engine = vector_index.as_query_engine()

response_vector = query_engine.query("What is the best way to optimize LLM?")
print("What is the best way to optimize LLM?")
print(response_vector.response)

# Evaluate Retrieval
qa_dataset = generate_question_context_pairs(
    nodes,
    llm=llm,
    num_questions_per_chunk=2
)

retriever = vector_index.as_retriever(similarity_top_k=2)

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)

async def evaluate_dataset():
    return await retriever_evaluator.aevaluate_dataset(qa_dataset)

def display_results(name, eval_results):
    """Display results from evaluate."""
    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    metric_df = pd.DataFrame(
        {"Retriever Name": [name], "Hit Rate": [hit_rate], "MRR": [mrr]}
    )

    return metric_df


eval_results = asyncio.run(evaluate_dataset())

print(display_results("OpenAI Embedding Retriever", eval_results))

# Evaluate Response
queries = list(qa_dataset.queries.values())  
gpt4 = OpenAI(temperature=0, model="gpt-4")
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)
vector_index = VectorStoreIndex(nodes, service_context = service_context_gpt4)
query_engine = vector_index.as_query_engine()

from llama_index.evaluation import FaithfulnessEvaluator
faithfulness_gpt4 = FaithfulnessEvaluator(service_context=service_context_gpt4)

eval_query = queries[5]
response_vector = query_engine.query(eval_query)

# Faithfulness evaluation
eval_result = faithfulness_gpt4.evaluate_response(response=response_vector)
print("Is Faithfulness evaluation passing?")
print(eval_result.passing)

from llama_index.evaluation import RelevancyEvaluator

# Relevancy evaluation
relevancy_gpt4 = RelevancyEvaluator(service_context=service_context_gpt4)
eval_result = relevancy_gpt4.evaluate_response(
    query=eval_query, response=response_vector
)
print("Is Relevancy evaluation passing?")
print(eval_result.passing)
