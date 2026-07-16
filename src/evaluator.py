import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper

from langchain_openai import ChatOpenAI  
from src.utils import CustomEmbeddings, EMBED_MODEL

load_dotenv()


def dataframe_to_ragas_dataset(df: pd.DataFrame) -> Dataset:
    data = {
        "question": df["question"].fillna("").astype(str).tolist(),
        "answer": df["answer"].fillna("").astype(str).tolist(),
        "contexts": df.apply(
            lambda row: [
                str(row["context_1"]) if pd.notna(row["context_1"]) else "",
                str(row["context_2"]) if pd.notna(row["context_2"]) else "",
                str(row["context_3"]) if pd.notna(row["context_3"]) else "",
            ],
            axis=1,
        ).tolist(),
    }
    return Dataset.from_dict(data)


def run_evaluation(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    dataset = dataframe_to_ragas_dataset(df)

    api_key = os.getenv("OPENROUTER_API_KEY")  
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set. Put it in .env and restart Streamlit.")

    base_llm = ChatOpenAI(
        model="meta-llama/llama-3.1-70b-instruct", 
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
    )
    evaluator_llm = LangchainLLMWrapper(base_llm)

    embeddings = CustomEmbeddings(EMBED_MODEL)

    faithfulness.llm = evaluator_llm
    answer_relevancy.llm = evaluator_llm
    answer_relevancy.embeddings = embeddings

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        raise_exceptions=False,
    )
    return result.to_pandas()