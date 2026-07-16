from langchain_openai import ChatOpenAI
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset

from src.utils import CustomEmbeddings, EMBED_MODEL

load_dotenv()


def get_evaluator_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")  

    if not api_key:
        st.error("خطا: OPENROUTER_API_KEY در فایل .env پیدا نشد.")
        st.stop()
    
    llm = ChatOpenAI(
        model="meta-llama/llama-3.1-70b-instruct",  
        openai_api_key=api_key,  
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_retries=3,
    )

    return LangchainLLMWrapper(llm)


def run_ragas_evaluation(df):
    """Converts dataframe to RAGAS dataset and runs evaluation."""

    
    contexts = []

    for _, row in df.iterrows():
        row_contexts = []

        for col in ["context_1", "context_2", "context_3"]:
            value = row.get(col)

            if pd.notna(value) and str(value).strip():
                row_contexts.append(str(value).strip())

        contexts.append(row_contexts)

    
    dataset = Dataset.from_dict(
        {
            "question": df["question"].fillna("").astype(str).tolist(),
            "answer": df["answer"].fillna("").astype(str).tolist(),
            "contexts": contexts,
        }
    )
    
    print(dataset[0])
    print(type(dataset[0]["contexts"]))

    evaluator_llm = get_evaluator_llm()

    embeddings = CustomEmbeddings(EMBED_MODEL)

    faithfulness.llm = evaluator_llm

    answer_relevancy.llm = evaluator_llm
    answer_relevancy.embeddings = embeddings

    try:
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
            ],
            raise_exceptions=True,
        )

        return result.to_pandas()

    except Exception as e:
        st.error(f"خطا در حین اجرای ارزیابی: {e}")
        return None



st.title("📊 پنل ارزیابی RAG با OpenRouter")

uploaded_file = st.file_uploader(
    "فایل CSV ارزیابی را آپلود کنید",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("پیش‌نمایش داده‌ها:")
    st.dataframe(df.head())

    st.write(f"تعداد رکوردها: {len(df)}")

    if st.button("شروع ارزیابی"):

        with st.spinner("در حال ارزیابی..."):

            results_df = run_ragas_evaluation(df)

        if results_df is not None:

            st.success("ارزیابی با موفقیت انجام شد!")

            st.dataframe(results_df)

            csv = results_df.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "دانلود نتایج",
                data=csv,
                file_name="ragas_results.csv",
                mime="text/csv",
            )