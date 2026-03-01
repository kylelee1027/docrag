from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
import os
load_dotenv()


def build_chain():
    # Loading local embeddings from vectorstore/
    embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    vectorstore = FAISS.load_local(
        "vectorstore/",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(
        search_kwargs={'k': 4}
    )

    # Init LLM
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0.1,
        max_tokens=1024,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a clinical decision support assitant. Answer the question based ONLY on the provided context. If the context doesn't contain enough information, say so. Cite which source and page each claim comes from
        Context: {context}
        Question: {question}

        Answer:"""
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain


if __name__ == "__main__":
    chain = build_chain()
    response = chain.invoke({"query": "what are the signs of sepsis?"})

    print("Answer:", response["result"])
    print("\nSources:")
    for doc in response["source_documents"]:
        print(f"  - {doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}")