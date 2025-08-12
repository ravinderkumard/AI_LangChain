import pandas as pd
import random
import json
import os
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda,RunnableBranch
from langchain_community.document_loaders import UnstructuredExcelLoader
from dotenv import load_dotenv

load_dotenv()

EXCEL_FILE = "Copy of 100DayCodingChallenge.xlsx"
SHEET_NAME = "Ravinder QuestionSheet"
USED_QUESTIONS_FILE = "used_questions.json"  # persistence file

MODEL_NAME = "gpt-4"  # or "gpt-3.5-turbo"

# Load previously used questions
def load_used_questions():
    if os.path.exists(USED_QUESTIONS_FILE):
        with open(USED_QUESTIONS_FILE, "r") as f:
            return json.load(f)
    return {}

# Save used questions before exit
def save_used_questions(data):
    with open(USED_QUESTIONS_FILE, "w") as f:
        json.dump(data, f)

used_questions_by_topic = load_used_questions()

# 1. Load Excel into LangChain Documents
def load_excel(file_path):
    df = pd.read_excel(file_path, sheet_name=SHEET_NAME, skiprows=20)
    docs = []
    for idx, row in df.iterrows():
        unique_id = f"{row['Ques No.']}"  # Guarantees uniqueness
        content = (
            f"Question No: {row['Ques No.']}\n"
            f"Question: {row['Question']}\n"
            f"Topic: {row['Topic']}\n"
            f"Status: {row['Tag']}\n"
            f"Notes: {row['Question Link']}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "id": unique_id,
                "topic": row["Topic"],
                "tag": row["Tag"]
            }
        ))
    return docs



# 2. Filter Runnable
# def filter_docs(docs, done_only=False, topic=None):
#     filtered = docs
#     if done_only:
#         filtered = [d for d in filtered if is_done(d.metadata["tag"])]
#     if topic:
#         topic_lower = topic.strip().lower()
#         filtered = [
#             d for d in filtered
#             if topic_lower in str(d.metadata["topic"]).strip().lower()
#         ]
#     return filtered

def filter_docs(docs, done_only=False, topic=None):
    filtered = docs
    if done_only:
        filtered = [d for d in filtered if is_done(d.metadata["tag"])]

    if topic and topic.lower() != "all":
        # Get list of unique topics from docs
        all_topics = list(set(d.metadata["topic"] for d in docs if pd.notna(d.metadata["topic"])))
        print(f"[DEBUG] Available topics: {all_topics}")
        # Use LLM to find related topics
        topic_prompt = PromptTemplate.from_template("""
        You are an assistant for categorizing DSA questions.
        The user has entered the topic: "{user_topic}".
        Here is the list of available topics: {topics}.
        Return only the topics from the list that are most relevant to the user topic.
        Respond with a comma-separated list of matching topics and nothing else.
        """)
        chain = topic_prompt | llm
        result = chain.invoke({"user_topic": topic, "topics": ", ".join(all_topics)}).content

        matching_topics = [t.strip() for t in result.split(",") if t.strip()]
        print(f"[DEBUG] LLM matched topics: {matching_topics}")

        filtered = [d for d in filtered if d.metadata["topic"] in matching_topics]

    return filtered

filter_runnable = RunnableLambda(lambda x: filter_docs(**x))

# 3. LLM setup
llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)

# 4. Actions
def summarize_questions(docs):
    summary_prompt = PromptTemplate.from_template("""
    You are a DSA reviewer. Summarize the following solved questions in a concise, easy-to-review list:
    {questions}
    """)
    chain = summary_prompt | llm
    result = chain.invoke({"questions": "\n\n".join(d.page_content for d in docs)})
    return result.content

def is_done(tag_value):
    return str(tag_value).strip().lower() in ["easy", "medium", "hard"]

def quiz_user(docs, num_questions=3):
    random_docs = random.sample(docs, min(num_questions, len(docs)))
    quiz_prompt = PromptTemplate.from_template("""
    You are a DSA quizmaster. Based on the following solved problems, give me {num_questions} question for revision from the list of questions, don't ask me question just give me questions in a list format:
    {questions}
    """)
    chain = quiz_prompt | llm
    result = chain.invoke({
        "num_questions": num_questions,
        "questions": "\n\n".join(d.page_content for d in random_docs)
    })
    return result.content


def search_by_topic(docs, topic, num_questions=5):
    # Step 1: Get all unique topics from docs
    all_topics = list(set(str(d.metadata["topic"]) for d in docs if pd.notna(d.metadata["topic"])))

    # Step 2: Use LLM to find the best matching topics
    if topic.lower() != "all":
        topic_prompt = PromptTemplate.from_template("""
        You are an assistant for selecting DSA questions.
        The user is interested in the topic: "{user_topic}".
        From the available topics: {topics}
        Return only the topics that match or are most relevant to the user topic.
        Output a comma-separated list of matching topics and nothing else.
        """)
        chain = topic_prompt | llm
        result = chain.invoke({
            "user_topic": topic,
            "topics": ", ".join(all_topics)
        }).content
        matching_topics = [t.strip() for t in result.split(",") if t.strip()]
        print(f"[DEBUG] LLM matched topics: {matching_topics}")

        topic_docs = [d for d in docs if str(d.metadata["topic"]) in matching_topics]
    else:
        topic_docs = docs

    if not topic_docs:
        return "No questions found for this topic."

    print("Search by Topic:", len(topic_docs))
    topic_key = topic.lower()

    # Step 3: Track used questions
    if topic_key not in used_questions_by_topic:
        used_questions_by_topic[topic_key] = set()
    else:
        used_questions_by_topic[topic_key] = set(used_questions_by_topic[topic_key])

    used_ids = used_questions_by_topic[topic_key]
    all_ids = [d.metadata["id"] for d in topic_docs]

    print(f"[DEBUG] Topic: {topic_key} | Total questions: {len(all_ids)} | Already used: {len(used_ids)}")

    available_ids = [qid for qid in all_ids if qid not in used_ids]

    if len(available_ids) < num_questions:
        print(f"[DEBUG] Resetting used IDs for topic '{topic_key}'")
        used_ids.clear()
        available_ids = all_ids

    chosen_ids = random.sample(available_ids, min(num_questions, len(available_ids)))
    used_ids.update(chosen_ids)

    chosen_docs = [d for d in topic_docs if d.metadata["id"] in chosen_ids]

    # Step 4: Return only the chosen questions
    return "\n".join(f"{i+1}. {d.page_content}" for i, d in enumerate(chosen_docs))

summarize_chain = RunnableLambda(summarize_questions)
quiz_chain = RunnableLambda(quiz_user)
search_chain = RunnableLambda(search_by_topic)
exit_chain = RunnableLambda(lambda _:"Exiting the program...")

menu_branch = RunnableBranch(
    (lambda data: data["choice"] == "1", summarize_chain),
    (lambda data: data["choice"] == "2", quiz_chain),
    (lambda data: data["choice"] == "3", search_chain),
    (lambda data: data["choice"] == "4", exit_chain),
    # Default if invalid choice
    RunnableLambda(lambda _: "Invalid choice. Please try again.")
)

def main():
    docs = load_excel(EXCEL_FILE)
    print(f"Loaded {len(docs)} DSA questions from sheet '{SHEET_NAME}'")
    
    while True:
        print("\n ----- DSA Question Loader -----  ")
        print("1. Summarize solved questions")
        print("2. Quiz me on solved questions")
        print("3. Search by topic")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")
        if choice == '1':
            solved_docs = filter_docs(docs, done_only=True)
            print(summarize_questions(solved_docs))
        elif choice == '2':
            solved_docs = filter_docs(docs, done_only=True)
            print(quiz_user(solved_docs))
        elif choice == '3':
            topic = input("Enter topic to search (or 'all' for all topics): ").strip()
            print(search_by_topic(docs, topic))
        elif choice == '4':
            save_used_questions({k: list(v) for k, v in used_questions_by_topic.items()})
            print("Progress saved. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
