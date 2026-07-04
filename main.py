from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

memory = MemorySaver()

router_prompt = ChatPromptTemplate.from_template("""
You are a routing assistant.

If the user is asking for a mathematical calculation,
return only:

calculator

Otherwise return only:

chatbot

Question:
{question}
""")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    operation: str

def chatbot(state: State):
    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


def calculator(state: State):
    return {
        "messages": [
            AIMessage(
                content="Calculator was selected."
            )
        ]
    }

def router(state: State):

    question = state["messages"][-1].content

    prompt = router_prompt.invoke({
        "question": question
    })

    response = llm.invoke(prompt)

    decision = response.content.strip().lower()

    return decision
    

graph_builder = StateGraph(State)

graph_builder.add_node(
    "chatbot",
    chatbot,
)

graph_builder.add_edge(
    START,
    router,
)

graph_builder.add_edge(
    "chatbot",
    END
)

graph_builder.add_edge(
    "calculator",
    END
)

graph_builder.add_node(
    "calculator",
    calculator
)

graph = graph_builder.compile(
    checkpointer=memory
)

config = {
    "configurable": {
        "thread_id": "1"
    }
}

while True:
    question = input("User >> ")

    if question in ["quit", "exit"]:
        print("Thanks for using my chatbot")
        break

    result = graph.invoke({
        "messages": [
            HumanMessage(content=question)
        ]
    },
    config=config,)

    print(f"AI >> {result["messages"][-1].content}")