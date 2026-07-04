from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str
    answer: str

def chatbot(state: State):
    return {
        "answer": f"You asked: {state['question']}"
    }

graph_builder = StateGraph(State)

graph_builder.add_node(
    "chatbot",
    chatbot,
)

graph_builder.add_edge(
    START,
    "chatbot",
)

graph_builder.add_edge(
    "chatbot",
    END,
)

graph = graph_builder.compile()

result = graph.invoke({
    "question": "What is LangGraph?"
})

result = graph.invoke({
    "question": "Where is LangGraph used?"
})

print(result)