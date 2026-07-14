"""
LangGraph orchestration: Extractor -> Researcher -> Evaluator -> END

This is a linear graph today, but it's built as a StateGraph (not a plain
function chain) so it's trivial to extend later — e.g. add a conditional
edge that loops back to the Researcher with broadened search terms if zero
trials are found, or a human-in-the-loop approval node before Evaluator.
"""
from langgraph.graph import StateGraph, END

from .schemas import AgentState
from .agents import extractor_agent, researcher_agent, evaluator_agent


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("extractor", extractor_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("evaluator", evaluator_agent)

    graph.set_entry_point("extractor")
    graph.add_edge("extractor", "researcher")
    graph.add_edge("researcher", "evaluator")
    graph.add_edge("evaluator", END)

    return graph.compile()


def run_pipeline(case_note: str) -> AgentState:
    """Convenience wrapper: run the full pipeline and return a clean AgentState."""
    app = build_graph()
    result = app.invoke(AgentState(case_note=case_note))
    if isinstance(result, AgentState):
        return result
    return AgentState(**result)


def stream_pipeline(case_note: str):
    """
    Run the pipeline and yield the cumulative AgentState after every node
    finishes (Extractor, then Researcher, then Evaluator). This lets a UI
    show live, step-by-step progress instead of a single blocking call.
    """
    app = build_graph()
    for state in app.stream(AgentState(case_note=case_note), stream_mode="values"):
        yield state if isinstance(state, AgentState) else AgentState(**state)
