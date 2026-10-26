
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict):
    approval_id: int
    proposed_action: str
    decision: str


def wait_for_human_approval(state: ApprovalState) -> dict:
    """Pause until a reviewer provides an approval decision."""
    decision = interrupt(
        {
            "approval_id": state["approval_id"],
            "proposed_action": state["proposed_action"],
            "message": "A human reviewer must decide before proceeding.",
        }
    )

    if decision not in {"approved", "rejected", "changes_requested"}:
        raise ValueError("Invalid approval decision.")

    return {"decision": decision}


def build_approval_graph(checkpointer: SqliteSaver):
    workflow = StateGraph(ApprovalState)

    workflow.add_node("wait_for_approval", wait_for_human_approval)
    workflow.add_edge(START, "wait_for_approval")
    workflow.add_edge("wait_for_approval", END)

    return workflow.compile(checkpointer=checkpointer)


def resume_approval(
    graph,
    thread_id: str,
    decision: str,
):
    """Resume the paused graph with a human decision."""
    return graph.invoke(
        Command(resume=decision),
        config={"configurable": {"thread_id": thread_id}},
    )
