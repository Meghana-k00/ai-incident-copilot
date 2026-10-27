
from langgraph.checkpoint.sqlite import SqliteSaver

from src.approval_workflow import (
    build_approval_graph,
    resume_approval,
)


def test_workflow_pauses_and_resumes_after_approval():
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = build_approval_graph(checkpointer)
        config = {
            "configurable": {
                "thread_id": "approval-test-001"
            }
        }

        initial_result = graph.invoke(
            {
                "approval_id": 123,
                "proposed_action": "Restart the test service",
                "decision": "",
            },
            config=config,
        )

        # The workflow must pause for human input.
        assert "__interrupt__" in initial_result
        interruption = initial_result["__interrupt__"][0]
        assert interruption.value["approval_id"] == 123
        assert (
            interruption.value["proposed_action"]
            == "Restart the test service"
        )

        # Resume the same saved workflow with a decision.
        final_result = resume_approval(
            graph=graph,
            thread_id="approval-test-001",
            decision="approved",
        )

        assert final_result["decision"] == "approved"


def test_workflow_rejects_invalid_resumed_decision():
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = build_approval_graph(checkpointer)
        config = {
            "configurable": {
                "thread_id": "approval-test-002"
            }
        }

        graph.invoke(
            {
                "approval_id": 124,
                "proposed_action": "Restart the test service",
                "decision": "",
            },
            config=config,
        )

        try:
            resume_approval(
                graph=graph,
                thread_id="approval-test-002",
                decision="maybe",
            )
        except ValueError as exc:
            assert "Invalid approval decision" in str(exc)
        else:
            raise AssertionError(
                "Invalid approval decision was accepted."
            )

def test_paused_workflow_resumes_after_checkpoint_reopens(tmp_path):
    database_path = tmp_path / "workflow_checkpoints.sqlite"
    thread_id = "approval-persistence-001"
    config = {"configurable": {"thread_id": thread_id}}

    # First connection: start the workflow and pause it.
    with SqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        checkpointer.setup()
        graph = build_approval_graph(checkpointer)

        result = graph.invoke(
            {
                "approval_id": 125,
                "proposed_action": "Restart the test service",
                "decision": "",
            },
            config=config,
        )

        assert "__interrupt__" in result

    # Second connection: reopen the same file and resume.
    with SqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        checkpointer.setup()
        graph = build_approval_graph(checkpointer)

        result = resume_approval(
            graph=graph,
            thread_id=thread_id,
            decision="approved",
        )

        assert result["decision"] == "approved"

def test_completed_workflow_cannot_be_resumed_again():
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = build_approval_graph(checkpointer)
        thread_id = "approval-repeat-001"
        config = {"configurable": {"thread_id": thread_id}}

        graph.invoke(
            {
                "approval_id": 126,
                "proposed_action": "Restart the test service",
                "decision": "",
            },
            config=config,
        )

        first_result = resume_approval(
            graph=graph,
            thread_id=thread_id,
            decision="approved",
        )
        assert first_result["decision"] == "approved"

        try:
            second_result = resume_approval(
                graph=graph,
                thread_id=thread_id,
                decision="rejected",
            )
        except (ValueError, RuntimeError):
            return

        assert second_result.get("decision") != "rejected", (
            "A completed approval workflow accepted a second decision."
        )


def test_completed_approval_decision_reads_real_checkpoint():
    from src.approval_workflow import get_completed_approval_decision

    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = build_approval_graph(checkpointer)
        thread_id = "completed-checkpoint-test"
        config = {"configurable": {"thread_id": thread_id}}

        graph.invoke(
            {
                "approval_id": 501,
                "proposed_action": "Restart the test service",
                "decision": "",
            },
            config=config,
        )

        resume_approval(graph, thread_id, "approved")

        result = get_completed_approval_decision(graph, thread_id)

        assert result == "approved"
