import json
import sys
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from worker import PubSubListener


class FakeMessage:
    def __init__(self, payload, attributes=None):
        self.data = json.dumps(payload).encode("utf-8")
        self.attributes = attributes or {}
        self.message_id = "worker-contract-message"
        self.acked = False
        self.nacked = False

    def ack(self):
        self.acked = True

    def nack(self):
        self.nacked = True


class StubListener(PubSubListener):
    child_exit_code: int = 0
    captured_event: dict[str, str] | None = None

    def _run_child(self, event):
        self.captured_event = event
        return self.child_exit_code


def make_listener(exit_code=0):
    return StubListener(
        project_id="rag-project",
        bucket="rag-output-extractor-",
        worker_stage="chunker",
        source_bucket_env="INPUT_BUCKET_NAME",
        worker_subscription="rag-chunker-workers",
        worker_child_command="main.py",
        child_exit_code=exit_code,
    )


class WorkerContractTest(unittest.TestCase):
    def test_successful_child_acknowledges_message(self):
        listener = make_listener()
        message = FakeMessage(
            {
                "bucket": "rag-output-extractor-vip-customer",
                "name": "folder/document.pdf",
            },
            {"eventType": "OBJECT_FINALIZE"},
        )

        listener.callback(message)

        self.assertTrue(message.acked)
        self.assertFalse(message.nacked)
        self.assertEqual(
            listener.captured_event,
            {
                "bucket": "rag-output-extractor-vip-customer",
                "file_name": "folder/document.pdf",
                "project_name": "vip-customer",
            },
        )
        self.assertEqual(listener.active_messages, 0)

    def test_failed_child_negatively_acknowledges_message(self):
        listener = make_listener(exit_code=42)
        message = FakeMessage(
            {"bucket": "rag-output-extractor-tenant-a", "name": "document.pdf"}
        )

        listener.callback(message)

        self.assertTrue(message.nacked)
        self.assertFalse(message.acked)
        self.assertEqual(listener.active_messages, 0)

    def test_invalid_message_is_acknowledged_without_processing(self):
        listener = make_listener()
        message = FakeMessage({"bucket": "unrelated-bucket", "name": "document.pdf"})

        listener.callback(message)

        self.assertTrue(message.acked)
        self.assertFalse(message.nacked)
        self.assertIsNone(listener.captured_event)

    @patch("worker.subprocess.run")
    def test_child_receives_message_context(self, run):
        run.return_value.returncode = 0
        listener = make_listener()
        event = {
            "bucket": "rag-output-extractor-tenant-a",
            "file_name": "folder/document.pdf",
            "project_name": "tenant-a",
        }

        self.assertEqual(PubSubListener._run_child(listener, event), 0)

        self.assertEqual(run.call_args.args[0], [sys.executable, "main.py"])
        child_env = run.call_args.kwargs["env"]
        self.assertEqual(child_env["WORKER_MODE"], "false")
        self.assertEqual(child_env["FILE_NAME"], event["file_name"])
        self.assertEqual(child_env["PROJECT_NAME"], event["project_name"])
        self.assertEqual(child_env["INPUT_BUCKET_NAME"], event["bucket"])

    def test_worker_configuration_is_validated(self):
        base = make_listener().model_dump()

        for override in (
            {"worker_stage": "unknown"},
            {"source_bucket_env": "source-bucket"},
            {"worker_max_messages": 0},
            {"worker_max_lease_seconds": 0},
            {"worker_shutdown_grace_seconds": -1},
        ):
            with self.subTest(override=override), self.assertRaises(ValidationError):
                StubListener(**{**base, **override})

        self.assertEqual(make_listener().worker_max_lease_seconds, 43200)


if __name__ == "__main__":
    unittest.main()
