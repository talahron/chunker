import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import CancelledError
from typing import Any, ClassVar

from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.subscriber.message import Message
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    return logger


class PubSubListener(BaseModel):
    """Pull GCS finalize messages and run one stage process per message."""

    supported_stages: ClassVar[frozenset[str]] = frozenset(
        {"convertor", "flipper", "extractor", "chunker"}
    )

    project_id: str = Field(min_length=1, description="GCP project ID")
    bucket: str = Field(min_length=1, description="source bucket prefix")
    worker_stage: str = Field(min_length=1, description="pipeline stage")
    source_bucket_env: str = Field(
        min_length=1,
        description="child-process environment variable for the source bucket",
    )
    worker_subscription: str = Field(min_length=1, description="pull subscription")
    worker_child_command: str = Field(min_length=1, description="stage command")
    worker_max_messages: int = Field(default=1, ge=1)
    worker_max_lease_seconds: int = Field(default=43200, ge=1)
    worker_shutdown_grace_seconds: int = Field(default=900, ge=0)

    model_config = ConfigDict(str_strip_whitespace=True)

    _logger: logging.Logger = PrivateAttr(default_factory=lambda: setup_logger("PubSubListener"))
    _subscriber: pubsub_v1.SubscriberClient | None = PrivateAttr(default=None)
    _subscription_name: str = PrivateAttr(default="")
    _streaming_pull_future: Any = PrivateAttr(default=None)
    _stopping: threading.Event = PrivateAttr(default_factory=threading.Event)
    _condition: threading.Condition = PrivateAttr(default_factory=threading.Condition)
    _active_messages: int = PrivateAttr(default=0)

    @field_validator("worker_stage")
    @classmethod
    def validate_worker_stage(cls, value: str) -> str:
        stage = value.lower()
        if stage not in cls.supported_stages:
            valid = ", ".join(sorted(cls.supported_stages))
            raise ValueError(f"worker_stage must be one of: {valid}")
        return stage

    @field_validator("source_bucket_env")
    @classmethod
    def validate_source_bucket_env(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", value):
            raise ValueError("source_bucket_env must be an uppercase environment variable name")
        return value

    @property
    def active_messages(self) -> int:
        with self._condition:
            return self._active_messages

    def setup(self):
        if self._subscriber is not None:
            raise RuntimeError("Pub/Sub listener is already initialized")

        self._subscriber = pubsub_v1.SubscriberClient()
        self._subscription_name = self._subscription_path()
        self._streaming_pull_future = self._subscriber.subscribe(
            self._subscription_name,
            callback=self.callback,
            flow_control=pubsub_v1.types.FlowControl(
                max_messages=self.worker_max_messages,
                max_lease_duration=self.worker_max_lease_seconds,
            ),
        )
        return self

    def _subscription_path(self) -> str:
        if self.worker_subscription.startswith("projects/"):
            return self.worker_subscription
        return (
            f"projects/{self.project_id}/subscriptions/"
            f"{self.worker_subscription}"
        )

    def _project_name_from_bucket(self, bucket: str) -> str:
        if not bucket.startswith(self.bucket):
            raise ValueError(
                f"bucket {bucket!r} does not match configured prefix {self.bucket!r}"
            )

        project_name = bucket.removeprefix(self.bucket)
        if not project_name:
            raise ValueError("bucket name does not contain a project name")
        return project_name

    def _parse_gcs_event(self, message: Message) -> dict[str, str] | None:
        payload = json.loads(message.data.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Pub/Sub message payload must be a JSON object")

        data = payload.get("data", payload)
        if not isinstance(data, dict):
            raise ValueError("Pub/Sub message data must be a JSON object")

        attributes = dict(message.attributes or {})
        event_type = attributes.get("eventType")
        if event_type and event_type != "OBJECT_FINALIZE":
            return None

        bucket = data.get("bucket") or attributes.get("bucketId")
        file_name = data.get("name") or attributes.get("objectId")
        if not bucket or not file_name:
            raise ValueError("Pub/Sub message is missing GCS bucket/name")

        return {
            "bucket": bucket,
            "file_name": file_name,
            "project_name": self._project_name_from_bucket(bucket),
        }

    @staticmethod
    def _command_args(command: str) -> list[str]:
        args = shlex.split(command)
        if not args:
            raise ValueError("worker child command cannot be empty")
        if args[0].endswith(".py"):
            return [sys.executable, *args]
        return args

    def _run_child(self, event: dict[str, str]) -> int:
        child_env = {
            **os.environ,
            "WORKER_MODE": "false",
            "FILE_NAME": event["file_name"],
            "PROJECT_NAME": event["project_name"],
            self.source_bucket_env: event["bucket"],
            "PYTHONUNBUFFERED": "1",
        }
        self._logger.info(
            "Processing stage=%s bucket=%s file=%s project=%s",
            self.worker_stage,
            event["bucket"],
            event["file_name"],
            event["project_name"],
        )
        completed = subprocess.run(
            self._command_args(self.worker_child_command),
            env=child_env,
            check=False,
        )
        return completed.returncode

    def run_worker(self) -> int:
        if self._subscriber is None or self._streaming_pull_future is None:
            raise RuntimeError("setup() must be called before run_worker()")

        def stop(_signum, _frame):
            self._stopping.set()
            self._streaming_pull_future.cancel()

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)

        self._logger.info(
            "Worker listening on %s; stage=%s; max_messages=%d",
            self._subscription_name,
            self.worker_stage,
            self.worker_max_messages,
        )

        exit_code = 0
        try:
            self._streaming_pull_future.result()
            if not self._stopping.is_set():
                exit_code = 1
        except CancelledError:
            if not self._stopping.is_set():
                exit_code = 1
        except Exception:
            self._logger.exception("Pub/Sub worker stopped unexpectedly")
            exit_code = 1
        finally:
            self._wait_for_active_messages()
            self._subscriber.close()

        return exit_code

    def _wait_for_active_messages(self) -> None:
        deadline = time.monotonic() + self.worker_shutdown_grace_seconds
        with self._condition:
            while self._active_messages > 0 and time.monotonic() < deadline:
                remaining = max(0, int(deadline - time.monotonic()))
                self._logger.info(
                    "Waiting for %d in-flight message(s); grace=%ds",
                    self._active_messages,
                    remaining,
                )
                self._condition.wait(timeout=min(5, remaining or 1))

            if self._active_messages > 0:
                self._logger.warning(
                    "Shutdown grace expired with %d in-flight message(s)",
                    self._active_messages,
                )

    def _mark_active(self, delta: int) -> None:
        with self._condition:
            self._active_messages += delta
            self._condition.notify_all()

    def callback(self, message: Message) -> None:
        self._mark_active(1)
        try:
            try:
                event = self._parse_gcs_event(message)
                if event is None:
                    self._logger.info(
                        "Skipping non-finalize Pub/Sub message %s",
                        message.message_id,
                    )
                    message.ack()
                    return
            except Exception as exc:
                self._logger.error(
                    "Dropping invalid Pub/Sub message %s: %s",
                    message.message_id,
                    exc,
                )
                message.ack()
                return

            exit_code = self._run_child(event)
            if exit_code == 0:
                self._logger.info("Acking Pub/Sub message %s", message.message_id)
                message.ack()
            else:
                self._logger.error(
                    "Child command exited with %d; nacking Pub/Sub message %s",
                    exit_code,
                    message.message_id,
                )
                message.nack()
        except Exception:
            self._logger.exception(
                "Worker failed while processing Pub/Sub message %s",
                message.message_id,
            )
            message.nack()
        finally:
            self._mark_active(-1)
