import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "scripts"))
import attest_full_judge_evidence as attester  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


class SyntheticFullRun:
    def __init__(self, root):
        self.root = Path(root)
        self.out_root = self.root / "judge"
        self.manifest_path = self.root / "manifest.json"
        self.module_path = self.root / "evidence_module.py"
        self.report_path = self.root / "attestation.json"
        self.run_summary = self.root / "capture" / "run_summary.json"
        self.source_video = self.root / "capture" / "source.mp4"
        self.tasks = []
        self._create()

    @staticmethod
    def observation(task_id, video_path, source_path, nonce):
        return {
            "recording_duration_s": 1.0,
            "recording_file": Path(video_path).name,
            "source_video_available": True,
            "source_video_file": Path(source_path).name,
            "timing_precision": "capture_log",
            "timeline": {"task_id": task_id, "nonce": nonce},
            "screenshot_sampling": {
                "count": 1,
                "items": [
                    {
                        "evidence_id": "EVIDENCE_IMAGE_000",
                        "time_s": 0.25,
                        "view": "full_ui",
                        "clock": "recording wall time",
                    }
                ],
            },
        }

    def _create_module(self):
        self.module_path.write_text(
            """import json
from pathlib import Path


def build_evidence(task, task_dir, max_images, workers, max_temporal_sheets):
    task_id = task["task_spec"]["id"]
    video = Path(task["evidence_video"]["path"])
    summary = json.loads(Path(task["timing_source"]["summary_json"]).read_text(encoding="utf-8"))
    source = Path(summary["task"]["local_video_path"])
    nonce = summary["nonce"]
    frame = Path(task_dir) / "evidence" / "frame_000_000000.250s.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(f"frame:{task_id}:{nonce}".encode("utf-8"))
    observation = {
        "recording_duration_s": 1.0,
        "recording_file": video.name,
        "source_video_available": True,
        "source_video_file": source.name,
        "timing_precision": "capture_log",
        "timeline": {"task_id": task_id, "nonce": nonce},
        "screenshot_sampling": {
            "count": 1,
            "items": [{
                "evidence_id": "EVIDENCE_IMAGE_000",
                "time_s": 0.25,
                "view": "full_ui",
                "clock": "recording wall time",
            }],
        },
    }
    return observation, [(0.25, frame, "full_ui")]
""",
            encoding="utf-8",
        )

    def _create(self):
        self.run_summary.parent.mkdir(parents=True)
        write_json(self.run_summary, {"status": "complete", "tasks": 75})
        self.source_video.write_bytes(b"synthetic-source-video")
        self._create_module()

        for index in range(1, 76):
            task_id = f"T{index:03d}"
            capture_dir = self.root / "capture" / task_id
            capture_dir.mkdir(parents=True)
            video = capture_dir / f"{task_id}_task.mp4"
            video.write_bytes(f"video:{task_id}".encode("utf-8"))
            events = capture_dir / "events.jsonl"
            events.write_text(
                json.dumps({"task_id": task_id, "t_ms": 250}) + "\n",
                encoding="utf-8",
            )
            summary = capture_dir / "summary.json"
            write_json(
                summary,
                {
                    "status": "ok",
                    "nonce": f"nonce-{task_id}",
                    "task": {
                        "id": task_id,
                        "local_video_path": str(self.source_video.resolve()),
                    },
                    "local_video_upload": {
                        "source_video_path": str(self.source_video.resolve())
                    },
                    "files": {
                        "events_jsonl": str(events.resolve()),
                        "task_mp4": str(video.resolve()),
                    },
                },
            )
            task_spec = {
                "id": task_id,
                "category": "synthetic",
                "user_prompt": f"query-{index}",
                "dimensions": {
                    "D1": {
                        "applicable": True,
                        "weight": 1.0,
                        "threshold": "G/S/B rubric without assigned rating",
                    }
                },
            }
            self.tasks.append(
                {
                    "task_spec": task_spec,
                    "evidence_video": {
                        "path": str(video.resolve()),
                        "bytes": video.stat().st_size,
                        "sha256": sha256(video),
                    },
                    "timing_source": {
                        "summary_json": str(summary.resolve()),
                        "recording_match": "same_capture_task",
                    },
                    "capture_provenance": {
                        "model_id": "synthetic-model",
                        "run_summary": str(self.run_summary.resolve()),
                        "task_index": index,
                    },
                }
            )

        manifest = {
            "schema_version": 1,
            "contains_human_labels": False,
            "capture_model_id": "synthetic-model",
            "capture_run_summary": str(self.run_summary.resolve()),
            "capture_run_summary_sha256": sha256(self.run_summary),
            "expected_task_count": 75,
            "task_count": 75,
            "missing_task_ids": [],
            "tasks": self.tasks,
        }
        write_json(self.manifest_path, manifest)
        manifest_hash = sha256(self.manifest_path)
        module_hash = sha256(self.module_path)

        for task in self.tasks:
            task_id = task["task_spec"]["id"]
            summary = json.loads(
                Path(task["timing_source"]["summary_json"]).read_text(encoding="utf-8")
            )
            observation = self.observation(
                task_id,
                task["evidence_video"]["path"],
                summary["task"]["local_video_path"],
                summary["nonce"],
            )
            first_task_dir = self.out_root / "01_first_pass" / task_id
            frame = first_task_dir / "evidence" / "frame_000_000000.250s.jpg"
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(f"frame:{task_id}:{summary['nonce']}".encode("utf-8"))
            write_json(
                first_task_dir / "evidence_preflight.json",
                {
                    "ok": True,
                    "task_id": task_id,
                    "manifest_sha256": manifest_hash,
                    "judge_script_sha256": module_hash,
                    "evidence_count": 1,
                    "observation": observation,
                    "evidence": [
                        {
                            "index": 0,
                            "timestamp_s": 0.25,
                            "view": "full_ui",
                            "path": str(frame.resolve()),
                            "bytes": frame.stat().st_size,
                            "sha256": sha256(frame),
                        }
                    ],
                },
            )
            for stage, directory, filename in attester.STAGES:
                stage_input = {
                    "task_spec": task["task_spec"],
                    "observation": observation,
                }
                if stage == "adjudication":
                    stage_input["candidate_judgments"] = []
                elif stage != "first_pass":
                    stage_input["first_pass_judgment"] = {}
                write_json(self.out_root / directory / task_id / filename, stage_input)

    def configure_motion_edge(self):
        task_id = "T001"
        stage = "first_pass"
        task_dir = self.out_root / "01_first_pass" / task_id
        frame = task_dir / "evidence" / "frame_000_000000.250s.jpg"
        edge = attester._motion_edge_path(frame)
        edge.write_bytes(b"synthetic-motion-edge-bytes")

        prompt = self.root / "judge_prompt.md"
        prompt.write_text("Synthetic blind Judge system prompt.\n", encoding="utf-8")
        manifest_hash = sha256(self.manifest_path)
        module_hash = sha256(self.module_path)
        prompt_hash = sha256(prompt)
        prompt_content_hash = hashlib.sha256(
            prompt.read_text(encoding="utf-8").strip().encode("utf-8")
        ).hexdigest()
        model = "synthetic-gpt"
        api_url = "https://judge.invalid/v1/chat/completions"
        input_modes = {
            "first_pass": "timestamped_images_temporal_sheets_plus_capture_timeline",
            "review_1": "first_pass_plus_same_timestamped_evidence",
            "review_2": "first_pass_plus_same_timestamped_evidence",
            "adjudication": "candidate_judgments_plus_same_timestamped_evidence",
            "final_review": "first_pass_plus_same_timestamped_evidence",
        }
        state_stages = {}
        identities = {}
        for stage_name, directory_name, _ in attester.STAGES:
            identity = {
                "stage": stage_name,
                "manifest_sha256": manifest_hash,
                "script_sha256": module_hash,
                "prompt_sha256": prompt_hash,
                "prompt_content_sha256": prompt_content_hash,
                "requested_model": model,
                "api_url": api_url,
                "native_video_input": False,
                "input_mode": input_modes[stage_name],
                "max_http_attempts_per_task": 5,
            }
            identities[stage_name] = identity
            state_stages[stage_name] = {
                "directory": str((self.out_root / directory_name).resolve()),
                "script": str(self.module_path.resolve()),
                "script_sha256": module_hash,
                "prompt": str(prompt.resolve()),
                "prompt_sha256": prompt_hash,
                "prompt_content_sha256": prompt_content_hash,
                "execution_identity": identity,
            }
        write_json(
            self.out_root / attester.PIPELINE_STATE_NAME,
            {
                "pipeline": "five_stage_judge",
                "contains_human_labels": False,
                "task_count": 75,
                "manifest_sha256": manifest_hash,
                "evidence_module_sha256": module_hash,
                "judge_model": model,
                "judge_stage_count": 5,
                "stages": state_stages,
            },
        )
        for stage_name, directory_name, _ in attester.STAGES:
            if stage_name != stage:
                write_json(
                    self.out_root
                    / directory_name
                    / task_id
                    / attester.PREDICTION_NAME,
                    {"evidence_variant": "complete_evidence"},
                )

        persisted_input = json.loads(
            (task_dir / "judge_input.json").read_text(encoding="utf-8")
        )
        user_prompt = attester.STAGE_USER_PROMPT_PREFIXES[stage] + json.dumps(
            persisted_input, ensure_ascii=False, indent=2
        )
        request_hash = attester._request_payload_sha256(
            model=model,
            system_prompt=prompt.read_text(encoding="utf-8").strip(),
            user_prompt=user_prompt,
            frames=[
                {
                    "timestamp_s": 0.25,
                    "view": "full_ui",
                    "motion_edge": {"path": str(edge.resolve())},
                }
            ],
            max_tokens=12000,
            request_seed=None,
        )
        prediction = {
            "ok": True,
            "task_id": task_id,
            "model": model,
            "api_url": api_url,
            "prompt_file": str(prompt.resolve()),
            "prompt_sha256": prompt_content_hash,
            "native_video_input": False,
            "input_mode": input_modes[stage],
            "evidence_image_count": 1,
            "evidence_image_count_sent": 1,
            "evidence_variant": "motion_edge_evidence",
            "request_attempt": 1,
            "judge_json": {"task_id": task_id},
        }
        prediction_path = task_dir / attester.PREDICTION_NAME
        write_json(prediction_path, prediction)

        attempts = [
            {
                "attempt": 1,
                "claimed_at": "2026-01-01T00:00:00+00:00",
                "status": "response",
                "method": "POST",
                "url": api_url,
                "request_sha256": request_hash,
                "finished_at": "2026-01-01T00:00:01+00:00",
                "status_code": 200,
            }
        ]
        identity = identities[stage]
        ledger_identity = {
            "schema_version": 2,
            "stage": stage,
            "max_attempts_per_task": 5,
            "manifest_sha256": manifest_hash,
            "task_ids_sha256": attester.canonical_sha256(
                sorted(task["task_spec"]["id"] for task in self.tasks)
            ),
            "task_count": 75,
            "stage_identity_sha256": attester.canonical_sha256(identity),
        }
        ledger = {
            **ledger_identity,
            "identity_sha256": attester.canonical_sha256(ledger_identity),
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:01+00:00",
            "revision": 1,
            "tasks": {task_id: {"attempts": attempts, "attempt_count": 1}},
        }
        ledger["integrity_sha256"] = attester.canonical_sha256(ledger)
        ledger_path = self.out_root / "attempt_ledgers" / f"{stage}.json"
        write_json(ledger_path, ledger)
        seal = {
            "schema_version": 1,
            "identity_sha256": ledger["identity_sha256"],
            "ledger_integrity_sha256": ledger["integrity_sha256"],
            "revision": ledger["revision"],
        }
        seal["seal_sha256"] = attester.canonical_sha256(seal)
        write_json(ledger_path.with_name(ledger_path.name + ".seal"), seal)

        prediction_reference = {
            "path": str(prediction_path.resolve()),
            "bytes": prediction_path.stat().st_size,
            "sha256": sha256(prediction_path),
        }
        execution = {
            "schema_version": 1,
            "pass": True,
            "stage": stage,
            "task_id": task_id,
            "stage_identity": identity,
            "artifacts": [prediction_reference],
            "attempt_ledger": {
                "path": str(ledger_path.resolve()),
                "identity_sha256": ledger["identity_sha256"],
                "attempt_count": 1,
                "attempts_sha256": attester.canonical_sha256(attempts),
            },
        }
        execution["attestation_integrity_sha256"] = attester.canonical_sha256(
            execution
        )
        write_json(task_dir / attester.EXECUTION_ATTESTATION_NAME, execution)
        return edge

    def run(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = attester.main(
                [
                    "--manifest",
                    str(self.manifest_path),
                    "--out-root",
                    str(self.out_root),
                    "--evidence-module",
                    str(self.module_path),
                    "--out",
                    str(self.report_path),
                ]
            )
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        return code, report


class FullEvidenceAttestationTest(unittest.TestCase):
    def make_run(self, temporary):
        return SyntheticFullRun(Path(temporary) / "fixture")

    def test_complete_75_task_five_stage_run_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_run(temporary)
            code, report = fixture.run()

            self.assertEqual(code, 0)
            self.assertTrue(report["pass"])
            self.assertFalse(report["contains_human_labels"])
            self.assertEqual(report["summary"]["tasks_attested"], 75)
            self.assertEqual(report["summary"]["tasks_passed"], 75)
            self.assertEqual(report["summary"]["stage_inputs_attested"], 375)
            self.assertTrue(report["checks"]["all_375_stage_inputs_match"])
            first = report["tasks"][0]
            self.assertEqual(len(first["stage_inputs"]), 5)
            self.assertTrue(first["rebuilt_evidence"]["frames_match_preflight"])
            roles = {
                role
                for source in first["timing_and_capture_sources"]
                for role in source["roles"]
            }
            self.assertIn("timing_source.summary_json", roles)
            self.assertIn("manifest.capture_run_summary", roles)
            self.assertIn("timing_summary.files.events_jsonl", roles)
            self.assertNotIn("judge_json", json.dumps(report, ensure_ascii=False))

    def test_synthetic_tampering_fails_closed(self):
        attacks = {
            "evidence_video_content": self._tamper_video,
            "preflight_frame_content": self._tamper_frame,
            "timing_summary_content": self._tamper_timing_summary,
            "stage_task_spec": self._tamper_stage_task_spec,
            "stage_observation": self._tamper_stage_observation,
            "missing_stage_input": self._remove_stage_input,
            "extra_stage_input": self._add_stage_input,
            "manifest_has_only_74_tasks": self._remove_manifest_task,
            "preflight_identity_drift": self._tamper_preflight_identity,
            "capture_run_summary_drift": self._tamper_capture_run_summary,
            "evidence_module_drift": self._tamper_module,
            "unexpected_evidence_file": self._add_unexpected_evidence_file,
        }
        for name, attack in attacks.items():
            with self.subTest(attack=name), tempfile.TemporaryDirectory() as temporary:
                fixture = self.make_run(temporary)
                attack(fixture)
                code, report = fixture.run()
                self.assertNotEqual(code, 0)
                self.assertFalse(report["pass"])

    def test_manifest_label_injection_is_rejected_without_copying_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_run(temporary)
            manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
            manifest["tasks"][0]["task_spec"]["expert_score"] = "SECRET_GOLD_VALUE"
            write_json(fixture.manifest_path, manifest)

            code, report = fixture.run()
            serialized = json.dumps(report, ensure_ascii=False)
            self.assertNotEqual(code, 0)
            self.assertFalse(report["pass"])
            self.assertEqual(report["summary"]["tasks_attested"], 0)
            self.assertNotIn("SECRET_GOLD_VALUE", serialized)
            self.assertIn("manifest_contains_forbidden_label_fields", report["errors"])

    def test_motion_edge_request_provenance_passes_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_run(temporary)
            fixture.configure_motion_edge()
            code, report = fixture.run()
            inventory = report["tasks"][0]["evidence_preflight"][
                "evidence_directory"
            ]
            self.assertEqual(code, 0)
            self.assertTrue(report["pass"])
            self.assertEqual(inventory["mode"], "motion_edge_evidence")
            self.assertTrue(inventory["complete_one_to_one"])
            self.assertEqual(len(inventory["successful_request_provenance"]), 1)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_run(temporary)
            edge = fixture.configure_motion_edge()
            edge.write_bytes(edge.read_bytes() + b"tampered")
            code, report = fixture.run()
            self.assertNotEqual(code, 0)
            self.assertFalse(report["pass"])
            self.assertIn(
                "stage_first_pass_motion_edge_request_hash_mismatch",
                report["tasks"][0]["errors"],
            )

    @staticmethod
    def _tamper_video(fixture):
        path = Path(fixture.tasks[0]["evidence_video"]["path"])
        path.write_bytes(path.read_bytes() + b"tampered")

    @staticmethod
    def _tamper_frame(fixture):
        path = (
            fixture.out_root
            / "01_first_pass"
            / "T001"
            / "evidence"
            / "frame_000_000000.250s.jpg"
        )
        path.write_bytes(b"tampered-frame")

    @staticmethod
    def _tamper_timing_summary(fixture):
        path = Path(fixture.tasks[0]["timing_source"]["summary_json"])
        value = json.loads(path.read_text(encoding="utf-8"))
        value["nonce"] = "tampered-nonce"
        write_json(path, value)

    @staticmethod
    def _tamper_stage_task_spec(fixture):
        path = fixture.out_root / "02_review_1" / "T001" / "review_input.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["task_spec"]["user_prompt"] = "tampered-query"
        write_json(path, value)

    @staticmethod
    def _tamper_stage_observation(fixture):
        path = fixture.out_root / "03_review_2" / "T001" / "review_input.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["observation"]["timeline"]["nonce"] = "tampered-observation"
        write_json(path, value)

    @staticmethod
    def _remove_stage_input(fixture):
        path = fixture.out_root / "05_final_review" / "T001" / "review_input.json"
        path.unlink()

    @staticmethod
    def _add_stage_input(fixture):
        path = fixture.out_root / "02_review_1" / "EXTRA" / "review_input.json"
        write_json(path, {"task_spec": {"id": "EXTRA"}, "observation": {}})

    @staticmethod
    def _remove_manifest_task(fixture):
        value = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
        value["tasks"].pop()
        value["task_count"] = 74
        write_json(fixture.manifest_path, value)

    @staticmethod
    def _tamper_preflight_identity(fixture):
        path = (
            fixture.out_root
            / "01_first_pass"
            / "T001"
            / "evidence_preflight.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        value["task_id"] = "T002"
        write_json(path, value)

    @staticmethod
    def _tamper_capture_run_summary(fixture):
        value = json.loads(fixture.run_summary.read_text(encoding="utf-8"))
        value["status"] = "tampered"
        write_json(fixture.run_summary, value)

    @staticmethod
    def _tamper_module(fixture):
        fixture.module_path.write_text(
            fixture.module_path.read_text(encoding="utf-8") + "\n# drift\n",
            encoding="utf-8",
        )

    @staticmethod
    def _add_unexpected_evidence_file(fixture):
        path = (
            fixture.out_root
            / "01_first_pass"
            / "T001"
            / "evidence"
            / "unexpected.bin"
        )
        path.write_bytes(b"unexpected")


if __name__ == "__main__":
    unittest.main()
