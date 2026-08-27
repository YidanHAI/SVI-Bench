import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIRST = load_module(
    "audit_judge_first_pass_blindness",
    "scripts/audit_judge_first_pass_blindness.py",
)
REVIEW = load_module(
    "audit_judge_review_blindness",
    "scripts/audit_judge_review_blindness.py",
)
ADJUDICATION = load_module(
    "audit_judge_adjudication_blindness",
    "scripts/audit_judge_adjudication_blindness.py",
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reasons(report):
    return {item["reason"] for item in report["findings"]}


class BlindFixture:
    def __init__(self, root, count):
        self.root = Path(root)
        self.ids = [f"A{index:04d}" for index in range(1, count + 1)]
        self.tasks = [
            {
                "task_spec": {
                    "id": task_id,
                    "trigger": "synthetic trigger",
                    "expected_response_window": "within two seconds",
                    "expected_key_information": "synthetic information",
                    "dimensions": {},
                },
                "evidence_video": {"path": f"/synthetic/{task_id}.mp4"},
                "timing_source": {},
            }
            for task_id in self.ids
        ]
        self.manifest = self.root / f"manifest_{count}.json"
        write_json(self.manifest, {
            "schema_version": 1,
            "contains_human_labels": False,
            "task_count": count,
            "expected_task_count": count,
            "tasks": self.tasks,
        })
        self.prompt = self.root / "prompt.md"
        self.prompt.write_text("Grade only the supplied blind evidence.\n", encoding="utf-8")
        self.scoring = self.root / "scoring.py"
        self.video = self.root / "video.py"
        self.evidence = self.root / "evidence.py"
        self.scoring.write_text(
            "def apply_dimension_mean_score(value, spec):\n    return 0.5\n",
            encoding="utf-8",
        )
        self.video.write_text("def load_api_key():\n    return 'synthetic'\n", encoding="utf-8")
        self.evidence.write_text(
            "from scoring import apply_dimension_mean_score\n"
            "from video import load_api_key\n\n"
            "def build_evidence(task):\n"
            "    return {'source': 'synthetic'}, []\n",
            encoding="utf-8",
        )

    def task_by_id(self, task_id):
        return next(item for item in self.tasks if item["task_spec"]["id"] == task_id)

    def make_first_pass(self):
        script = self.root / "judge.py"
        script.write_text(
            "import json\n"
            "from evidence import build_evidence\n\n"
            "def call_gpt(user_prompt, frames):\n"
            "    return user_prompt, frames\n\n"
            "def judge_task(task):\n"
            "    task_spec = task['task_spec']\n"
            "    observation, frames = build_evidence(task)\n"
            "    judge_input = {\n"
            "        'task_spec': task_spec,\n"
            "        'observation': observation,\n"
            "    }\n"
            "    user_prompt = 'JUDGE_INPUT_JSON\\n' + json.dumps(judge_input)\n"
            "    return call_gpt(user_prompt, frames)\n",
            encoding="utf-8",
        )
        input_dir = self.root / "first_pass"
        for task in self.tasks:
            task_id = task["task_spec"]["id"]
            write_json(input_dir / task_id / "judge_input.json", {
                "task_spec": task["task_spec"],
                "observation": {"source": "synthetic"},
            })
        return script, input_dir

    def make_review(self):
        script = self.root / "review.py"
        script.write_text(
            "import json\n"
            "from evidence import build_evidence\n\n"
            "def call_gpt(user_prompt, frames):\n"
            "    return user_prompt, frames\n\n"
            "def review_task(task, first_result):\n"
            "    task_spec = task['task_spec']\n"
            "    observation, frames = build_evidence(task)\n"
            "    review_input = {\n"
            "        'task_spec': task_spec,\n"
            "        'observation': observation,\n"
            "        'first_pass_judgment': first_result['judge_json'],\n"
            "        'first_pass_provenance': {},\n"
            "    }\n"
            "    user_prompt = 'REVIEW_INPUT_JSON\\n' + json.dumps(review_input)\n"
            "    return call_gpt(user_prompt, frames)\n",
            encoding="utf-8",
        )
        first_dir = self.root / "review_first"
        review_dir = self.root / "review_inputs"
        for task in self.tasks:
            task_id = task["task_spec"]["id"]
            prediction_path = first_dir / task_id / "prediction.json"
            prediction = {
                "ok": True,
                "task_id": task_id,
                "model": "synthetic-model",
                "prompt_sha256": "a" * 64,
                "judge_json": {"task_id": task_id, "judge_status": "ok", "dimensions": {}},
            }
            write_json(prediction_path, prediction)
            write_json(review_dir / task_id / "review_input.json", {
                "task_spec": task["task_spec"],
                "observation": {"source": "synthetic"},
                "first_pass_judgment": prediction["judge_json"],
                "first_pass_provenance": {
                    "model": prediction["model"],
                    "prompt_sha256": prediction["prompt_sha256"],
                    "prediction_file_sha256": FIRST.sha256(prediction_path),
                },
            })
        return script, first_dir, review_dir

    def make_adjudication(self):
        script = self.root / "adjudicator.py"
        script.write_text(
            "import json\n"
            "from evidence import build_evidence\n\n"
            "def call_gpt(user_prompt, frames):\n"
            "    return user_prompt, frames\n\n"
            "def adjudicate_task(task, candidates):\n"
            "    task_spec = task['task_spec']\n"
            "    observation, frames = build_evidence(task)\n"
            "    adjudication_input = {\n"
            "        'task_spec': task_spec,\n"
            "        'observation': observation,\n"
            "        'candidate_judgments': candidates,\n"
            "    }\n"
            "    user_prompt = 'ADJUDICATION_INPUT_JSON\\n' + json.dumps(adjudication_input)\n"
            "    return call_gpt(user_prompt, frames)\n",
            encoding="utf-8",
        )
        candidate_dirs = [self.root / "candidate_1", self.root / "candidate_2"]
        out_dir = self.root / "adjudication"
        for task in self.tasks:
            task_id = task["task_spec"]["id"]
            candidates = []
            for index, directory in enumerate(candidate_dirs, start=1):
                prediction_path = directory / task_id / "prediction.json"
                prediction = {
                    "ok": True,
                    "task_id": task_id,
                    "model": f"synthetic-model-{index}",
                    "prompt_sha256": str(index) * 64,
                    "review_stage": f"synthetic-stage-{index}",
                    "judge_json": {"task_id": task_id, "judge_status": "ok", "dimensions": {}},
                }
                write_json(prediction_path, prediction)
                candidates.append({
                    "candidate": f"candidate_{index}",
                    "judge_json": prediction["judge_json"],
                    "provenance": {
                        "path": str(prediction_path.resolve()),
                        "sha256": FIRST.sha256(prediction_path),
                        "model": prediction["model"],
                        "stage": prediction["review_stage"],
                        "prompt_sha256": prediction["prompt_sha256"],
                    },
                })
            write_json(out_dir / task_id / "adjudication_input.json", {
                "task_spec": task["task_spec"],
                "observation": {"source": "synthetic"},
                "candidate_judgments": candidates,
            })
            write_json(out_dir / task_id / "prediction.json", {
                "ok": True,
                "task_id": task_id,
                "judge_json": {"task_id": task_id},
            })
        return script, candidate_dirs, out_dir


class JudgeBlindnessScopeTest(unittest.TestCase):
    def test_safe_regex_compile_is_not_dynamic_code_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "regex_helper.py"
            source.write_text(
                "import re\n"
                "PATTERN = re.compile(r'^[A-Z]+$')\n",
                encoding="utf-8",
            )

            self.assertEqual(FIRST.audit_source_file(source, set()), [])

    def test_builtin_compile_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for expression in (
                "compile('value = 1', '<string>', 'exec')",
                "builtins.compile('value = 1', '<string>', 'exec')",
            ):
                source = Path(tmp) / "dynamic_helper.py"
                source.write_text(
                    "import builtins\n"
                    f"CODE = {expression}\n",
                    encoding="utf-8",
                )

                findings = FIRST.audit_source_file(source, set())
                self.assertIn("forbidden_source_call", {
                    item["reason"] for item in findings
                })

    def test_full_first_pass_scope_is_bound_and_recursive(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BlindFixture(tmp, 75)
            script, input_dir = fixture.make_first_pass()
            report = FIRST.audit_first_pass(
                fixture.manifest, script, fixture.prompt, input_dir
            )

            self.assertEqual(report["status"], "pass", report["findings"])
            scope = report["scope"]
            self.assertEqual(scope["cohort"], "formal_75")
            self.assertEqual(scope["expected_tasks"], 75)
            self.assertEqual(scope["checked_tasks"], 75)
            self.assertEqual(scope["manifest_sha256"], FIRST.sha256(fixture.manifest))
            self.assertEqual(scope["judge_script_sha256"], FIRST.sha256(script))
            self.assertEqual(scope["system_prompt_sha256"], FIRST.sha256(fixture.prompt))
            self.assertEqual(scope["auditor_sha256"], FIRST.sha256(FIRST.AUDITOR_PATH))
            dependency_names = {Path(item["path"]).name for item in scope["dependencies"]}
            self.assertEqual(dependency_names, {"evidence.py", "scoring.py", "video.py"})

    def test_transitive_dependency_human_label_attack_fails_all_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BlindFixture(tmp, 75)
            fixture.scoring.write_text(
                "gold_labels = {'synthetic': 'G'}\n"
                "def apply_dimension_mean_score(value, spec):\n"
                "    return gold_labels.get('synthetic')\n",
                encoding="utf-8",
            )
            first_script, first_input = fixture.make_first_pass()
            first_report = FIRST.audit_first_pass(
                fixture.manifest, first_script, fixture.prompt, first_input
            )
            review_script, first_dir, review_dir = fixture.make_review()
            review_report = REVIEW.audit_review(
                fixture.manifest,
                first_dir,
                review_script,
                fixture.prompt,
                review_dir,
            )
            adjudicator, candidate_dirs, adjudication_dir = fixture.make_adjudication()
            adjudication_report = ADJUDICATION.audit_adjudication(
                fixture.manifest,
                adjudicator,
                fixture.prompt,
                adjudication_dir,
                candidate_dirs,
            )

            for report in (first_report, review_report, adjudication_report):
                self.assertEqual(report["status"], "fail")
                self.assertIn("human_label_identifier_in_source", reasons(report))
                dependency_names = {
                    Path(item["path"]).name for item in report["scope"]["dependencies"]
                }
                self.assertIn("scoring.py", dependency_names)

    def test_transitive_task_id_special_case_and_dynamic_import_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BlindFixture(tmp, 75)
            fixture.evidence.write_text(
                "import importlib\n"
                "module_name = 'scoring'\n"
                "scoring = importlib.import_module(module_name)\n\n"
                "def build_evidence(task):\n"
                "    if task['task_spec']['id'] == 'A0001':\n"
                "        return {'forced': True}, []\n"
                "    return {'forced': False}, []\n",
                encoding="utf-8",
            )
            script, input_dir = fixture.make_first_pass()
            report = FIRST.audit_first_pass(
                fixture.manifest, script, fixture.prompt, input_dir
            )

            self.assertEqual(report["status"], "fail")
            self.assertIn("literal_task_ids", reasons(report))
            self.assertIn("dynamic_import_not_statically_auditable", reasons(report))

    def test_nested_human_label_in_request_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BlindFixture(tmp, 75)
            script, input_dir = fixture.make_first_pass()
            target = input_dir / fixture.ids[0] / "judge_input.json"
            value = FIRST.load_json(target)
            value["observation"]["human_scores"] = {"D1": 1}
            write_json(target, value)

            report = FIRST.audit_first_pass(
                fixture.manifest, script, fixture.prompt, input_dir
            )

            self.assertEqual(report["status"], "fail")
            self.assertIn("forbidden_human_label_key", reasons(report))

    def test_human_source_reference_inside_task_spec_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BlindFixture(tmp, 75)
            fixture.tasks[0]["task_spec"]["notes"] = "expert_scores.xlsx"
            write_json(fixture.manifest, {
                "schema_version": 1,
                "contains_human_labels": False,
                "task_count": 75,
                "expected_task_count": 75,
                "tasks": fixture.tasks,
            })
            script, input_dir = fixture.make_first_pass()

            report = FIRST.audit_first_pass(
                fixture.manifest, script, fixture.prompt, input_dir
            )

            self.assertEqual(report["status"], "fail")
            self.assertIn("forbidden_human_source_reference", reasons(report))

    def test_unpersisted_runtime_context_cannot_be_added_to_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BlindFixture(tmp, 75)
            script, input_dir = fixture.make_first_pass()
            text = script.read_text(encoding="utf-8")
            text = text.replace(
                "    user_prompt = 'JUDGE_INPUT_JSON\\n' + json.dumps(judge_input)\n",
                "    extra_context = task.get('extra_context', '')\n"
                "    user_prompt = 'JUDGE_INPUT_JSON\\n' + json.dumps(judge_input) + extra_context\n",
            )
            script.write_text(text, encoding="utf-8")

            report = FIRST.audit_first_pass(
                fixture.manifest, script, fixture.prompt, input_dir
            )

            self.assertEqual(report["status"], "fail")
            self.assertIn("unapproved_user_prompt_inputs", reasons(report))

    def test_manifest_schema_count_duplicates_and_label_flag_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BlindFixture(tmp, 75)
            script, input_dir = fixture.make_first_pass()
            manifest = FIRST.load_json(fixture.manifest)
            manifest["contains_human_labels"] = True
            manifest["task_count"] = 74
            manifest["tasks"][1]["task_spec"]["id"] = manifest["tasks"][0]["task_spec"]["id"]
            write_json(fixture.manifest, manifest)

            report = FIRST.audit_first_pass(
                fixture.manifest, script, fixture.prompt, input_dir, expected_tasks=75
            )

            self.assertEqual(report["status"], "fail")
            self.assertIn("manifest_not_explicitly_label_free", reasons(report))
            self.assertIn("manifest_task_count_mismatch", reasons(report))
            self.assertIn("duplicate_manifest_task_ids", reasons(report))

    def test_corrupt_upstream_prediction_fails_without_crashing_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BlindFixture(tmp, 75)
            review_script, first_dir, review_dir = fixture.make_review()
            (first_dir / fixture.ids[0] / "prediction.json").write_text(
                "{not-json\n", encoding="utf-8"
            )

            report = REVIEW.audit_review(
                fixture.manifest,
                first_dir,
                review_script,
                fixture.prompt,
                review_dir,
            )

            self.assertEqual(report["status"], "fail")
            self.assertIn("invalid_first_pass_prediction_json", reasons(report))

    def test_review_and_adjudication_emit_complete_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BlindFixture(tmp, 75)
            review_script, first_dir, review_dir = fixture.make_review()
            review_report = REVIEW.audit_review(
                fixture.manifest,
                first_dir,
                review_script,
                fixture.prompt,
                review_dir,
            )
            adjudicator, candidate_dirs, adjudication_dir = fixture.make_adjudication()
            adjudication_report = ADJUDICATION.audit_adjudication(
                fixture.manifest,
                adjudicator,
                fixture.prompt,
                adjudication_dir,
                candidate_dirs,
            )

            for report, script_key in (
                (review_report, "review_script_sha256"),
                (adjudication_report, "adjudicator_script_sha256"),
            ):
                self.assertEqual(report["status"], "pass", report["findings"])
                scope = report["scope"]
                self.assertEqual(scope["cohort"], "formal_75")
                self.assertEqual(scope["expected_tasks"], 75)
                self.assertEqual(scope["checked_tasks"], 75)
                self.assertIn("manifest_sha256", scope)
                self.assertIn(script_key, scope)
                self.assertIn("system_prompt_sha256", scope)
                self.assertIn("request_inputs_sha256", scope)
                self.assertIn("auditor_sha256", scope)
                self.assertIn("audit_library_sha256", scope)
                dependency_names = {
                    Path(item["path"]).name for item in scope["dependencies"]
                }
                self.assertTrue({"evidence.py", "scoring.py", "video.py"} <= dependency_names)

if __name__ == "__main__":
    unittest.main()
