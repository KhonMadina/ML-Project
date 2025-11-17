import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation.finalize_dataset import parse_adjudication_log
from annotation.adjudicate import load_label_studio_json
import json

def test_parse_adjudication_log_tolerant(tmp_path):
    log = tmp_path / "adjudication_log.md"
    content = """
## Decisions
- Item ID: 1001
  - Final Label: POS
* Item ID: 1002
  * Final  Label :  neutral
1. Item ID: 1003
   - Final Label: -
- Item ID: 1004
  - Final Label: <POS/NEG/NEU>
- Item ID: 1005
  - Final Label: NEGATIVE
""".strip()
    log.write_text(content, encoding="utf-8")

    decisions = parse_adjudication_log(log)
    # 1001 -> POS, 1002 -> NEU (normalized), 1003 -> NEG (normalized '-'), 1004 skipped (placeholder), 1005 -> NEGATIVE -> NEG
    assert decisions.get("1001") == "POS"
    assert decisions.get("1002") == "NEU"
    assert decisions.get("1003") == "NEG"
    assert "1004" not in decisions
    assert decisions.get("1005") == "NEG"


def test_load_label_studio_json_variants(tmp_path):
    # Build a minimal LS-like JSON with variants
    data = [
        {
            "id": 1,
            "data": {"text": "great"},
            "annotations": [
                {"completed_by": {"username": "a"}, "result": [{"type": "choices", "value": {"choices": ["POS"]}}]}
            ]
        },
        {
            "task_id": 2,
            "text": "bad",
            "completions": [
                {"completed_by": "b", "results": [{"result_type": "choices", "value": {"labels": ["NEG"]}}]}
            ]
        },
        {
            "id": 3,
            "data": {"text": "meh"},
            "annotations": [
                {"completed_by": {"username": "c"}, "result": [{"type": "choices", "value": {"choices": []}}]}
            ]
        }
    ]
    p = tmp_path / "export.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    anns = load_label_studio_json(p)
    # Expect two valid annotations (POS and NEG), the empty choices should be skipped
    assert len(anns) == 2
    labels = sorted(set(a.label for a in anns))
    assert labels == ["NEG", "POS"]
