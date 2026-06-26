#!/usr/bin/env python3
"""Tests for ethnocoder."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock heavy dependencies not needed for pure-function tests
sys.modules.setdefault("pymupdf", MagicMock())
sys.modules.setdefault("litellm", MagicMock())
sys.modules.setdefault("rich", MagicMock())
sys.modules.setdefault("rich.console", MagicMock())
sys.modules.setdefault("rich.table", MagicMock())
sys.modules.setdefault("rich.text", MagicMock())

from code_traits import (
    build_coding_prompt,
    build_review_message,
    model_dirname,
    parse_codings,
    strip_fences,
    validate_option_codes,
)
from evaluate import load_codings, load_codings_as_dict, normalize_code
from setup_dataset import strip_pages


# ---------------------------------------------------------------------------
# code_traits: strip_fences
# ---------------------------------------------------------------------------

class TestStripFences:
    def test_no_fences(self):
        assert strip_fences('{"a": 1}') == '{"a": 1}'

    def test_json_fences(self):
        assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_plain_fences(self):
        assert strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_whitespace(self):
        assert strip_fences('  \n```json\n{"a": 1}\n```\n  ') == '{"a": 1}'

    def test_no_closing_fence(self):
        assert strip_fences('```json\n{"a": 1}') == '{"a": 1}'

    def test_no_opening_fence(self):
        assert strip_fences('{"a": 1}\n```') == '{"a": 1}'


# ---------------------------------------------------------------------------
# code_traits: parse_codings
# ---------------------------------------------------------------------------

class TestParseCodings:
    def test_codings_wrapper(self):
        text = json.dumps({"codings": [{"id": 1, "code": "0"}]})
        assert parse_codings(text) == [{"id": 1, "code": "0"}]

    def test_bare_list(self):
        text = json.dumps([{"id": 1, "code": "0"}])
        assert parse_codings(text) == [{"id": 1, "code": "0"}]

    def test_double_brace_bug(self):
        text = "{{" + json.dumps({"codings": [{"id": 2, "code": "1"}]})[1:]
        result = parse_codings(text)
        assert result == [{"id": 2, "code": "1"}]

    def test_raw_response_wrapper(self):
        inner = json.dumps({"codings": [{"id": 3, "code": "2"}]})
        text = json.dumps({"timestamp": "2024-01-01", "raw_response": inner})
        assert parse_codings(text) == [{"id": 3, "code": "2"}]

    def test_raw_response_with_fences(self):
        inner = "```json\n" + json.dumps({"codings": [{"id": 4, "code": "0"}]}) + "\n```"
        text = json.dumps({"raw_response": inner})
        assert parse_codings(text) == [{"id": 4, "code": "0"}]

    def test_empty_codings(self):
        text = json.dumps({"codings": []})
        assert parse_codings(text) == []


# ---------------------------------------------------------------------------
# code_traits: model_dirname
# ---------------------------------------------------------------------------

class TestModelDirname:
    def test_with_provider(self):
        assert model_dirname("ollama/llama3.2") == "llama3.2"

    def test_with_anthropic(self):
        assert model_dirname("anthropic/claude-opus-4-6") == "claude-opus-4-6"

    def test_bare_model(self):
        assert model_dirname("claude-opus-4-6") == "claude-opus-4-6"


# ---------------------------------------------------------------------------
# code_traits: build_coding_prompt
# ---------------------------------------------------------------------------

class TestBuildCodingPrompt:
    def test_option_variable(self):
        variables = [{"ID": "2", "Name": "Belief in god(s)", "Datatype": "Option",
                       "Section": "Belief", "Description": "A god is a supernatural agent."}]
        codes = {"2": [{"Name": "0", "Description": "Absent"}, {"Name": "1", "Description": "Present"}]}
        prompt = build_coding_prompt(variables, codes)
        assert "ID 2: Belief in god(s)" in prompt
        assert "0=Absent" in prompt
        assert "1=Present" in prompt
        assert "Assign one code from the list above." in prompt

    def test_int_variable(self):
        variables = [{"ID": "45", "Name": "Population", "Datatype": "Int",
                       "Section": "Social", "Description": ""}]
        prompt = build_coding_prompt(variables, {})
        assert "Assign an integer value." in prompt

    def test_float_variable(self):
        variables = [{"ID": "28", "Name": "Distance", "Datatype": "Float",
                       "Section": "Isolation", "Description": "Distance in km."}]
        prompt = build_coding_prompt(variables, {})
        assert "Assign a numeric value." in prompt

    def test_text_variable(self):
        variables = [{"ID": "1", "Name": "Time Focus", "Datatype": "Text",
                       "Section": "Time", "Description": ""}]
        prompt = build_coding_prompt(variables, {})
        assert "Provide a brief text value." in prompt

    def test_section_headers(self):
        variables = [
            {"ID": "1", "Name": "V1", "Datatype": "Text", "Section": "A", "Description": ""},
            {"ID": "2", "Name": "V2", "Datatype": "Text", "Section": "A", "Description": ""},
            {"ID": "3", "Name": "V3", "Datatype": "Text", "Section": "B", "Description": ""},
        ]
        prompt = build_coding_prompt(variables, {})
        assert prompt.count("[A]") == 1
        assert prompt.count("[B]") == 1

    def test_option_no_codes(self):
        variables = [{"ID": "99", "Name": "Unknown", "Datatype": "Option",
                       "Section": "X", "Description": ""}]
        prompt = build_coding_prompt(variables, {})
        assert "Assign one code from the list above." in prompt
        assert "Valid codes:" not in prompt


# ---------------------------------------------------------------------------
# code_traits: validate_option_codes
# ---------------------------------------------------------------------------

class TestValidateOptionCodes:
    def setup_method(self):
        self.variables = [{"ID": "2", "Datatype": "Option"}]
        self.codes_by_var = {"2": [{"Name": "0"}, {"Name": "1"}, {"Name": "2"}]}

    def test_valid_code_passes(self):
        codings = [{"id": "2", "code": "1", "confidence": "high"}]
        result = validate_option_codes(codings, self.codes_by_var, self.variables)
        assert "_invalid" not in result[0]

    def test_invalid_code_flagged(self):
        codings = [{"id": "2", "code": "9", "confidence": "high"}]
        result = validate_option_codes(codings, self.codes_by_var, self.variables)
        assert result[0]["_invalid"] is True
        assert "0" in result[0]["_valid_codes"]

    def test_non_option_variable_ignored(self):
        variables = [{"ID": "28", "Datatype": "Float"}]
        codings = [{"id": "28", "code": "999"}]
        result = validate_option_codes(codings, {}, variables)
        assert "_invalid" not in result[0]


# ---------------------------------------------------------------------------
# code_traits: build_review_message
# ---------------------------------------------------------------------------

class TestBuildReviewMessage:
    def test_no_review_needed(self):
        codings = [{"id": "1", "code": "0", "confidence": "high"}]
        msg, n_inv, n_low = build_review_message(codings)
        assert msg is None
        assert n_inv == 0
        assert n_low == 0

    def test_invalid_codes(self):
        codings = [{"id": "1", "code": "9", "_invalid": True, "_valid_codes": ["0", "1"]}]
        msg, n_inv, n_low = build_review_message(codings)
        assert msg is not None
        assert n_inv == 1
        assert "invalid codes" in msg
        assert "0, 1" in msg

    def test_low_confidence(self):
        codings = [{"id": "1", "code": "0", "confidence": "low"}]
        msg, n_inv, n_low = build_review_message(codings)
        assert msg is not None
        assert n_low == 1
        assert "low or absent confidence" in msg

    def test_absent_confidence(self):
        codings = [{"id": "1", "code": "0", "confidence": "absent"}]
        msg, n_inv, n_low = build_review_message(codings)
        assert msg is not None
        assert n_low == 1

    def test_invalid_not_counted_as_low(self):
        codings = [{"id": "1", "code": "9", "confidence": "low",
                     "_invalid": True, "_valid_codes": ["0", "1"]}]
        msg, n_inv, n_low = build_review_message(codings)
        assert n_inv == 1
        assert n_low == 0


# ---------------------------------------------------------------------------
# evaluate: normalize_code
# ---------------------------------------------------------------------------

class TestNormalizeCode:
    def test_none(self):
        assert normalize_code(None) == ""

    def test_integer_string(self):
        assert normalize_code("1") == "1"

    def test_float_to_int(self):
        assert normalize_code("1.0") == "1"

    def test_actual_float(self):
        assert normalize_code("3.5") == "3.5"

    def test_text(self):
        assert normalize_code("some text") == "some text"

    def test_int_value(self):
        assert normalize_code(2) == "2"

    def test_whitespace(self):
        assert normalize_code("  3  ") == "3"


# ---------------------------------------------------------------------------
# evaluate: load_codings / load_codings_as_dict
# ---------------------------------------------------------------------------

class TestLoadCodings:
    def _write(self, data: str) -> Path:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        f.write(data)
        f.close()
        return Path(f.name)

    def test_bare_list(self):
        path = self._write(json.dumps([{"id": 1, "code": "0"}]))
        assert load_codings(path) == [{"id": 1, "code": "0"}]

    def test_codings_wrapper(self):
        path = self._write(json.dumps({"codings": [{"id": 1, "code": "0"}]}))
        assert load_codings(path) == [{"id": 1, "code": "0"}]

    def test_with_fences(self):
        inner = json.dumps({"codings": [{"id": 1, "code": "0"}]})
        path = self._write(f"```json\n{inner}\n```")
        assert load_codings(path) == [{"id": 1, "code": "0"}]

    def test_double_brace(self):
        inner = json.dumps({"codings": [{"id": 1, "code": "0"}]})
        path = self._write("{" + inner)
        assert load_codings(path) == [{"id": 1, "code": "0"}]

    def test_raw_response(self):
        inner = json.dumps({"codings": [{"id": 1, "code": "0"}]})
        path = self._write(json.dumps({"raw_response": inner}))
        assert load_codings(path) == [{"id": 1, "code": "0"}]


class TestLoadCodingsAsDict:
    def _write(self, data) -> Path:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        f.write(json.dumps(data))
        f.close()
        return Path(f.name)

    def test_basic(self):
        path = self._write({"codings": [{"id": 1, "code": "0"}, {"id": 2, "code": "1"}]})
        assert load_codings_as_dict(path) == {"1": "0", "2": "1"}

    def test_skips_nulls(self):
        path = self._write([{"id": 1, "code": "0"}, {"id": 2, "code": None}])
        result = load_codings_as_dict(path)
        assert "1" in result
        assert "2" not in result

    def test_variable_key(self):
        path = self._write([{"variable": 5, "code": "2"}])
        assert load_codings_as_dict(path) == {"5": "2"}

    def test_normalizes_codes(self):
        path = self._write([{"id": 1, "code": "1.0"}])
        assert load_codings_as_dict(path) == {"1": "1"}


# ---------------------------------------------------------------------------
# setup_dataset: strip_pages / parse_sources
# ---------------------------------------------------------------------------

class TestStripPages:
    def test_with_pages(self):
        assert strip_pages("buck1952[39-41]") == "buck1952"

    def test_without_pages(self):
        assert strip_pages("buck1952") == "buck1952"

    def test_multiple_brackets(self):
        assert strip_pages("smith2000[1-5][10]") == "smith2000"

    def test_whitespace(self):
        assert strip_pages("  buck1952[39]  ") == "buck1952"



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
