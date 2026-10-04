import json

from ethnocoder.evaluate import load_codings, load_codings_as_dict, normalize_code


class TestNormalizeCode:
    def test_float_with_zero_fraction_becomes_int(self):
        assert normalize_code("2.0") == "2"

    def test_float_with_fraction_preserved(self):
        assert normalize_code("2.5") == "2.5"

    def test_none_becomes_empty(self):
        assert normalize_code(None) == ""

    def test_non_numeric_passthrough(self):
        assert normalize_code("present") == "present"

    def test_whitespace_stripped(self):
        assert normalize_code("  3  ") == "3"

    def test_numeric_type_input(self):
        assert normalize_code(4.0) == "4"


class TestLoadCodingsAsDict:
    def test_round_trip(self, tmp_path):
        p = tmp_path / "doc.json"
        p.write_text(json.dumps({"codings": [{"id": "1", "code": "2.0"}]}))
        assert load_codings_as_dict(p) == {"1": "2"}

    def test_variable_key_fallback(self, tmp_path):
        p = tmp_path / "doc.json"
        p.write_text(json.dumps({"codings": [{"variable": "7", "code": "x"}]}))
        assert load_codings_as_dict(p) == {"7": "x"}

    def test_null_code_dropped(self, tmp_path):
        p = tmp_path / "doc.json"
        p.write_text(json.dumps({"codings": [{"id": "1", "code": None}, {"id": "2", "code": "a"}]}))
        assert load_codings_as_dict(p) == {"2": "a"}


class TestLoadCodings:
    def test_codings_wrapper_from_file(self, tmp_path):
        p = tmp_path / "doc.json"
        p.write_text(json.dumps({"codings": [{"id": 1, "code": "0"}]}))
        assert load_codings(p) == [{"id": 1, "code": "0"}]

    def test_fenced_file(self, tmp_path):
        p = tmp_path / "doc.json"
        inner = json.dumps({"codings": [{"id": 1, "code": "0"}]})
        p.write_text(f"```json\n{inner}\n```")
        assert load_codings(p) == [{"id": 1, "code": "0"}]
