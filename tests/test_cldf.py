from ethnocoder.datasets.cldf import build_source_index, codings_for_source, strip_pages


class TestStripPages:
    def test_removes_page_range(self):
        assert strip_pages("smith1990[23-45]") == "smith1990"

    def test_no_pages_passthrough(self):
        assert strip_pages("smith1990") == "smith1990"

    def test_strips_surrounding_whitespace(self):
        assert strip_pages("  smith1990 [1] ") == "smith1990"


class TestBuildSourceIndex:
    def test_groups_rows_by_stripped_key(self):
        rows = [
            {"Source": ["a[1]"], "x": 1, "Language_ID": "abcd9876"},
            {"Source": ["a[2]"], "x": 2, "Language_ID": "abce1234"},
            {"Source": ["b"], "x": 3, "Language_ID": "abcd9876"},
        ]
        index = build_source_index(rows)
        assert set(index) == {"abcd9876_a", "abce1234_a", "abcd9876_b"}
        assert len(index["abcd9876_a"]) == 1
        assert len(index["abce1234_a"]) == 1

    def test_skips_empty_keys(self):
        # A page-only source strips to "" and is dropped.
        assert build_source_index([{"Source": ["[1-2]"]}]) == {}

    def test_row_with_multiple_sources(self):
        index = build_source_index([{"Source": ["a", "b"], "x": 1, "Language_ID": "abcd9876"}])
        assert "abcd9876_a" in index and "abcd9876_b" in index


class TestCodingsForSource:
    VARS = [{"ID": "1"}, {"ID": "2"}, {"ID": "3"}]
    CODE_NAMES = {"c10": "present"}

    def test_code_id_resolved_to_name(self):
        rows = [{"Parameter_ID": "1", "Code_ID": "c10", "Value": None}]
        result = codings_for_source(rows, self.VARS, self.CODE_NAMES)
        assert {"id": "1", "code": "present", 'comment': None, 'source': None, 'source_comment': None, 'coders': None} in result

    def test_unknown_code_id_passthrough(self):
        rows = [{"Parameter_ID": "1", "Code_ID": "cXX", "Value": None}]
        result = codings_for_source(rows, self.VARS, self.CODE_NAMES)
        assert {"id": "1", "code": "cXX", 'comment': None, 'source': None, 'source_comment': None, 'coders': None} in result

    def test_value_fallback_when_no_code_id(self):
        rows = [{"Parameter_ID": "2", "Code_ID": None, "Value": "42"}]
        result = codings_for_source(rows, self.VARS, self.CODE_NAMES)
        assert {"id": "2", "code": "42", 'comment': None, 'source': None, 'source_comment': None, 'coders': None} in result

    def test_missing_param_is_none(self):
        result = codings_for_source([], self.VARS, self.CODE_NAMES)
        assert all(c["code"] is None for c in result)
        assert [c["id"] for c in result] == ["1", "2", "3"]
