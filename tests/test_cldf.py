from ethnocoder.cldf import build_source_index, codings_for_source, strip_pages


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
            {"Source": ["a[1]"], "x": 1},
            {"Source": ["a[2]"], "x": 2},
            {"Source": ["b"], "x": 3},
        ]
        index = build_source_index(rows)
        assert set(index) == {"a", "b"}
        assert len(index["a"]) == 2

    def test_skips_empty_keys(self):
        # A page-only source strips to "" and is dropped.
        assert build_source_index([{"Source": ["[1-2]"]}]) == {}

    def test_row_with_multiple_sources(self):
        index = build_source_index([{"Source": ["a", "b"], "x": 1}])
        assert "a" in index and "b" in index


class TestCodingsForSource:
    VARS = [{"ID": "1"}, {"ID": "2"}, {"ID": "3"}]
    CODE_NAMES = {"c10": "present"}

    def test_code_id_resolved_to_name(self):
        rows = [{"Parameter_ID": "1", "Code_ID": "c10", "Value": None}]
        result = codings_for_source(rows, self.VARS, self.CODE_NAMES)
        assert {"id": "1", "code": "present"} in result

    def test_unknown_code_id_passthrough(self):
        rows = [{"Parameter_ID": "1", "Code_ID": "cXX", "Value": None}]
        result = codings_for_source(rows, self.VARS, self.CODE_NAMES)
        assert {"id": "1", "code": "cXX"} in result

    def test_value_fallback_when_no_code_id(self):
        rows = [{"Parameter_ID": "2", "Code_ID": None, "Value": "42"}]
        result = codings_for_source(rows, self.VARS, self.CODE_NAMES)
        assert {"id": "2", "code": "42"} in result

    def test_missing_param_is_none(self):
        result = codings_for_source([], self.VARS, self.CODE_NAMES)
        assert all(c["code"] is None for c in result)
        assert [c["id"] for c in result] == ["1", "2", "3"]
