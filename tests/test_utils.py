from ethnocoder.utils import parse_codings, strip_fences


class TestStripFences:
    def test_no_fences_passthrough(self):
        assert strip_fences('{"a": 1}') == '{"a": 1}'

    def test_strips_surrounding_whitespace(self):
        assert strip_fences('  {"a": 1}  ') == '{"a": 1}'

    def test_strips_json_fence(self):
        assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_strips_bare_fence(self):
        assert strip_fences('```\n[1, 2]\n```') == '[1, 2]'

    def test_strips_fence_with_leading_whitespace(self):
        assert strip_fences('\n\n```json\n{"a": 1}\n```\n') == '{"a": 1}'

    def test_opening_fence_without_closing(self):
        assert strip_fences('```json\n{"a": 1}') == '{"a": 1}'

    def test_closing_fence_without_opening(self):
        assert strip_fences('{"a": 1}\n```') == '{"a": 1}'


class TestParseCodings:
    def test_bare_list(self):
        assert parse_codings('[{"id": "1", "code": "a"}]') == [{"id": "1", "code": "a"}]

    def test_codings_wrapper(self):
        assert parse_codings('{"codings": [{"id": "1"}]}') == [{"id": "1"}]

    def test_dict_without_codings_returns_empty(self):
        assert parse_codings('{"other": 1}') == []

    def test_double_brace_bug(self):
        # Leading "{{" from a malformed LLM response is repaired to a single brace.
        assert parse_codings('{{"codings": [{"id": "1"}]}') == [{"id": "1"}]

    def test_raw_response_wrapper(self):
        text = '{"raw_response": "```json\\n[{\\"id\\": \\"1\\"}]\\n```"}'
        assert parse_codings(text) == [{"id": "1"}]

    def test_raw_response_ignored_when_codings_present(self):
        assert parse_codings('{"codings": [{"id": "9"}], "raw_response": "junk"}') == [{"id": "9"}]
