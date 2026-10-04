from ethnocoder.code import (
    build_coding_prompt,
    build_review_message,
    clean_codings,
    model_dirname,
    resolve_model_config,
    validate_option_codes,
)


class TestResolveModelConfig:
    def test_claude_is_anthropic(self):
        assert resolve_model_config("claude-3-5-sonnet", None) == (True, None)

    def test_openai_prefix_not_anthropic(self):
        # "claude" in the name but routed via openai/ → not treated as anthropic.
        assert resolve_model_config("openai/claude-proxy", None) == (False, None)

    def test_non_claude_not_anthropic(self):
        assert resolve_model_config("gpt-4", None) == (False, None)

    def test_lm_studio_gets_default_api_base(self):
        assert resolve_model_config("lm_studio/foo", None) == (False, "http://localhost:1234/v1")

    def test_explicit_api_base_not_overwritten(self):
        assert resolve_model_config("lm_studio/foo", "http://x") == (False, "http://x")


class TestModelDirname:
    def test_strips_provider_prefix(self):
        assert model_dirname("openai/gpt-4") == "gpt-4"

    def test_bare_name(self):
        assert model_dirname("gpt-4") == "gpt-4"

    def test_takes_last_segment(self):
        assert model_dirname("lm_studio/org/model") == "model"


class TestCleanCodings:
    def test_drops_underscore_keys(self):
        codings = [{"id": "1", "code": "a", "_invalid": True, "_valid_codes": ["a"]}]
        assert clean_codings(codings) == [{"id": "1", "code": "a"}]

    def test_keeps_normal_keys(self):
        codings = [{"id": "1", "confidence": "high"}]
        assert clean_codings(codings) == [{"id": "1", "confidence": "high"}]


class TestValidateOptionCodes:
    VARIABLES = [
        {"ID": "1", "Datatype": "Option"},
        {"ID": "2", "Datatype": "Int"},
    ]
    CODES = {"1": [{"Name": "a"}, {"Name": "b"}]}

    def test_valid_code_unchanged(self):
        result = validate_option_codes([{"id": "1", "code": "a"}], self.CODES, self.VARIABLES)
        assert result == [{"id": "1", "code": "a"}]

    def test_invalid_code_flagged(self):
        result = validate_option_codes([{"id": "1", "code": "z"}], self.CODES, self.VARIABLES)
        assert result[0]["_invalid"] is True
        assert result[0]["_valid_codes"] == ["a", "b"]

    def test_non_option_var_ignored(self):
        result = validate_option_codes([{"id": "2", "code": "5"}], self.CODES, self.VARIABLES)
        assert result == [{"id": "2", "code": "5"}]

    def test_unknown_var_id_ignored(self):
        result = validate_option_codes([{"id": "99", "code": "x"}], self.CODES, self.VARIABLES)
        assert result == [{"id": "99", "code": "x"}]


class TestBuildReviewMessage:
    def test_all_clean_returns_none(self):
        codings = [{"id": "1", "confidence": "high"}]
        assert build_review_message(codings) == (None, 0, 0)

    def test_invalid_counted(self):
        codings = [{"id": "1", "code": "z", "_invalid": True, "_valid_codes": ["a", "b"]}]
        msg, n_invalid, n_low = build_review_message(codings)
        assert n_invalid == 1 and n_low == 0
        assert msg is not None and "a, b" in msg

    def test_low_confidence_counted(self):
        codings = [{"id": "1", "confidence": "low"}]
        _, n_invalid, n_low = build_review_message(codings)
        assert n_invalid == 0 and n_low == 1

    def test_absent_confidence_counted_as_low(self):
        _, _, n_low = build_review_message([{"id": "1", "confidence": "absent"}])
        assert n_low == 1

    def test_invalid_not_double_counted_as_low(self):
        codings = [{"id": "1", "code": "z", "confidence": "low", "_invalid": True, "_valid_codes": ["a"]}]
        _, n_invalid, n_low = build_review_message(codings)
        assert n_invalid == 1 and n_low == 0


class TestBuildCodingPrompt:
    def test_option_lists_sorted_valid_codes(self):
        variables = [{"ID": "1", "Name": "Subsistence", "Datatype": "Option", "Description": ""}]
        codes = {"1": [{"Name": "b", "Description": "beta"}, {"Name": "a", "Description": "alpha"}]}
        prompt = build_coding_prompt(variables, codes)
        assert "Valid codes: a=alpha | b=beta" in prompt
        assert "Assign one code from the list above." in prompt

    def test_description_truncated_to_100_chars(self):
        variables = [{"ID": "1", "Name": "V", "Datatype": "Option", "Description": ""}]
        codes = {"1": [{"Name": "a", "Description": "x" * 150}]}
        prompt = build_coding_prompt(variables, codes)
        assert "a=" + "x" * 100 in prompt
        assert "x" * 101 not in prompt

    def test_int_datatype_instruction(self):
        variables = [{"ID": "1", "Name": "Count", "Datatype": "Int", "Description": ""}]
        assert "Assign an integer value." in build_coding_prompt(variables, {})

    def test_float_datatype_instruction(self):
        variables = [{"ID": "1", "Name": "Ratio", "Datatype": "Float", "Description": ""}]
        assert "Assign a numeric value." in build_coding_prompt(variables, {})

    def test_text_datatype_instruction(self):
        variables = [{"ID": "1", "Name": "Note", "Datatype": "Text", "Description": ""}]
        assert "Provide a brief text value." in build_coding_prompt(variables, {})

    def test_section_header_once_per_section(self):
        variables = [
            {"ID": "1", "Name": "A", "Datatype": "Int", "Description": "", "Section": "S1"},
            {"ID": "2", "Name": "B", "Datatype": "Int", "Description": "", "Section": "S1"},
            {"ID": "3", "Name": "C", "Datatype": "Int", "Description": "", "Section": "S2"},
        ]
        prompt = build_coding_prompt(variables, {})
        assert prompt.count("[S1]") == 1
        assert prompt.count("[S2]") == 1

    def test_no_description_line_when_blank(self):
        variables = [{"ID": "1", "Name": "V", "Datatype": "Int", "Description": ""}]
        prompt = build_coding_prompt(variables, {})
        assert "ID 1: V" in prompt
