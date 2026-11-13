import json
import argparse

import pytest

import ethnocoder.models.code as code_mod
from ethnocoder.models.code import Coder

# (
#     build_coding_prompt,
#     build_review_message,
#     clean_codings,
#     code_document,
#     estimate_tokens,
#     model_dirname,
#     pack_batches,
#     resolve_model_config,
#     validate_option_codes,
# )


class TestResolveModelConfig:

    @classmethod
    def setup_class(cls):

        # Define args for coder object
        args = argparse.Namespace()

        # Add fake model, language name, lgcode
        args.model = "fake-model"
        args.language = "fake-language"
        args.lgcode = "fake-lgcode"
        args.text = "path/to/fake/file"

        # Create Coder object
        cls.coder = Coder(args=args)
    
    def test_claude_is_anthropic(self):
        assert self.coder.resolve_model_config("claude-3-5-sonnet", None) == (True, None)

    def test_openai_prefix_not_anthropic(self):
        # "claude" in the name but routed via openai/ → not treated as anthropic.
        assert self.coder.resolve_model_config("openai/claude-proxy", None) == (False, None)

    def test_non_claude_not_anthropic(self):
        assert self.coder.resolve_model_config("gpt-4", None) == (False, None)

    def test_lm_studio_gets_default_api_base(self):
        assert self.coder.resolve_model_config("lm_studio/foo", None) == (False, "http://localhost:1234/v1")

    def test_explicit_api_base_not_overwritten(self):
        assert self.coder.resolve_model_config("lm_studio/foo", "http://x") == (False, "http://x")


class TestModelDirname:

    @classmethod
    def setup_class(cls):

        # Define args for coder object
        args = argparse.Namespace()

        # Add fake model, language name, lgcode
        args.model = "fake-model"
        args.language = "fake-language"
        args.lgcode = "fake-lgcode"
        args.text = "path/to/fake/file"

        # Create Coder object
        cls.coder = Coder(args=args)

    def test_strips_provider_prefix(self):
        assert "openai/" not in self.coder.model_dirname("openai/gpt-4")

    def test_bare_name(self):
        assert "gpt-4" in self.coder.model_dirname("gpt-4")

    def test_takes_last_segment(self):
        assert "org" not in self.coder.model_dirname("lm_studio/org/model")


class TestCleanCodings:

    @classmethod
    def setup_class(cls):

        # Define args for coder object
        args = argparse.Namespace()

        # Add fake model, language name, lgcode
        args.model = "fake-model"
        args.language = "fake-language"
        args.lgcode = "fake-lgcode"
        args.text = "path/to/fake/file"

        # Create Coder object
        cls.coder = Coder(args=args)

    def test_drops_underscore_keys(self):
        codings = [{"id": "1", "code": "a", "_invalid": True, "_valid_codes": ["a"]}]
        assert self.coder.clean_codings(codings) == [{"id": "1", "code": "a"}]

    def test_keeps_normal_keys(self):
        codings = [{"id": "1", "confidence": "high"}]
        assert self.coder.clean_codings(codings) == [{"id": "1", "confidence": "high"}]


# class TestValidateOptionCodes:

#     @classmethod
#     def setup_class(cls):

#         # Define args for coder object
#         args = argparse.Namespace()

#         # Add fake model, language name, lgcode
#         args.model = "fake-model"
#         args.language_name = "fake-language"
#         args.lgcode = "fake-lgcode"

#         # Create Coder object
#         cls.coder = Coder(args=args)

#     VARIABLES = [
#         {"ID": "1", "Datatype": "Option"},
#         {"ID": "2", "Datatype": "Int"},
#     ]
#     CODES = {"1": [{"Name": "a"}, {"Name": "b"}]}

#     def test_valid_code_unchanged(self):
#         result = self.coder.validate_option_codes([{"id": "1", "code": "a"}], self.CODES, self.VARIABLES)
#         assert result == [{"id": "1", "code": "a"}]

#     def test_invalid_code_flagged(self):
#         result = self.coder.validate_option_codes([{"id": "1", "code": "z"}], self.CODES, self.VARIABLES)
#         assert result[0]["_invalid"] is True
#         assert result[0]["_valid_codes"] == ["a", "b"]

#     def test_non_option_var_ignored(self):
#         result = self.coder.validate_option_codes([{"id": "2", "code": "5"}], self.CODES, self.VARIABLES)
#         assert result == [{"id": "2", "code": "5"}]

#     def test_unknown_var_id_ignored(self):
#         result = self.coder.validate_option_codes([{"id": "99", "code": "x"}], self.CODES, self.VARIABLES)
#         assert result == [{"id": "99", "code": "x"}]

#     def test_null_code_not_flagged(self):
#         result = self.coder.validate_option_codes([{"id": "1", "code": None}], self.CODES, self.VARIABLES)
#         assert result == [{"id": "1", "code": None}]

#     def test_empty_code_not_flagged(self):
#         result = self.coder.validate_option_codes([{"id": "1", "code": ""}], self.CODES, self.VARIABLES)
#         assert result == [{"id": "1", "code": ""}]


class TestBuildReviewMessage:

    @classmethod
    def setup_class(cls):

        # Define args for coder object
        args = argparse.Namespace()

        # Add fake model, language name, lgcode
        args.model = "fake-model"
        args.language = "fake-language"
        args.lgcode = "fake-lgcode"
        args.text = "path/to/fake/file"

        # Create Coder object
        cls.coder = Coder(args=args)

    def test_all_clean_returns_none(self):
        codings = [{"id": "1", "confidence": "high"}]
        retdict = {
            "n_missing": 0,
            "n_superfluous": 0,
            "n_duplicate": 0,
            "n_invalid": 0,
            "n_low_confidence": 0,
            "n_quote_missing": 0,
        }
        assert self.coder.build_review_message(codings) == (None, retdict)

    def test_invalid_counted(self):
        codings = [{"id": "1", "code": "z", "_invalid": True, "_valid_codes": ["a", "b"]}]
        msg, retdict = self.coder.build_review_message(codings)
        assert retdict["n_invalid"] == 1 and retdict["n_low_confidence"] == 0
        assert msg is not None and "a, b" in msg

    def test_low_confidence_counted(self):
        codings = [{"id": "1", "confidence": "low"}]
        _, retdict = self.coder.build_review_message(codings)
        assert retdict["n_invalid"] == 0 and retdict["n_low_confidence"] == 1

    # def test_absent_confidence_counted_as_low(self):
    #     _, retdict = self.coder.build_review_message([{"id": "1", "confidence": "absent"}])
    #     assert retdict["n_low_confidence"] == 1

    def test_invalid_not_double_counted_as_low(self):
        codings = [{"id": "1", "code": "z", "confidence": "low", "_invalid": True, "_valid_codes": ["a"]}]
        _, retdict = self.coder.build_review_message(codings)
        assert retdict["n_invalid"] == 1 and retdict["n_low_confidence"] == 0


# class TestEstimateTokens:
#     def test_empty(self):
#         assert estimate_tokens("") == 0

#     def test_four_chars_per_token(self):
#         assert estimate_tokens("a" * 8) == 2


# class TestPackBatches:

#     @classmethod
#     def setup_class(cls):

#         # Define args for coder object
#         args = argparse.Namespace()

#         # Add fake model, language name, lgcode
#         args.model = "fake-model"
#         args.language = "fake-language"
#         args.lgcode = "fake-lgcode"
#         args.text = "path/to/fake/file"

#         # Create Coder object
#         cls.coder = Coder(args=args)

#     def test_none_budget_single_batch(self):
#         items = [{"ID": "a",}, {"ID": "b", "Main_domain": "clause"}, {"ID": "c"}]
#         sizes = {"a": 40, "b": 40, "c": 40}
#         assert self.coder.pack_batches(
#             variables=items, 
#             var_size=sizes, 
#             prefix_tokens=10, 
#             context_budget=None,
#             response_reserve_per_var=0
#         ) == [["a", "b", "c"]]

#     def test_fits_in_one_batch(self):
#         items = ["a", "b", "c"]
#         sizes = [10, 10, 10]
#         assert self.coder.pack_batches(items, sizes, prefix_tokens=10, context_budget=100,
#                             response_reserve_per_var=0) == [["a", "b", "c"]]

#     def test_splits_into_multiple_batches(self):
#         items = ["a", "b", "c", "d"]
#         sizes = [40, 40, 40, 40]
#         # prefix 10 + 40 + 40 = 90 ok; + 40 = 130 > 100 -> split
#         assert self.coder.pack_batches(items, sizes, prefix_tokens=10, context_budget=100,
#                             response_reserve_per_var=0) == [["a", "b"], ["c", "d"]]

#     def test_reserve_counts_against_budget(self):
#         items = ["a", "b"]
#         sizes = [40, 40]
#         # effective size 50 each; 10 + 50 = 60 ok, + 50 = 110 > 100 -> one per batch
#         assert self.coder.pack_batches(items, sizes, prefix_tokens=10, context_budget=100,
#                             response_reserve_per_var=10) == [["a"], ["b"]]

#     def test_single_item_too_big_raises(self):
#         with pytest.raises(ValueError):
#             self.coder.pack_batches(
#                 variables = ["a"], var_size = {"a": 95}, prefix_tokens=10, context_budget=100,
#                          response_reserve_per_var=0)

#     def test_empty_items(self):
#         assert self.coder.pack_batches([], {}, prefix_tokens=10, context_budget=100,
#                             response_reserve_per_var=0) == []


def _fake_stream(calls):
    def stream(messages, model, is_anthropic, api_base):
        calls.append(messages)
        user = messages[-1]["content"]
        ids = [
            line.split("ID ", 1)[1].split(":")[0].strip()
            for line in user.splitlines()
            if line.startswith("ID ")
        ]
        return json.dumps({"codings": [{"id": i, "code": "1"} for i in ids]})

    return stream


def _int_vars(n, desc=""):
    return [
        {"ID": str(i), "Name": f"V{i}", "Datatype": "Int", "Description": desc}
        for i in range(n)
    ]


# class TestCodeDocument:

#     @classmethod
#     def setup_class(cls):

#         # Define args for coder object
#         args = argparse.Namespace()

#         # Add fake model, language name, lgcode
#         args.model = "fake-model"
#         args.language = "fake-language"
#         args.text = "path/to/fake/file"
#         args.lgcode = "fake-lgcode"

#         # Create Coder object
#         cls.coder = Coder(args=args)

#     def test_single_call_when_no_budget(self, tmp_path, monkeypatch):
#         calls = []
#         monkeypatch.setattr(code_mod, "llm_stream", _fake_stream(calls))
#         messages = [{"role": "system", "content": "sys"}]
#         result = self.coder.code_document_inner(
#             "stem", 
#             _int_vars(4), 
#             {}, 
#             "m", 
#             None, 
#             False, 
#             messages,
#             pdf_prefix="PDF", 
#             out_dir=tmp_path, 
#             context_budget=None,
#         )
#         assert len(calls) == 1
#         assert {c["id"] for c in result} == {"0", "1", "2", "3"}
#         saved = json.loads((tmp_path / "stem.json").read_text())
#         assert {c["id"] for c in saved["codings"]} == {"0", "1", "2", "3"}

#     def test_batches_split_and_merge(self, tmp_path, monkeypatch):
#         calls = []
#         monkeypatch.setattr(code_mod, "llm_stream", _fake_stream(calls))
#         messages = [{"role": "system", "content": "sys"}]
#         result = self.coder.code_document_inner(
#             "stem", _int_vars(4, desc="x" * 800), {}, "m", None, False, messages,
#             pdf_prefix="PDF", out_dir=tmp_path, context_budget=300,
#             response_reserve_per_var=0,
#         )
#         assert len(calls) >= 2
#         assert {c["id"] for c in result} == {"0", "1", "2", "3"}
#         assert len(result) == 4
#         assert "=== batch 1 ===" in (tmp_path / "stem.txt").read_text()

#     def test_identical_prefix_across_batches(self, tmp_path, monkeypatch):
#         calls = []
#         monkeypatch.setattr(code_mod, "llm_stream", _fake_stream(calls))
#         messages = [{"role": "system", "content": "sys"}]
#         self.coder.code_document_inner(
#             "stem", _int_vars(4, desc="x" * 800), {}, "m", None, False, messages,
#             pdf_prefix="PDF", out_dir=tmp_path, context_budget=300,
#             response_reserve_per_var=0,
#         )
#         # every batch's first message (system) and PDF prefix must match for KV reuse
#         systems = {m[0]["content"] for m in calls}
#         prefixes = {m[1]["content"].split("---")[0] for m in calls}
#         assert systems == {"sys"}
#         assert prefixes == {"PDF\n\n"}


class TestBuildCodingPrompt:

    @classmethod
    def setup_class(cls):

        # Define args for coder object
        args = argparse.Namespace()

        # Add fake model, language name, lgcode
        args.model = "fake-model"
        args.language = "fake-language"
        args.text = "path/to/fake/file"
        args.lgcode = "fake-lgcode"

        # Create Coder object
        cls.coder = Coder(args=args)

    def test_option_lists_sorted_valid_codes(self):
        variables = [{"ID": "1", "Name": "Subsistence", "Datatype": "Option", "Description": ""}]
        codes = {"1": [{"Name": "b", "Description": "beta"}, {"Name": "a", "Description": "alpha"}]}
        prompt = self.coder.build_coding_prompt(variables, codes, "doctext full")

        inprompt = "## Valid codes\na = alpha\nb = beta"

        assert inprompt in prompt
        # assert "Assign one code from the list above." in prompt

    def test_description_truncated_to_100_chars(self):
        variables = [{"ID": "1", "Name": "V", "Datatype": "Option", "Description": ""}]
        codes = {"1": [{"Name": "a", "Description": "x" * 150}]}
        prompt = self.coder.build_coding_prompt(variables, codes, "doctext full")
        assert "a = " + "x" * 100 in prompt
        assert "x" * 101 not in prompt

    def test_int_datatype_instruction(self):
        variables = [{"ID": "1", "Name": "Count", "Datatype": "Int", "Description": ""}]
        assert "Assign an integer value." in self.coder.build_coding_prompt(variables, {}, "doctext full")

    def test_float_datatype_instruction(self):
        variables = [{"ID": "1", "Name": "Ratio", "Datatype": "Float", "Description": ""}]
        assert "Assign a numeric value." in self.coder.build_coding_prompt(variables, {}, "doctext full")

    def test_text_datatype_instruction(self):
        variables = [{"ID": "1", "Name": "Note", "Datatype": "Text", "Description": ""}]
        assert "Provide a brief text value." in self.coder.build_coding_prompt(variables, {}, "doctext full")

    def test_section_header_once_per_section(self):
        variables = [
            {"ID": "1", "Name": "A", "Datatype": "Int", "Description": "", "Section": "S1"},
            {"ID": "2", "Name": "B", "Datatype": "Int", "Description": "", "Section": "S1"},
            {"ID": "3", "Name": "C", "Datatype": "Int", "Description": "", "Section": "S2"},
        ]
        prompt = self.coder.build_coding_prompt(variables, {}, "doctext full")
        assert prompt.count("[S1]") == 1
        assert prompt.count("[S2]") == 1

    def test_no_description_line_when_blank(self):
        variables = [{"ID": "1", "Name": "V", "Datatype": "Int", "Description": ""}]
        prompt = self.coder.build_coding_prompt(variables, {}, "doctext full")
        assert "# 1\nV" in prompt
