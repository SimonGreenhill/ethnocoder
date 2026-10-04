from ethnocoder.summarise import compare


class TestCompare:
    def test_match_and_mismatch(self):
        gold = {"1": "a", "2": "b"}
        coded = {"1": "a", "2": "x"}
        assert compare(gold, coded) == (1, 2)

    def test_missing_coded_counts_as_diff(self):
        gold = {"1": "a", "2": "b"}
        coded = {"1": "a"}
        assert compare(gold, coded) == (1, 2)

    def test_all_match(self):
        gold = {"1": "a", "2": "b"}
        assert compare(gold, dict(gold)) == (2, 2)

    def test_empty_gold(self):
        assert compare({}, {"1": "a"}) == (0, 0)
