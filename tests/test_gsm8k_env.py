from gsm8k_env import parse_final_answer


def test_parse_hash_delimiter():
    out = "Reasoning...\n#### 1,234"
    assert parse_final_answer(out) == "1234"


def test_parse_fallback_last_number():
    out = "First 10 then maybe 20, final answer is 30"
    assert parse_final_answer(out) == "30"


def test_parse_currency_and_decimal():
    out = "Total is $40.00"
    assert parse_final_answer(out) == "40"
