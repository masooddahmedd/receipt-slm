from slm.cord import canonical, is_correct, parse_truth, rows_to_text, target_json, to_amount
from slm.evaluate import parse

GT = {
    "menu": [{"nm": "EGG TART", "cnt": "2", "price": "26.000",
              "sub": {"nm": "EXTRA CREAM", "cnt": "1", "price": "3.000"}},
             {"nm": "ICE TEA", "cnt": "1", "price": "8.000"}],
    "sub_total": {"subtotal_price": "37.000", "tax_price": "3.700"},
    "total": {"total_price": "40.700"},
}


def test_amounts_keep_indonesian_format():
    assert to_amount("45.000") == 45000
    assert to_amount("Rp 1,250") == 1250
    assert to_amount(None) is None


def test_target_keeps_printed_strings_and_flattens_sub_items():
    t = target_json(GT)
    assert [i["name"] for i in t["line_items"]] == ["EGG TART", "EXTRA CREAM", "ICE TEA"]
    assert t["subtotal"] == "37.000" and t["total"] == "40.700"


def test_prediction_scored_like_receipt_cascade():
    truth = parse_truth(GT)
    pred = {"line_items": [{"name": "Ice Tea", "quantity": 1, "price": "8,000"},
                           {"name": "EGG TAR", "quantity": 2, "price": "26.000"},
                           {"name": "EXTRA CREAM", "quantity": 1, "price": "3.000"}],
            "subtotal": "37.000", "tax": "3,700", "total": "40.700"}
    for f in ("line_items", "subtotal", "tax", "total"):
        assert is_correct(f, canonical(pred, f), truth), f


def test_wrong_quantity_fails_line_items():
    truth = parse_truth(GT)
    pred = target_json(GT)
    pred["line_items"][0]["quantity"] = 1
    assert not is_correct("line_items", canonical(pred, "line_items"), truth)


def test_rows_join_boxes_on_same_line():
    boxes = [[[[0, 0], [10, 0], [10, 10], [0, 10]], "TEA"], [[[50, 1], [60, 1], [60, 11], [50, 11]], "8.000"],
             [[[0, 30], [10, 30], [10, 40], [0, 40]], "TOTAL"]]
    assert rows_to_text(boxes) == "TEA    8.000\nTOTAL"


def test_parse_handles_code_fences():
    assert parse('```json\n{"total": "1"}\n```') == {"total": "1"}
    assert parse("not json") is None
