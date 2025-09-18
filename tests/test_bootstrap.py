from quilleval.bootstrap import bootstrap_ci
def test_bootstrap_basic():
    lo, hi = bootstrap_ci([1,2,3,4,5], iterations=200)
    assert lo <= hi
