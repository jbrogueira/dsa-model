"""The general government accounts file (data/gov_accounts_GR.json): it loads,
and its 2023 totals, cash benefits and health purchases are those of
docs/BUDGET_ALIGNMENT_PLAN.md section 2."""
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, '..', 'data', 'gov_accounts_GR.json')


@pytest.fixture(scope='module')
def acc():
    if not os.path.exists(PATH):
        pytest.skip('data/gov_accounts_GR.json not present; run build_gov_accounts_GR.py')
    return json.load(open(PATH))


def test_file_loads_with_the_expected_structure(acc):
    for key in ('source', 'extracted', 'unit', 'years', 'revenue', 'expenditure', 'cofog'):
        assert key in acc
    assert acc['unit'] == 'percent of GDP'
    assert 2023 in acc['years']
    for block in ('revenue', 'expenditure'):
        for item in acc[block].values():
            assert set(item) >= {'label', 'model_line', 'values'}
            assert all(isinstance(v, float) for v in item['values'].values())


def test_total_revenue_2023(acc):
    assert acc['revenue']['TR']['values']['2023'] == pytest.approx(48.1, abs=0.3)


def test_total_expenditure_2023(acc):
    assert acc['expenditure']['TE']['values']['2023'] == pytest.approx(49.5, abs=0.3)


def test_cash_social_benefits_2023(acc):
    assert acc['expenditure']['D62']['values']['2023'] == pytest.approx(17.3, abs=0.3)


def test_health_purchases_2023(acc):
    """Compensation of employees, intermediate consumption and purchased
    social transfers in kind of the health function (COFOG 07), 2023."""
    gf07 = acc['cofog']['GF07']
    purchases = sum(gf07[item]['2023'] for item in ('D1', 'P2', 'D632'))
    assert purchases == pytest.approx(5.6, abs=0.3)
