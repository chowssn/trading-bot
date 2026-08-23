"""Calculates ROIC from raw FMP financial statement components.

Approximates Bloomberg's RETURN_ON_INV_CAPITAL methodology as closely as
possible using data available from FMP's `/stable` API. This is an
approximation, not a reproduction — see the `methodology_notes` returned by
`calculate_roic()` for exactly what is included and omitted vs. Bloomberg.

Invested capital (Bloomberg approximation for Industrials/Tech):

    invested_capital = totalDebt + totalStockholdersEquity
                        + deferredTaxLiabilitiesNonCurrent

Omits Bloomberg's BS_ALLOW_DOUBTFUL_ACC_REC and BS_ACCRUED_INCOME_TAXES
adjustments, which are not exposed as discrete fields by FMP.

NOPAT (Bloomberg approximation):

    effective_tax_rate = clamp(incomeTaxExpense / incomeBeforeTax, 0%, 50%)
    nopat = operatingIncome * (1 - effective_tax_rate)

Bloomberg's own build starts from net income before extraordinary items,
adds back after-tax interest expense, and further adjusts for pension and
FX effects. Using operatingIncome implicitly captures the interest add-back
(interest expense sits below the operating line) but omits the pension/FX
adjustments, which FMP does not expose.

Usage:
    from equity.tools.roic_calculator import calculate_roic
    result = calculate_roic("MSFT", api_key, periods=5)

Note on FMP subscription limits: this account's `/stable` plan caps the
`limit` query parameter at `FMP_STATEMENT_LIMIT_CAP` (currently 5) records
for the balance-sheet/income/cash-flow statement endpoints, and `page`
does not return older records beyond that cap on this tier (confirmed by
inspecting the response — `page=1` returns the identical 5 most-recent
records as `page=0`). Since computing N years of average invested capital
needs N+1 years of balance sheets, `periods=5` (the default) can only
produce 4 years of averaged invested capital / annual ROIC on this plan —
`calculate_roic()` degrades gracefully (uses however many years it can
compute) and records the shortfall in `methodology_notes` rather than
raising, unless zero usable years are available.
"""

import logging

import requests

logger = logging.getLogger(__name__)

FMP_BASE_URL = "https://financialmodelingprep.com/stable"
REQUEST_TIMEOUT_SECONDS = 15
MUSD = 1_000_000.0  # divide raw FMP dollar figures by this to get $M
FMP_STATEMENT_LIMIT_CAP = 5  # this account's subscription tier caps `limit` at 5 for statement endpoints


def _fmp_get_list(endpoint: str, ticker: str, api_key: str, limit: int, period: str | None = None) -> list[dict]:
    """Fetch up to `limit` records for `ticker` from an FMP `/stable` statement endpoint.

    Returns an empty list (and logs a warning) on any request failure or on
    an empty/unexpected response.
    """
    url = f"{FMP_BASE_URL}/{endpoint}"
    params = {"symbol": ticker, "limit": limit, "apikey": api_key}
    if period is not None:
        params["period"] = period
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.warning("FMP %s request failed for %s: %s", endpoint, ticker, exc)
        return []

    if not isinstance(data, list):
        logger.warning("FMP %s returned unexpected payload for %s: %r", endpoint, ticker, data)
        return []
    return data


def _sort_desc_by_date(records: list[dict]) -> list[dict]:
    """Sort statement records most-recent-first by their `date` field.

    FMP's `/stable` statement endpoints already return most-recent-first,
    but we don't rely on that implicitly — records without a usable date
    sort last.
    """
    return sorted(records, key=lambda r: r.get("date") or "", reverse=True)


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _invested_capital_raw(balance_sheet: dict) -> float | None:
    """Bloomberg-approximation invested capital, in raw (unscaled) dollars."""
    total_debt = _safe_float(balance_sheet.get("totalDebt"))
    total_equity = _safe_float(balance_sheet.get("totalStockholdersEquity"))
    if total_debt is None or total_equity is None:
        return None
    deferred_tax_liab = _safe_float(balance_sheet.get("deferredTaxLiabilitiesNonCurrent")) or 0.0
    return total_debt + total_equity + deferred_tax_liab


def _effective_tax_rate(tax_expense, pretax_income) -> float:
    """incomeTaxExpense / incomeBeforeTax, clamped to [0, 0.50]. 0.0 if undefined."""
    tax_expense_f = _safe_float(tax_expense)
    pretax_income_f = _safe_float(pretax_income)
    if tax_expense_f is None or not pretax_income_f:
        return 0.0
    rate = tax_expense_f / pretax_income_f
    return max(0.0, min(0.50, rate))


def _nopat_raw(operating_income, tax_rate: float) -> float | None:
    """NOPAT in raw (unscaled) dollars, or None if operatingIncome is missing."""
    operating_income_f = _safe_float(operating_income)
    if operating_income_f is None:
        return None
    return operating_income_f * (1 - tax_rate)


def calculate_roic(ticker: str, api_key: str, periods: int = 5) -> dict:
    """Calculate ROIC for `ticker` from raw FMP financial statements.

    Fetches `periods + 1` years of annual balance sheets (for year-over-year
    averaging), `periods` years of annual income statements, and `periods`
    years of annual cash flow statements (fetched for auditability per the
    Bloomberg-comparison spec; not consumed by the ROIC/NOPAT/invested-capital
    formulas below — see `methodology_notes`).

    `roic_current` uses trailing-twelve-month (T12M) operatingIncome /
    incomeTaxExpense / incomeBeforeTax — summed from the 4 most recent
    quarterly income statements when FMP has quarterly data, falling back to
    the most recent annual period otherwise — divided by the average
    invested capital from the two most recent annual balance sheets.
    `roic_5y_avg` is the simple average of `periods` annual ROIC values, each
    computed as annual NOPAT over that year's average invested capital.

    Raises RuntimeError if FMP returns insufficient data to compute any
    annual ROIC value (e.g. unknown ticker, no statement history).
    """
    if not api_key:
        raise ValueError("api_key is required")
    if periods < 1:
        raise ValueError("periods must be >= 1")

    balance_sheets = _sort_desc_by_date(
        _fmp_get_list(
            "balance-sheet-statement", ticker, api_key,
            limit=min(periods + 1, FMP_STATEMENT_LIMIT_CAP), period="annual",
        )
    )
    income_statements = _sort_desc_by_date(
        _fmp_get_list(
            "income-statement", ticker, api_key,
            limit=min(periods, FMP_STATEMENT_LIMIT_CAP), period="annual",
        )
    )
    cash_flow_statements = _sort_desc_by_date(
        _fmp_get_list(
            "cash-flow-statement", ticker, api_key,
            limit=min(periods, FMP_STATEMENT_LIMIT_CAP), period="annual",
        )
    )

    if len(balance_sheets) < 2 or not income_statements:
        raise RuntimeError(
            f"Insufficient FMP data for {ticker}: got {len(balance_sheets)} balance sheet(s), "
            f"{len(income_statements)} income statement(s) — need at least 2 and 1 respectively"
        )

    n_years = min(periods, len(income_statements), len(balance_sheets) - 1)
    annual_records = []
    for i in range(n_years):
        bs_curr, bs_prior, inc = balance_sheets[i], balance_sheets[i + 1], income_statements[i]

        ic_curr_raw = _invested_capital_raw(bs_curr)
        ic_prior_raw = _invested_capital_raw(bs_prior)
        if ic_curr_raw is None or ic_prior_raw is None:
            logger.warning("Skipping %s period %s: missing totalDebt/totalStockholdersEquity", ticker, inc.get("date"))
            continue

        avg_ic_raw = (ic_curr_raw + ic_prior_raw) / 2
        if not avg_ic_raw:
            logger.warning("Skipping %s period %s: zero average invested capital", ticker, inc.get("date"))
            continue

        tax_rate = _effective_tax_rate(inc.get("incomeTaxExpense"), inc.get("incomeBeforeTax"))
        nopat_raw = _nopat_raw(inc.get("operatingIncome"), tax_rate)
        if nopat_raw is None:
            logger.warning("Skipping %s period %s: missing operatingIncome", ticker, inc.get("date"))
            continue

        annual_records.append({
            "date": inc.get("date") or bs_curr.get("date"),
            "invested_capital_musd": ic_curr_raw / MUSD,
            "invested_capital_prior_musd": ic_prior_raw / MUSD,
            "avg_invested_capital_musd": avg_ic_raw / MUSD,
            "operating_income_musd": _safe_float(inc.get("operatingIncome")) / MUSD,
            "income_tax_expense_musd": (_safe_float(inc.get("incomeTaxExpense")) or 0.0) / MUSD,
            "income_before_tax_musd": (_safe_float(inc.get("incomeBeforeTax")) or 0.0) / MUSD,
            "effective_tax_rate_pct": tax_rate * 100,
            "nopat_musd": nopat_raw / MUSD,
            "roic_pct": 100 * nopat_raw / avg_ic_raw,
            "raw_balance_sheet_current": bs_curr,
            "raw_balance_sheet_prior": bs_prior,
            "raw_income_statement": inc,
        })

    if not annual_records:
        raise RuntimeError(f"Could not compute any annual ROIC value for {ticker} — check FMP data availability")

    roic_5y_avg = sum(r["roic_pct"] for r in annual_records) / len(annual_records)
    latest = annual_records[0]

    # --- Current-period (T12M) NOPAT and ROIC ---
    quarterly_income = _sort_desc_by_date(
        _fmp_get_list("income-statement", ticker, api_key, limit=4, period="quarter")
    )
    if len(quarterly_income) >= 4:
        t12m_source = "quarterly"
        q4 = quarterly_income[:4]
        quarters_used = [q.get("date") for q in q4]
        op_income_t12m_raw = sum(_safe_float(q.get("operatingIncome")) or 0.0 for q in q4)
        tax_expense_t12m_raw = sum(_safe_float(q.get("incomeTaxExpense")) or 0.0 for q in q4)
        pretax_t12m_raw = sum(_safe_float(q.get("incomeBeforeTax")) or 0.0 for q in q4)
    else:
        t12m_source = "annual_fallback"
        quarters_used = [latest["date"]]
        op_income_t12m_raw = latest["operating_income_musd"] * MUSD
        tax_expense_t12m_raw = latest["income_tax_expense_musd"] * MUSD
        pretax_t12m_raw = latest["income_before_tax_musd"] * MUSD

    tax_rate_current = _effective_tax_rate(tax_expense_t12m_raw, pretax_t12m_raw)
    nopat_current_raw = _nopat_raw(op_income_t12m_raw, tax_rate_current)

    invested_capital_current_musd = latest["invested_capital_musd"]  # single most-recent-period snapshot
    avg_invested_capital_current_musd = latest["avg_invested_capital_musd"]  # ROIC denominator
    nopat_current_musd = nopat_current_raw / MUSD if nopat_current_raw is not None else None
    roic_current = (
        100 * nopat_current_musd / avg_invested_capital_current_musd
        if nopat_current_musd is not None and avg_invested_capital_current_musd
        else None
    )

    methodology_notes = [
        "Invested capital = totalDebt + totalStockholdersEquity + deferredTaxLiabilitiesNonCurrent. "
        "Omits Bloomberg's BS_ALLOW_DOUBTFUL_ACC_REC (allowance for doubtful accounts receivable) and "
        "BS_ACCRUED_INCOME_TAXES adjustments — neither is exposed as a discrete line item by FMP's "
        "/stable/balance-sheet-statement endpoint. Both are typically small relative to total invested "
        "capital for asset-light tech/industrials names.",
        "NOPAT approximated as operatingIncome * (1 - effective_tax_rate). Bloomberg's own "
        "RETURN_ON_INV_CAPITAL build starts from net income before extraordinary items, adds back "
        "after-tax interest expense, and further adjusts for pension and FX effects. operatingIncome "
        "implicitly captures the interest add-back (interest expense sits below the operating line in the "
        "income statement) but omits the pension/FX adjustments, which are unavailable from FMP.",
        "effective_tax_rate = incomeTaxExpense / incomeBeforeTax, clamped to [0%, 50%] to prevent one-off "
        "tax items (large tax benefits/valuation allowance releases, etc.) from producing a negative or "
        ">100% effective rate that would distort NOPAT.",
        f"roic_current uses trailing-twelve-month (T12M) operatingIncome/incomeTaxExpense/incomeBeforeTax "
        f"({'sum of the 4 most recent quarterly income statements' if t12m_source == 'quarterly' else 'most recent annual period — quarterly data was unavailable for this ticker'}), "
        f"divided by average invested capital from the two most recent annual balance sheets.",
        "roic_5y_avg is the simple average of annual ROIC values (annual NOPAT / that year's average "
        "invested capital vs. the prior year) — it is not a T12M-weighted figure.",
        "Annual cash-flow-statement data was fetched per spec but is not consumed by this ROIC/NOPAT/"
        "invested-capital formula (Bloomberg's own build is balance-sheet/income-statement driven); it is "
        "included in components['cash_flow_statements_raw'] for reference/auditability only.",
    ]
    if len(annual_records) < periods:
        methodology_notes.append(
            f"Requested periods={periods} but only {len(annual_records)} annual period(s) were "
            f"available/computable, so roic_5y_avg is actually a {len(annual_records)}-year average. "
            f"This FMP account's subscription tier caps the statement `limit` parameter at "
            f"{FMP_STATEMENT_LIMIT_CAP} records (pagination via `page` does not reach further back on this "
            f"tier), and computing N years of average invested capital needs N+1 years of balance sheets."
        )

    components = {
        "annual": [
            {k: v for k, v in r.items() if k not in ("raw_balance_sheet_current", "raw_balance_sheet_prior", "raw_income_statement")}
            for r in annual_records
        ],
        "annual_raw": [
            {
                "date": r["date"],
                "balance_sheet_current": r["raw_balance_sheet_current"],
                "balance_sheet_prior": r["raw_balance_sheet_prior"],
                "income_statement": r["raw_income_statement"],
            }
            for r in annual_records
        ],
        "current_period": {
            "t12m_source": t12m_source,
            "quarters_used": quarters_used,
            "operating_income_t12m_musd": op_income_t12m_raw / MUSD,
            "income_tax_expense_t12m_musd": tax_expense_t12m_raw / MUSD,
            "income_before_tax_t12m_musd": pretax_t12m_raw / MUSD,
            "effective_tax_rate_pct": tax_rate_current * 100,
            "nopat_musd": nopat_current_musd,
            "invested_capital_musd": invested_capital_current_musd,
            "avg_invested_capital_musd": avg_invested_capital_current_musd,
        },
        "cash_flow_statements_raw": cash_flow_statements,
    }

    return {
        "ticker": ticker,
        "roic_current": roic_current,
        "roic_5y_avg": roic_5y_avg,
        "invested_capital_current": invested_capital_current_musd,
        "nopat_current": nopat_current_musd,
        "effective_tax_rate": tax_rate_current * 100,
        "components": components,
        "methodology_notes": methodology_notes,
    }
