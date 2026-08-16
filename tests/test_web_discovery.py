"""Tests for web search discovery."""

from research_assistant.discovery.publisher import (
    extract_year,
    publisher_from_url,
    unwrap_redirect_url,
)
from research_assistant.discovery.web import parse_ddg_html

SAMPLE_HTML = """
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.servicenow.com%2Fproducts%2fitsm.html">
    ServiceNow ITSM Platform Overview 2023
  </a>
  <a class="result__snippet">
    ServiceNow is the market-leading enterprise cloud platform for digital workflow automation.
  </a>
</div>
<div class="result">
  <a class="result__a" href="https://docs.servicenow.com/bundle/utah-it-service-management/page/product/it-service-management/concept/c_ITSM.html">
    IT Service Management - ServiceNow Docs
  </a>
  <a class="result__snippet">
    Multi-instance architecture with isolated customer instances.
  </a>
</div>
"""


def test_unwrap_redirect_url():
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.servicenow.com%2Fproducts%2Fitsm.html"
    assert unwrap_redirect_url(href) == "https://www.servicenow.com/products/itsm.html"


def test_publisher_from_url_servicenow():
    assert publisher_from_url("https://www.servicenow.com/products/itsm.html") == "ServiceNow"
    assert publisher_from_url("https://docs.servicenow.com/bundle/utah") == "ServiceNow"


def test_extract_year_from_title():
    assert extract_year("ServiceNow ITSM Platform Overview 2023", "") == "2023"


def test_parse_ddg_html_maps_results():
    papers = parse_ddg_html(SAMPLE_HTML, limit=2)

    assert len(papers) == 2
    assert papers[0].source == "web"
    assert papers[0].publisher == "ServiceNow"
    assert papers[0].published_date == "2023"
    assert papers[0].landing_url.startswith("https://www.servicenow.com")
    assert "workflow automation" in papers[0].abstract
