"""Generate the sample PDF used by the demo, tests and golden eval set.

Bluestone Mutual is a FICTIONAL insurer. Every figure below is invented for demonstration.
Usage: python scripts/make_sample_pdf.py [output.pdf]
"""

import sys
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table

PAGES = [
    ("Bluestone Mutual Insurance Company - FY2026 Annual Report", [
        "This report is fictional and was generated for a retrieval-augmented generation demo. "
        "Bluestone Mutual is not a real company and none of the figures describe a real business.",
        "Bluestone Mutual is a policyholder-owned property and casualty insurer headquartered in Charlotte, "
        "North Carolina. It writes personal auto, homeowners, and small commercial coverage in 14 states "
        "across the Southeast and Mid-Atlantic.",
    ]),
    ("Letter from the Chief Executive Officer", [
        "Fiscal year 2026 was the strongest year in Bluestone Mutual's history. Chief Executive Officer "
        "Dana Whitfield highlighted record gross written premium and a return to underwriting profit.",
        "Gross written premium grew 11.4% to $2.84 billion, driven by rate actions in personal auto and "
        "new small commercial business in Georgia and Tennessee.",
        "The company returned $42 million to members as policyholder dividends.",
    ]),
    ("Financial Highlights", [
        "Net income for FY2026 was $214 million, compared with $96 million in FY2025.",
        "The combined ratio improved to 94.2% from 99.1% in the prior year. The loss ratio was 66.8% and the "
        "expense ratio was 27.4%.",
        "Policyholder surplus ended the year at $3.1 billion. The investment portfolio returned 4.9%.",
        "TABLE",
    ]),
    ("Claims Operations", [
        "Bluestone Mutual processed 412,000 claims in FY2026. Average claim cycle time fell from 14.2 days to "
        "9.6 days after the rollout of automated first notice of loss intake.",
        "Hurricane Delphine in September 2026 generated 18,300 claims and $167 million in net catastrophe losses.",
        "Customer satisfaction with the claims experience reached a Net Promoter Score of 61.",
    ]),
    ("Technology and Data Strategy", [
        "The company completed migration of its policy and claims data warehouse to a cloud data platform, "
        "retiring 37 legacy on-premise reporting jobs.",
        "An AI-assisted document intake service now classifies and extracts data from 72% of incoming claim "
        "documents, with human review of every low-confidence extraction.",
        "Technology investment totaled $118 million, or 4.2% of gross written premium.",
    ]),
    ("Risk Management", [
        "The Enterprise Risk Committee meets quarterly and reports to the Board Audit and Risk Committee.",
        "The company's catastrophe reinsurance program provides $900 million of protection above a "
        "$75 million retention for a single event.",
        "Cyber risk is managed under a framework aligned to NIST CSF 2.0, with annual third-party penetration testing.",
    ]),
    ("Workforce and Community", [
        "Bluestone Mutual employed 3,450 people at fiscal year end, 38% of them in hybrid or remote roles.",
        "Employees volunteered 21,000 hours, and the Bluestone Foundation granted $6.5 million to community "
        "organizations focused on disaster preparedness and financial literacy.",
    ]),
    ("Outlook for FY2027", [
        "Management targets gross written premium growth of 7% to 9% and a combined ratio between 95% and 96% "
        "in FY2027.",
        "Planned initiatives include entry into Alabama and Kentucky, expansion of usage-based auto insurance, "
        "and a generative AI assistant for claims adjusters.",
    ]),
]

TABLE = [
    ["Metric", "FY2026", "FY2025"],
    ["Gross written premium", "$2.84B", "$2.55B"],
    ["Net income", "$214M", "$96M"],
    ["Combined ratio", "94.2%", "99.1%"],
    ["Policyholder surplus", "$3.1B", "$2.8B"],
]


def build(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    ss = getSampleStyleSheet()
    story = []
    for i, (title, paras) in enumerate(PAGES):
        story.append(Paragraph(title, ss["Title"] if i == 0 else ss["Heading1"]))
        story.append(Spacer(1, 10))
        for p in paras:
            story.append(Table(TABLE) if p == "TABLE" else Paragraph(p, ss["BodyText"]))
            story.append(Spacer(1, 8))
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(out), pagesize=LETTER, title="Bluestone Mutual FY2026 Annual Report (fictional)").build(story)
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample/bluestone_mutual_fy2026_report.pdf")
    print("wrote", build(target))
