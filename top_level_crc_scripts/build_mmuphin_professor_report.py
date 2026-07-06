from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs_mmuphin_dataset_scouting"
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUTPUT_DIR / "mmuphin_crc_professor_report.pdf"

PAGE_W, PAGE_H = landscape(letter)
MARGIN_X = 0.55 * inch
MARGIN_TOP = 0.55 * inch
MARGIN_BOTTOM = 0.48 * inch
CONTENT_W = PAGE_W - 2 * MARGIN_X
CONTENT_H = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM

NAVY = colors.HexColor("#243B53")
TEAL = colors.HexColor("#0B7285")
BLUE = colors.HexColor("#4C78A8")
RED = colors.HexColor("#E45756")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#52606D")
LINE = colors.HexColor("#CBD5E1")
PALE = colors.HexColor("#EEF6F7")
LIGHT = colors.HexColor("#F8FAFC")

styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=10,
    )
)
styles.add(
    ParagraphStyle(
        name="Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=MUTED,
        spaceAfter=12,
    )
)
styles.add(
    ParagraphStyle(
        name="Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        textColor=NAVY,
        spaceBefore=5,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="BodyCompact",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.2,
        textColor=INK,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.3,
        leading=11,
        textColor=MUTED,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="FigureTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="FigureLabel",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
        textColor=INK,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="Callout",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=TEAL,
        alignment=TA_LEFT,
    )
)


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_X, 0.38 * inch, PAGE_W - MARGIN_X, 0.38 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        MARGIN_X,
        0.20 * inch,
        "MMUPHin CRC dataset scouting - candidate benchmark update",
    )
    canvas.drawRightString(
        PAGE_W - MARGIN_X,
        0.20 * inch,
        f"Page {doc.page}",
    )
    canvas.restoreState()


frame = Frame(
    MARGIN_X,
    MARGIN_BOTTOM,
    CONTENT_W,
    CONTENT_H,
    leftPadding=0,
    rightPadding=0,
    topPadding=0,
    bottomPadding=0,
)
doc = BaseDocTemplate(
    str(PDF_PATH),
    pagesize=landscape(letter),
    leftMargin=MARGIN_X,
    rightMargin=MARGIN_X,
    topMargin=MARGIN_TOP,
    bottomMargin=MARGIN_BOTTOM,
    title="MMUPHin CRC Dataset Scouting Report",
    author="CRC benchmark scouting workflow",
)
doc.addPageTemplates(
    [PageTemplate(id="report", frames=[frame], onPage=header_footer)]
)


def p(text, style="BodyCompact"):
    return Paragraph(text, styles[style])


def bullet(text):
    return Paragraph(
        f"&#8226;&nbsp;&nbsp;{text}",
        ParagraphStyle(
            "BulletTemp",
            parent=styles["BodyCompact"],
            leftIndent=10,
            firstLineIndent=-8,
            spaceAfter=3,
        ),
    )


def table(data, widths, header=True, font_size=8.5):
    result = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    result.setStyle(TableStyle(commands))
    return result


def callout(text):
    box = Table([[p(text, "Callout")]], colWidths=[CONTENT_W])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return box


def figure_page(title, left_path, right_path, caption):
    image_w = 4.85 * inch
    image_h = 3.23 * inch
    left = Image(str(left_path), width=image_w, height=image_h)
    right = Image(str(right_path), width=image_w, height=image_h)
    panel = Table(
        [
            [p("Raw abundance", "FigureLabel"), p("MMUPHin adjusted", "FigureLabel")],
            [left, right],
        ],
        colWidths=[CONTENT_W / 2, CONTENT_W / 2],
    )
    panel.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return [
        p(title, "FigureTitle"),
        panel,
        Spacer(1, 8),
        callout(caption),
    ]


story = []

# Page 1
story.extend(
    [
        p(
            "MMUPHin CRC Dataset Scouting Report:<br/>"
            "A Candidate Benchmark for scGPT-style Batch Correction",
            "ReportTitle",
        ),
        p(
            "Professor-facing update | Dataset verification and baseline assessment only",
            "Subtitle",
        ),
        callout(
            "Bottom line: the built-in MMUPHin colorectal cancer dataset is a "
            "credible candidate benchmark. It contains five studies with CRC and "
            "control samples, shows measurable study effects, and retains similar "
            "disease discrimination after MMUPHin adjustment."
        ),
        Spacer(1, 9),
        p("1. Executive Summary", "Section"),
        p(
            "We checked MMUPHin from the bioBakery/Huttenhower laboratory and "
            "verified its built-in colorectal cancer (CRC) dataset. The dataset "
            "combines five studies and includes explicit study and CRC/control "
            "labels. Every study contains both disease groups."
        ),
        p(
            "The raw abundance data contains a measurable study effect. MMUPHin "
            "reduces study-associated variation and study predictability while "
            "leaving CRC/control discrimination nearly unchanged. This supports "
            "using the data as a controlled benchmark for a future scGPT-style "
            "correction experiment. It does not mean that scGPT or BiomeGPT has "
            "already been tested successfully."
        ),
        p("2. Professor's Question", "Section"),
        p(
            "The goal is to find one disease represented by multiple microbiome "
            "studies that show study or batch effects, then test whether a "
            "scGPT-style method can reduce those effects while preserving the "
            "biological disease signal."
        ),
        p("3. Dataset Structure", "Section"),
        table(
            [
                ["Item", "Verified value", "Why it matters"],
                ["Microbial features/species", "484", "Common feature space"],
                ["Samples", "551", "Compact evaluation dataset"],
                ["Studies", "5", "Multi-study batch structure"],
                ["Disease label", "study_condition", "CRC vs control"],
                ["Study label", "studyID", "Batch target"],
                ["Both labels in every study", "Yes", "No perfect confounding"],
                ["Smallest study-condition group", "27", "No empty/singleton cell"],
            ],
            [2.35 * inch, 1.35 * inch, 6.15 * inch],
        ),
    ]
)
story.append(PageBreak())

# Page 2
story.extend(
    [
        p("4. Evidence of Study/Batch Effect in Raw Data", "Section"),
        p(
            "Study identity explains <b>7.86%</b> of Bray-Curtis variation after "
            "controlling for CRC/control condition. The effect is statistically "
            "detectable by PERMANOVA (<b>p = 0.001</b>)."
        ),
        p(
            "A cross-validated classifier predicts the five study labels with "
            "<b>0.765 balanced accuracy</b>, compared with a <b>0.200 chance "
            "level</b>. These two diagnostics show that the raw microbiome "
            "profiles retain substantial study-specific structure."
        ),
        p("5. MMUPHin Adjustment Result", "Section"),
        table(
            [
                ["Metric", "Raw", "Adjusted", "Interpretation"],
                [
                    "Study R2, controlling for condition",
                    "7.86%",
                    "3.00%",
                    "61.8% relative reduction",
                ],
                [
                    "Study classifier balanced accuracy",
                    "0.765",
                    "0.647",
                    "Study remains predictable, but less strongly",
                ],
                [
                    "Disease LOSO mean within-study AUC",
                    "0.713",
                    "0.706",
                    "CRC/control discrimination is mostly preserved",
                ],
                [
                    "Condition R2, controlling for study",
                    "0.79%",
                    "0.88%",
                    "Disease-associated variation is not erased",
                ],
            ],
            [3.05 * inch, 0.9 * inch, 0.9 * inch, 5.0 * inch],
        ),
        Spacer(1, 8),
        p(
            "MMUPHin clearly reduces study signal, but it does not completely "
            "eliminate it. Disease-related performance changes only slightly, "
            "which indicates that the CRC/control signal is largely retained."
        ),
        p(
            "The LOSO result is a disease-signal retention diagnostic for the "
            "globally adjusted matrix. It should not be interpreted as proof that "
            "a newly trained correction model generalizes to an entirely unseen "
            "study."
        ),
        p("7. Why This Dataset Is Suitable as a Benchmark", "Section"),
        Table(
            [
                [
                    [
                        bullet("One disease area: colorectal cancer."),
                        bullet("Five independent studies."),
                        bullet("Study and CRC/control labels are available."),
                        bullet("Both disease groups occur in every study."),
                    ],
                    [
                        bullet("Raw data has measurable study signal."),
                        bullet("MMUPHin supplies a microbiome baseline."),
                        bullet("Study signal falls after adjustment."),
                        bullet("Disease discrimination stays similar."),
                    ],
                ]
            ],
            colWidths=[CONTENT_W / 2, CONTENT_W / 2],
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ]
            ),
        ),
        Spacer(1, 6),
        callout(
            "Key benchmark criterion: reduce study predictability while "
            "preserving CRC/control predictability."
        ),
    ]
)
story.append(PageBreak())

# Page 3
story.extend(
    figure_page(
        "Figure 1. Raw vs MMUPHin-adjusted PCA colored by study",
        SOURCE / "mmuphin_crc_raw_pca_by_study.png",
        SOURCE / "mmuphin_crc_adjusted_pca_by_study.png",
        "The adjusted representation shows greater mixing among study colors. "
        "PCA is qualitative; the main evidence is the reduction in study R2 "
        "and study-classifier balanced accuracy.",
    )
)
story.append(PageBreak())

# Page 4
story.extend(
    figure_page(
        "Figure 2. Raw vs MMUPHin-adjusted PCA colored by CRC/control condition",
        SOURCE / "mmuphin_crc_raw_pca_by_condition.png",
        SOURCE / "mmuphin_crc_adjusted_pca_by_condition.png",
        "CRC and control samples overlap in both representations, as expected "
        "for heterogeneous human microbiome data. The adjustment does not "
        "visibly collapse the disease structure, consistent with the nearly "
        "stable disease AUC.",
    )
)
story.append(PageBreak())

# Page 5
story.extend(
    [
        p("8. Limitations", "Section"),
        bullet(
            "The dataset has 551 samples and is not suitable for foundation-model "
            "pretraining from scratch."
        ),
        bullet(
            "It should be used for benchmarking, fine-tuning, or lightweight adaptation."
        ),
        bullet(
            "PCA is qualitative; PERMANOVA and prediction metrics are the main evidence."
        ),
        bullet("MMUPHin reduces the study effect but does not fully remove it."),
        bullet(
            "Future methods must use the same data, splits, transformations, and "
            "evaluation protocol for a fair comparison."
        ),
        p("9. Proposed Next Step", "Section"),
        p(
            "Package the CRC data as one fixed benchmark and compare three "
            "representations: <b>(1)</b> raw abundance, <b>(2)</b> MMUPHin-adjusted "
            "abundance, and <b>(3)</b> our scGPT-style or BiomeGPT-style "
            "batch-aware correction."
        ),
        p(
            "The next step should not be pretraining from scratch. If feature "
            "mapping is feasible, adapt an existing BiomeGPT representation to "
            "CRC. If mapping is not feasible, use a small scGPT-style "
            "masked-abundance reconstruction model."
        ),
        table(
            [
                ["Evaluation family", "Primary measures"],
                [
                    "Batch removal",
                    "Study balanced accuracy, macro-F1, and study PERMANOVA R2",
                ],
                [
                    "Disease preservation",
                    "CRC/control AUC, balanced accuracy, and condition R2",
                ],
                [
                    "Visual diagnostics",
                    "PCA or UMAP before and after correction",
                ],
            ],
            [2.4 * inch, 7.45 * inch],
        ),
        Spacer(1, 10),
        p("10. Final Takeaway", "Section"),
        callout(
            "The MMUPHin CRC dataset appears to be a suitable disease-specific, "
            "multi-study candidate benchmark. It shows measurable study/batch "
            "effects, MMUPHin reduces those effects while mostly preserving "
            "disease signal, and it can now be used to test whether our "
            "scGPT-style correction achieves a similar or better tradeoff."
        ),
        Spacer(1, 10),
        p("Sources", "Section"),
        p(
            "MMUPHin project: https://huttenhower.sph.harvard.edu/mmuphin/<br/>"
            "Bioconductor vignette: "
            "https://bioconductor.org/packages/release/bioc/vignettes/MMUPHin/"
            "inst/doc/MMUPHin.html<br/>"
            "bioBakery discussion: "
            "https://forum.biobakery.org/t/input-data-for-mmuphin/2794",
            "Small",
        ),
    ]
)

doc.build(story)
print(PDF_PATH)
