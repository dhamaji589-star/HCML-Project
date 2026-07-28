"""Create a formatted Word report without external docx dependencies."""

from __future__ import annotations

import html
import zipfile
from pathlib import Path

from PIL import Image


OUTPUT = Path("hcml_project/docs/HCML_MAD22_Report_Draft.docx")
SUCCESS_FIGURE = Path("results/success_rate_by_method.png")
QUALITATIVE_FIGURE = Path("results/qualitative_best_examples.png")


def xml_escape(text: str) -> str:
    return html.escape(text, quote=False)


def run(text: str, bold: bool = False, italic: bool = False, size: int | None = None) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if italic:
        props.append("<w:i/>")
    if size:
        props.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return f"<w:r>{rpr}<w:t xml:space=\"preserve\">{xml_escape(text)}</w:t></w:r>"


def para(
    text: str = "",
    style: str = "BodyText",
    align: str | None = None,
    bold: bool = False,
    italic: bool = False,
    size: int | None = None,
) -> str:
    align_xml = f'<w:jc w:val="{align}"/>' if align else ""
    return (
        f"<w:p><w:pPr><w:pStyle w:val=\"{style}\"/>{align_xml}</w:pPr>"
        f"{run(text, bold=bold, italic=italic, size=size)}</w:p>"
    )


def caption(text: str) -> str:
    return para(text, style="Caption", italic=True)


def heading(text: str, level: int = 1) -> str:
    return para(text, style=f"Heading{level}")


def image_paragraph(rid: str, image_path: Path, width_inches: float) -> str:
    with Image.open(image_path) as image:
        width_px, height_px = image.size
    height_inches = width_inches * height_px / width_px
    cx = int(width_inches * 914400)
    cy = int(height_inches * 914400)
    return f"""
<w:p>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:effectExtent l="0" t="0" r="0" b="0"/>
        <wp:docPr id="1" name="{xml_escape(image_path.name)}"/>
        <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr>
                <pic:cNvPr id="0" name="{xml_escape(image_path.name)}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rid}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def table(rows: list[list[str]]) -> str:
    column_count = len(rows[0])
    grid = "".join('<w:gridCol w:w="1800"/>' for _ in range(column_count))
    table_rows = []
    for row_index, row in enumerate(rows):
        cells = []
        for cell in row:
            shading = '<w:shd w:fill="EDEDED"/>' if row_index == 0 else ""
            cells.append(
                "<w:tc>"
                f"<w:tcPr><w:tcW w:w=\"1800\" w:type=\"dxa\"/>{shading}</w:tcPr>"
                f"{para(cell, style='TableText', bold=row_index == 0)}"
                "</w:tc>"
            )
        table_rows.append(f"<w:tr>{''.join(cells)}</w:tr>")
    return (
        "<w:tbl>"
        "<w:tblPr><w:tblStyle w:val=\"TableGrid\"/><w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblLook w:firstRow=\"1\" w:lastRow=\"0\" w:firstColumn=\"0\" "
        "w:lastColumn=\"0\" w:noHBand=\"0\" w:noVBand=\"1\"/></w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>{''.join(table_rows)}</w:tbl>"
    )


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="BodyText">
    <w:name w:val="Body Text"/>
    <w:pPr><w:spacing w:after="80" w:line="252" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/><w:basedOn w:val="BodyText"/>
    <w:pPr><w:spacing w:before="120" w:after="40"/><w:keepNext/></w:pPr>
    <w:rPr><w:b/><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:sz w:val="25"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/><w:basedOn w:val="BodyText"/>
    <w:pPr><w:spacing w:before="80" w:after="30"/><w:keepNext/></w:pPr>
    <w:rPr><w:b/><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Caption">
    <w:name w:val="caption"/><w:basedOn w:val="BodyText"/>
    <w:pPr><w:spacing w:after="60"/></w:pPr>
    <w:rPr><w:i/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="18"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="TableText">
    <w:name w:val="table text"/><w:basedOn w:val="BodyText"/>
    <w:pPr><w:spacing w:after="0" w:line="220" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="18"/></w:rPr>
  </w:style>
  <w:style w:type="table" w:styleId="TableGrid">
    <w:name w:val="Table Grid"/>
    <w:tblPr><w:tblBorders>
      <w:top w:val="single" w:sz="4" w:space="0" w:color="888888"/>
      <w:left w:val="single" w:sz="4" w:space="0" w:color="888888"/>
      <w:bottom w:val="single" w:sz="4" w:space="0" w:color="888888"/>
      <w:right w:val="single" w:sz="4" w:space="0" w:color="888888"/>
      <w:insideH w:val="single" w:sz="4" w:space="0" w:color="888888"/>
      <w:insideV w:val="single" w:sz="4" w:space="0" w:color="888888"/>
    </w:tblBorders></w:tblPr>
  </w:style>
</w:styles>
"""


def document_xml() -> str:
    parts = [
        para(
            "Hidden Identity Recovery from Face Morphing Attacks using NegFaceDiff and AdaptDiff",
            style="Heading1",
            align="center",
            size=30,
        ),
        para("Student: Vishu | HCML MAD22 Project", align="center", italic=True),
        heading("Abstract"),
        para(
            "Face morphing attacks combine identity information from two subjects into one facial image. "
            "This project studies hidden-identity recovery: given a morph image and one known contributor, "
            "the goal is to generate an image that is closer to the other contributor. The pipeline uses "
            "pretrained NegFaceDiff/AdaptDiff components, DM_CASIA, a latent autoencoder, and ElasticFaceArc "
            "identity embeddings. Five MAD22 morphing subsets were evaluated independently. The experiments "
            "show that AdaptDiff gives higher recovery success than NegFaceDiff for every morphing method. "
            "The visual outputs remain blurry, so the main conclusion is about identity recovery in embedding "
            "space rather than photorealistic face reconstruction."
        ),
        heading("1. Introduction"),
        para(
            "Face morphing is a biometric attack in which two face identities are blended into one image. "
            "Such an image can retain enough information from both contributors to match against either "
            "person. This project asks whether information about the unknown contributor can be recovered "
            "when the other contributor is already known."
        ),
        para(
            "For a morph created from identities A and B, the recovery task is directional. If A is known, "
            "B is the hidden target; if B is known, A becomes the hidden target. Each morph is therefore "
            "converted into two directed recovery trials."
        ),
        heading("2. Method"),
        para(
            "The work uses pretrained models and performs inference only. Metadata preparation stores the "
            "morph image, known identity image, and hidden identity image for each trial. ElasticFaceArc with "
            "an iresnet100 backbone extracts 512-dimensional embeddings. The morph embedding is the positive "
            "context, the known identity embedding is the negative context, and the hidden identity embedding "
            "is reserved for evaluation."
        ),
        para(
            "The image branch works in latent space. The morph image is encoded with the first-stage "
            "autoencoder, noised with the 1000-step forward schedule expected by DM_CASIA, sampled with DDIM "
            "for 200 denoising steps, and decoded back into an image."
        ),
        table(
            [
                ["Setting", "Adapt flag", "Negative weight", "Interpretation"],
                ["NegFaceDiff", "false", "0.5", "Fixed negative identity guidance"],
                ["AdaptDiff", "true", "1.0", "Adaptive negative identity guidance"],
            ]
        ),
        heading("3. Experimental Protocol"),
        para(
            "The evaluated MAD22 morphing subsets are OpenCV, FaceMorpher, MIPGAN-I, MIPGAN-II, and WebMorph. "
            "Each subset is reported separately because different morphing algorithms can preserve identity "
            "information differently."
        ),
        para(
            "For each directed trial, both NegFaceDiff and AdaptDiff generate one candidate recovery image. "
            "The generated image is embedded with ElasticFaceArc and compared with the known and hidden "
            "identity embeddings. A trial is successful when cosine(generated, hidden) is greater than "
            "cosine(generated, known). The margin is cosine(generated, hidden) minus cosine(generated, known)."
        ),
        heading("4. Results"),
        image_paragraph("rIdImage1", SUCCESS_FIGURE, 6.2),
        caption("Figure 1. Hidden-identity recovery success rate for each morphing method."),
        table(
            [
                ["Morphing method", "NegFaceDiff success", "NegFaceDiff margin", "AdaptDiff success", "AdaptDiff margin"],
                ["OpenCV", "60.0%", "0.069954", "62.5%", "0.090291"],
                ["FaceMorpher", "75.0%", "0.045358", "82.5%", "0.059136"],
                ["MIPGAN-I", "80.0%", "0.063586", "88.8%", "0.084853"],
                ["MIPGAN-II", "72.5%", "0.060798", "80.0%", "0.078214"],
                ["WebMorph", "71.2%", "0.063933", "75.0%", "0.087545"],
            ]
        ),
        caption("Table 1. Success is reported as a percentage. The margin is the mean hidden-minus-known cosine difference."),
        para(
            "AdaptDiff improves over NegFaceDiff for every morphing method. The improvement appears both in "
            "success percentage and in mean margin, meaning that the generated embeddings are more strongly "
            "shifted toward the hidden identity and away from the known identity."
        ),
        heading("5. Qualitative Analysis"),
        image_paragraph("rIdImage2", QUALITATIVE_FIGURE, 6.1),
        caption(
            "Figure 2. Qualitative examples selected from successful AdaptDiff cases with strong positive margins."
        ),
        para(
            "The generated images are often blurry and color-shifted. Therefore, the method should not be "
            "described as clean photo-quality reconstruction. The better interpretation is that the generated "
            "images contain identity cues that ElasticFaceArc places closer to the hidden contributor."
        ),
        heading("6. Discussion and Conclusion"),
        para(
            "The consistent advantage of AdaptDiff suggests that adaptive negative guidance is better suited "
            "to this recovery setting than fixed negative guidance. A likely reason is that diffusion sampling "
            "changes over time: early steps influence broad structure, while later steps refine details and "
            "identity-related features. Adaptive guidance can balance exploration and identity separation more "
            "flexibly than a fixed negative weight."
        ),
        para(
            "The main limitations are visual quality, no task-specific fine-tuning, and the use of ElasticFaceArc "
            "for both conditioning and evaluation. Even with these limitations, the pipeline is complete and "
            "reproducible: it prepares metadata, aligns faces, extracts identity embeddings, samples with the "
            "latent diffusion model, decodes generated images, and evaluates them in embedding space."
        ),
        heading("References"),
        para(
            "[1] E. Caldeira, T. Chettaoui, N. Damer, and F. Boutros, AdaptDiff: Adaptive Guidance in "
            "Diffusion Models for Diverse and Identity-Consistent Face Synthesis, 2026. Code: "
            "https://github.com/EduardaCaldeira/NegFaceDiff/"
        ),
        para(
            "[2] M. Huber et al., SYN-MAD 2022: Competition on Face Morphing Attack Detection Based on "
            "Privacy-aware Synthetic Training Data, IJCB, 2022. Code: https://github.com/marcohuber/SYN-MAD-2022"
        ),
        para(
            "[3] F. Boutros, N. Damer, F. Kirchbuchner, and A. Kuijper, ElasticFace: Elastic Margin Loss "
            "for Deep Face Recognition, CVPR Workshops, 2022."
        ),
        para("[4] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, ArcFace, CVPR, 2019."),
        para("[5] J. Ho, A. Jain, and P. Abbeel, Denoising Diffusion Probabilistic Models, NeurIPS, 2020."),
        para("[6] J. Song, C. Meng, and S. Ermon, Denoising Diffusion Implicit Models, ICLR, 2021."),
        para("[7] R. Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models, CVPR, 2022."),
    ]
    section = (
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="900" w:right="900" w:bottom="900" w:left="900" '
        'w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
        'mc:Ignorable="w14 wp14"><w:body>'
        + "".join(parts)
        + section
        + "</w:body></w:document>"
    )


def content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
"""


def root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""


def document_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdImage1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/success_rate_by_method.png"/>
  <Relationship Id="rIdImage2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/qualitative_best_examples.png"/>
  <Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types_xml())
        docx.writestr("_rels/.rels", root_rels_xml())
        docx.writestr("word/_rels/document.xml.rels", document_rels_xml())
        docx.writestr("word/styles.xml", styles_xml())
        docx.writestr("word/document.xml", document_xml())
        docx.write(SUCCESS_FIGURE, "word/media/success_rate_by_method.png")
        docx.write(QUALITATIVE_FIGURE, "word/media/qualitative_best_examples.png")
    print(f"Word report written: {OUTPUT}")


if __name__ == "__main__":
    main()
