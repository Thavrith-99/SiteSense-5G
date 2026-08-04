"""
Generate two shareable Word documents for Team Neural Shield:
  1. TEAM_ONBOARDING_SiteSense5G.docx        - onboarding brief (fuller, faithful)
  2. AGAIF2026_SiteSense5G_AugustMVP_Plan_KH-002.docx - August-only MVP plan

Run: py generate_docs.py
"""

from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parent.parent
NAVY = RGBColor(0x1F, 0x38, 0x64)
BLUE = RGBColor(0x2E, 0x5F, 0x9E)
GREY = RGBColor(0x59, 0x59, 0x59)
HDR_BG = "2E5F9E"
ALT_BG = "EAF0F8"


def set_cell_bg(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def base_styles(doc):
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)


def h1(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(20)
    r.font.color.rgb = NAVY
    p.space_after = Pt(2)
    return p


def h2(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = BLUE
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    return p


def sub(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(10)
    r.font.color.rgb = GREY
    return p


def bullet(doc, text, bold_lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
        p.add_run(text)
    else:
        p.add_run(text)
    return p


def table(doc, headers, rows, widths=None, header_bg=HDR_BG):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(htext)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(9.5)
        set_cell_bg(hdr[i], header_bg)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(9.5)
            if ri % 2 == 1:
                set_cell_bg(cells[i], ALT_BG)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)
    return t


# =====================================================================
# Shared: the faithful objective
# =====================================================================
OBJECTIVE = (
    "SiteSense 5G is a GeoAI decision-support tool that fuses tower locations, network signal "
    "(RSRP) KPIs, population, demand and terrain to: (1) map current coverage and quantify the "
    "population and villages (kampungs) out of coverage; (2) flag overloaded or underperforming "
    "towers; and (3) recommend the optimal location for a new 5G tower - ranked not only by how much "
    "previously-unserved population each site would connect, but by real demand (footfall and "
    "enterprise/venue zones) and backhaul feasibility, so no site becomes a stranded asset. "
    "The result is a transparent, repeatable workflow a planner can rerun for any district in ASEAN, "
    "delivered as a web tool plus an open pilot notebook."
)

AUG_VS_FULL = [
    ["Fuse towers + population + terrain", "Yes (towers + population; terrain light)", "Full terrain/LiDAR"],
    ["Map coverage + count unserved people/villages", "Yes (Steps 1-2)", "-"],
    ["Flag overloaded/underperforming towers", "Yes, basic (Step 3, load proxy)", "Refine with RSRP KPIs"],
    ["Recommend sites by unserved population", "Yes (Step 4 core)", "-"],
    ["...ranked by demand (footfall, venues)", "Light (OSM POIs as proxy)", "Full demand layers"],
    ["...ranked by backhaul feasibility", "Simple proxy", "Line-of-sight modelling"],
    ["RSRP LSTM signal prediction", "No - not needed", "Yes (finale)"],
    ["GNN network optimisation", "No - not needed", "Yes (finale)"],
    ["FastAPI + PostGIS + MapLibre", "No (Streamlit + files)", "Yes (finale)"],
    ["Web tool + pilot notebook", "Yes (Colab notebook + Streamlit)", "-"],
]


# =====================================================================
# DOC 1: ONBOARDING BRIEF
# =====================================================================
def build_onboarding():
    doc = Document()
    base_styles(doc)

    h1(doc, "Team Onboarding - SiteSense 5G")
    sub(doc, "ASEAN GeoAI Fusion 2026 - Team Neural Shield (KH-002, Cambodia)")
    doc.add_paragraph()

    h2(doc, "1. What is ASEAN GeoAI Fusion 2026?")
    doc.add_paragraph(
        "A regional ASEAN program and hackathon (on the CollabHub / MCMC platform) that challenges "
        "teams to solve real-world problems using GeoAI."
    )
    p = doc.add_paragraph()
    p.add_run("GeoAI = Geospatial + AI ").bold = True
    p.add_run("- using AI / data science on map and location data (towers, population, terrain, "
              "roads) to make better real-world decisions.")
    doc.add_paragraph(
        "It runs in stages: bootcamps -> certified assessment (due 11 Aug) -> Virtual Hackathon "
        "(12-13 Aug, selection) -> Physical Finale (19-22 Sep, Malaysia, awards). Our team is "
        "\"Neural Shield\" (Cambodia)."
    )

    h2(doc, "2. What are we building? - SiteSense 5G")
    p = doc.add_paragraph()
    p.add_run("The one-line question: ").bold = True
    p.add_run("\"Where should the next 5G tower go?\"").italic = True
    doc.add_paragraph("Full objective (as submitted):")
    q = doc.add_paragraph(OBJECTIVE)
    q.paragraph_format.left_indent = Inches(0.3)
    for r in q.runs:
        r.italic = True
    doc.add_paragraph(
        "Scope: Penang Island, Malaysia (best open data). Note: Penang has strong 4G but zero 5G in "
        "the open data - the whole island is a greenfield 5G build-out, which is our story."
    )
    p = doc.add_paragraph()
    p.add_run("The 4 steps: ").bold = True
    bullet(doc, "See the network - map every tower + its coverage.")
    bullet(doc, "Find the gap - overlay population -> count people/villages with no coverage.")
    bullet(doc, "Spot the strain - flag overloaded / underperforming towers.")
    bullet(doc, "Pick the site - a GeoAI model ranks the best new tower locations by unserved "
                "population + demand (footfall, venues) + backhaul feasibility, so no site is a stranded asset.")

    h2(doc, "3. What is the actual deliverable?")
    doc.add_paragraph(
        "A web app - an interactive Streamlit dashboard (opens in a browser at a URL) that delivers "
        "the 4-step workflow: the map, the coverage gaps, and the ranked recommended tower sites. "
        "It is a decision-support tool, an interactive map, and a GeoAI model in one - plus an open "
        "pilot notebook (Colab). It is NOT a native mobile app (but works in a phone browser)."
    )

    h2(doc, "4. Two finish lines (important)")
    table(doc,
          ["", "Virtual Hackathon", "Physical Finale"],
          [["Dates", "12-13 Aug 2026", "19-22 Sep 2026 (Malaysia)"],
           ["Theme", "BUILD & PITCH", "SHOWCASE"],
           ["Goal", "Get SELECTED as finalist (MVP + 5-min pitch)", "Refine + showcase -> awards"]],
          widths=[0.9, 3.1, 3.1])
    doc.add_paragraph(
        "August = get selected with a working MVP + a strong pitch. The heavy engineering (LSTM, GNN, "
        "FastAPI, PostGIS, MapLibre) is September finale work."
    )

    h2(doc, "5. The stack (so you know your tools)")
    bullet(doc, "Python (pandas, geopandas, rasterio, shapely) in Google Colab notebooks.", "Data & analysis: ")
    bullet(doc, "OpenCelliD towers, WorldPop population, OpenStreetMap, Google Earth Engine terrain.", "Data: ")
    bullet(doc, "Streamlit + Folium map -> deployed to a public URL.", "App (MVP): ")
    bullet(doc, "PostGIS, FastAPI, MapLibre, trained LSTM/GNN (September finale only).", "Advanced: ")
    bullet(doc, "CollabHub Git repository.", "Code lives on: ")

    h2(doc, "6. Your roles")
    table(doc,
          ["Member", "Role", "Mostly works in"],
          [["Sroas Thavrith (Lead)", "Data fusion, GIS & coverage, site-scoring, coordination, pitch", "Colab + coordination"],
           ["Kunvuth Sereyrith", "Data engineering (towers, OSM, boundaries, RSRP prep)", "Colab / QGIS"],
           ["Hean Mengfong", "Frontend / dashboard UI + demo video", "Streamlit"],
           ["Phylong (Data Science)", "Models & analytics: strain, site-scoring, features (LSTM = Sep)", "Colab"],
           ["Sonavin (Full Stack)", "Repo, dashboard build, deployment (backend/DB = Sep)", "Local + VS Code"]],
          widths=[1.7, 3.6, 1.6])
    doc.add_paragraph(
        "Note: the official CollabHub team is 3 members (locked, 2-3 rule). Phylong and Sonavin are "
        "work-plan contributors - confirm their status for the formal submission."
    )

    h2(doc, "7. Key dates")
    bullet(doc, "5 Aug - mentor selection opens")
    bullet(doc, "11 Aug - bootcamp assessment deadline (everyone must complete)")
    bullet(doc, "12-13 Aug - Virtual Hackathon: submit + 5-min pitch (selection)")
    bullet(doc, "19-22 Sep - Physical Finale in Malaysia (if selected)")

    doc.add_paragraph()
    fp = doc.add_paragraph()
    fp.add_run("Full detail: ").bold = True
    fp.add_run("see the master plan (AGAIF2026_SiteSense5G_ProjectPlan_KH-002.xlsx) and the "
               "August MVP plan (AGAIF2026_SiteSense5G_AugustMVP_Plan_KH-002.docx).")

    out = ROOT / "TEAM_ONBOARDING_SiteSense5G.docx"
    doc.save(out)
    return out


# =====================================================================
# DOC 2: AUGUST MVP PLAN
# =====================================================================
def build_mvp_plan():
    doc = Document()
    base_styles(doc)

    h1(doc, "SiteSense 5G - August MVP Plan")
    sub(doc, "Virtual Hackathon (12-13 Aug 2026) - Team Neural Shield (KH-002, Cambodia)")
    doc.add_paragraph()

    h2(doc, "1. Purpose of this document")
    doc.add_paragraph(
        "This is the AUGUST-ONLY plan - the slice of SiteSense 5G we build to get SELECTED as a "
        "finalist at the Virtual Hackathon. The full 6-12 month vision lives in the submitted Project "
        "Setup and the master task plan; this document deliberately scopes DOWN to what wins selection: "
        "a working MVP prototype + a strong 5-min pitch."
    )

    h2(doc, "2. What SiteSense 5G is (full objective, as submitted)")
    q = doc.add_paragraph(OBJECTIVE)
    for r in q.runs:
        r.italic = True

    h2(doc, "3. What we build in August (the MVP)")
    doc.add_paragraph(
        "A web app - an interactive Streamlit dashboard (opens in a browser at a URL) that delivers "
        "the 4-step workflow by reading precomputed data files. Plus an open pilot notebook (Colab)."
    )
    p = doc.add_paragraph()
    p.add_run("No separate backend or database in August. ").bold = True
    p.add_run("Streamlit combines frontend + backend in one Python app, and reads data files "
              "(CSV / GeoJSON / GeoTIFF) instead of a database. This keeps August low-risk.")
    p = doc.add_paragraph()
    p.add_run("Data flow: ").bold = True
    p.add_run("Colab (Python) computes towers + population + site-scoring model -> saves result FILES "
              "-> Streamlit reads the files -> Dashboard in the browser (map, gaps, recommended sites).")
    doc.add_paragraph("The four steps delivered:")
    bullet(doc, "Step 1 - See the network: map all towers + coverage footprints.")
    bullet(doc, "Step 2 - Find the gap: population vs coverage -> unserved people & villages.")
    bullet(doc, "Step 3 - Spot the strain: flag overloaded / weak towers (load proxy).")
    bullet(doc, "Step 4 - Pick the site: rank new-tower sites by unserved population + light demand "
                "(OSM POIs) + a simple backhaul proxy.")

    h2(doc, "4. August MVP vs full submitted vision")
    table(doc,
          ["Full vision (submitted objective)", "August MVP (for selection)", "Later (Sep finale)"],
          AUG_VS_FULL,
          widths=[2.9, 2.5, 1.5])

    h2(doc, "5. Scope guardrails")
    bullet(doc, "August = get selected. Success = a working dashboard + a tight 5-min pitch.", "IN scope: ")
    bullet(doc, "LSTM, GNN, FastAPI, PostGIS, MapLibre, native mobile, full demand/backhaul modelling.",
           "OUT of August scope (-> September finale): ")
    bullet(doc, "Keep demand + backhaul as LIGHT proxies so the pitch still matches the full objective.",
           "Rule: ")
    bullet(doc, "Nobody starts a September/stretch item until the MVP is built AND deployed.", "Hard rule: ")

    h2(doc, "6. Team & roles (August)")
    table(doc,
          ["Member", "August focus"],
          [["Sroas Thavrith (Lead)", "Data fusion, Steps 1-2, site-scoring logic (Step 4), integration, deck & pitch"],
           ["Kunvuth Sereyrith", "Data engineering: tower clean, OSM villages/POIs, admin boundaries"],
           ["Phylong (Data Science)", "Step 3 strain, feature engineering, site-scoring model + evaluation"],
           ["Sonavin (Full Stack)", "Repo, Streamlit app skeleton, deployment & hosting"],
           ["Hean Mengfong", "Dashboard UI (the 4 tabs), polish, demo video"]],
          widths=[1.9, 5.0])

    h2(doc, "7. August task list (critical path - Core & Gate only)")
    sub(doc, "Stretch / September items are intentionally excluded. Status: Todo / In progress / Done.")
    aug_tasks = [
        ["G1", "Bootcamp Certified Assessment (ALL members)", "All", "9 Aug", "GATE", "Todo"],
        ["G2", "Confirm Project Setup approval on CollabHub", "Sroas Thavrith", "6 Aug", "GATE", "Todo"],
        ["A1", "Review submitted Project Setup details w/ team", "All", "5 Aug", "Core", "Todo"],
        ["A2", "Select mentor + refine scope/methodology/data", "Sroas Thavrith", "5 Aug", "Core", "Todo"],
        ["A3", "Create CollabHub repo + folder structure", "Sonavin", "5 Aug", "Core", "Todo"],
        ["A4", "Set up Colab + Google Drive + GEE signup", "All", "5 Aug", "Core", "Todo"],
        ["A5", "Fix deck theme to 'Industry Improvement'", "Sroas Thavrith", "5 Aug", "Core", "Todo"],
        ["D1", "Tower data -> Penang (clean)", "Kunvuth Sereyrith", "5 Aug", "Core", "Done"],
        ["D2", "Population raster -> Penang clip", "Kunvuth Sereyrith", "5 Aug", "Core", "Done"],
        ["D3", "OSM villages / POIs / roads (Penang)", "Kunvuth Sereyrith", "7 Aug", "Core", "Todo"],
        ["D4", "Admin boundaries (Penang)", "Kunvuth Sereyrith", "7 Aug", "Core", "Todo"],
        ["D5", "Data fusion + analysis grid (single CRS)", "Sroas Thavrith", "8 Aug", "Core", "Todo"],
        ["M1", "Step 1 - tower + coverage map", "Sroas Thavrith", "5 Aug", "Core", "Done"],
        ["M2", "Step 2 - coverage gap vs population", "Sroas Thavrith", "6 Aug", "Core", "In progress"],
        ["M3", "Step 3 - overloaded/weak tower flags", "Phylong", "8 Aug", "Core", "Todo"],
        ["M4", "Demand & feature engineering (POIs, load, density)", "Phylong", "8 Aug", "Core", "Todo"],
        ["M5", "Step 4 - site-scoring model (MCDA)", "Phylong / Sroas Thavrith", "9 Aug", "Core", "Todo"],
        ["M6", "Step 4 - ranked site shortlist + score breakdown", "Sroas Thavrith", "9 Aug", "Core", "Todo"],
        ["M7", "Model evaluation + charts for deck", "Phylong", "10 Aug", "Core", "Todo"],
        ["M8", "Validation & sanity check vs Penang geography", "All", "10 Aug", "Core", "Todo"],
        ["F1", "Streamlit app skeleton (layout, map, KPIs, tabs)", "Sonavin / Hean Mengfong", "8 Aug", "Core", "Todo"],
        ["F2", "Tab 1-4 (network, gap, strain, sites)", "Hean Mengfong", "10 Aug", "Core", "Todo"],
        ["F3", "Polish & branding (legend, tooltips, layout)", "Hean Mengfong", "11 Aug", "Core", "Todo"],
        ["F4", "Deploy dashboard -> public URL", "Sonavin", "11 Aug", "Core", "Todo"],
        ["I1", "End-to-end integration (data -> app)", "Sroas Thavrith", "10 Aug", "Core", "Todo"],
        ["I2", "Reproducibility (requirements, README)", "Sonavin", "11 Aug", "Core", "Todo"],
        ["I3", "QA / bug bash on fresh machine", "All", "11 Aug", "Core", "Todo"],
        ["I4", "Push repository snapshot to CollabHub", "Sonavin", "11 Aug", "Core", "Todo"],
        ["P1", "Update pitch deck (framing, results, data slide)", "Sroas Thavrith", "11 Aug", "Core", "Todo"],
        ["P2", "Speaker script / 5-min talk track", "Sroas Thavrith", "11 Aug", "Core", "Todo"],
        ["P3", "Record demo video (fallback if live app fails)", "Hean Mengfong", "11 Aug", "Core", "Todo"],
        ["P4", "Datasets bundle + declarations", "Sroas Thavrith", "11 Aug", "Core", "Todo"],
        ["P5", "Pitch dry-run (timed)", "All", "12 Aug", "Core", "Todo"],
        ["S1", "Complete Virtual Submission on CollabHub", "Sroas Thavrith", "12 Aug", "GATE", "Todo"],
        ["S2", "Pitch & judging", "All", "12-13 Aug", "GATE", "Todo"],
    ]
    table(doc,
          ["ID", "Task", "Owner", "Due", "Priority", "Status"],
          aug_tasks,
          widths=[0.4, 3.0, 1.7, 0.8, 0.7, 0.8])

    h2(doc, "8. The critical path (if time gets tight, protect THESE)")
    doc.add_paragraph("The minimum that still makes a complete, pitchable entry:")
    bullet(doc, "D1 -> D2 -> M1 (network map) -> M2 (gap) -> M5/M6 (site recommendation) -> "
                "F1-F4 (dashboard + deploy) -> P1/P2 (deck + pitch) -> S1 (submit).")
    doc.add_paragraph("Everything else is enhancement. Assessments (G1) and submission (S1/S2) are "
                      "non-negotiable gates.")

    h2(doc, "9. Timeline to 12-13 Aug")
    table(doc,
          ["Date", "Target"],
          [["5 Aug", "Setup done; mentor booked; repo + Colab ready"],
           ["7 Aug", "All data collected & fused"],
           ["9 Aug", "Steps 1-4 producing numbers; assessments submitted (buffer to 11th)"],
           ["10 Aug", "Dashboard tabs working end-to-end"],
           ["11 Aug", "App deployed; deck + script + demo done; QA passed"],
           ["12 Aug", "Virtual Submission uploaded; dry-run"],
           ["12-13 Aug", "Pitch & judging"]],
          widths=[1.1, 5.8])

    h2(doc, "10. Virtual Submission checklist")
    for item in [
        "Project Setup approved (unlocks submission)",
        "Bootcamp assessments - all members",
        "Pitch deck (correct framing, results, data sources)",
        "Repository snapshot on CollabHub",
        "Demo (video + live dashboard link)",
        "Datasets + provenance/licenses",
        "Declarations (originality, data, AI-use)",
        "5-min speaker script",
    ]:
        bullet(doc, item)

    h2(doc, "11. August tech stack (only what we need now)")
    table(doc,
          ["Layer", "August tool", "Note"],
          [["Frontend", "Streamlit", "browser web app"],
           ["Backend", "(built into Streamlit)", "no separate server"],
           ["Database", "data files (CSV/GeoJSON/GeoTIFF)", "no DB needed"],
           ["Analysis / model", "Python + pandas/geopandas/rasterio/shapely (Colab)", "site-scoring = MCDA"],
           ["Maps", "Folium", "inside Streamlit"],
           ["Data prep", "QGIS (OSM extract)", "desktop"],
           ["Version control", "Git on CollabHub", "repo snapshot"]],
          widths=[1.4, 3.4, 2.1])

    out = ROOT / "AGAIF2026_SiteSense5G_AugustMVP_Plan_KH-002.docx"
    doc.save(out)
    return out


def main():
    o1 = build_onboarding()
    o2 = build_mvp_plan()
    print("Saved:")
    print(" ", o1)
    print(" ", o2)


if __name__ == "__main__":
    main()
