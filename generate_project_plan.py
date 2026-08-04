"""
Generate the SiteSense 5G team project plan as a formatted .xlsx workbook.
Team Neural Shield (KH-002) - ASEAN GeoAI Fusion 2026.

Run: py generate_project_plan.py
Output: ../AGAIF2026_SiteSense5G_ProjectPlan_KH-002.xlsx
"""

from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

OUT = Path(__file__).resolve().parent.parent / "AGAIF2026_SiteSense5G_ProjectPlan_KH-002.xlsx"

# ---- palette ----
NAVY = "1F3864"
BLUE = "2E5F9E"
LIGHT = "DCE6F1"
GREEN = "C6EFCE"      # Core / done
AMBER = "FFEB9C"      # Stretch
RED = "FFC7CE"        # Gate / critical
GREY = "F2F2F2"
WHITE = "FFFFFF"

HDR_FONT = Font(bold=True, color=WHITE, size=11, name="Calibri")
TITLE_FONT = Font(bold=True, color=NAVY, size=16, name="Calibri")
SUB_FONT = Font(italic=True, color="595959", size=10)
CELL_FONT = Font(size=10, name="Calibri")
BOLD = Font(bold=True, size=10, name="Calibri")

thin = Side(style="thin", color="BFBFBF")
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
WRAP = Alignment(wrap_text=True, vertical="top")
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)


def hdr_fill():
    return PatternFill("solid", fgColor=BLUE)


def style_header_row(ws, row, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = HDR_FONT
        cell.fill = hdr_fill()
        cell.alignment = CENTER
        cell.border = BORDER


def write_table(ws, start_row, headers, rows, widths, prio_col=None, status_col=None):
    # header
    for i, h in enumerate(headers, 1):
        ws.cell(row=start_row, column=i, value=h)
    style_header_row(ws, start_row, len(headers))
    # rows
    r = start_row + 1
    for row in rows:
        for i, val in enumerate(row, 1):
            cell = ws.cell(row=r, column=i, value=val)
            cell.font = CELL_FONT
            cell.alignment = WRAP
            cell.border = BORDER
        # priority colouring
        if prio_col is not None:
            pv = str(row[prio_col - 1]).lower()
            fill = None
            if "gate" in pv:
                fill = PatternFill("solid", fgColor=RED)
            elif "core" in pv:
                fill = PatternFill("solid", fgColor=GREEN)
            elif "stretch" in pv:
                fill = PatternFill("solid", fgColor=AMBER)
            if fill:
                ws.cell(row=r, column=prio_col).fill = fill
        if status_col is not None:
            sv = str(row[status_col - 1]).lower()
            if "done" in sv:
                ws.cell(row=r, column=status_col).fill = PatternFill("solid", fgColor=GREEN)
            elif "progress" in sv or "draft" in sv:
                ws.cell(row=r, column=status_col).fill = PatternFill("solid", fgColor=AMBER)
        r += 1
    # widths
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    return r


# =====================================================================
# 1. OVERVIEW
# =====================================================================
def sheet_overview(wb):
    ws = wb.active
    ws.title = "Overview"
    ws.sheet_view.showGridLines = False
    ws["A1"] = "SiteSense 5G — Team Project Plan"
    ws["A1"].font = TITLE_FONT
    ws["A2"] = "GeoAI for Telecommunication Tower Site Selection & 5G Coverage-Gap Planning"
    ws["A2"].font = SUB_FONT
    ws.merge_cells("A1:F1"); ws.merge_cells("A2:F2")

    info = [
        ("Team", "Neural Shield  (KH-002, Cambodia)"),
        ("Members", "Sroas Thavrith (Lead) · Kunvuth Sereyrith · Hean Mengfong · Phylong · Sonavin"),
        ("Roles", "Sroas Thavrith = lead / data fusion / site-scoring / pitch  |  "
                  "Kunvuth Sereyrith = data engineering (towers, terrain/GEE, boundaries)  |  "
                  "Hean Mengfong = frontend/dashboard & demo  |  "
                  "Phylong = data science / ML models  |  "
                  "Sonavin = full-stack (repo, backend, DB, deploy)"),
        ("Note on team size", "CollabHub team is locked at 3 (2-3 rule). Phylong & Sonavin are work-plan contributors; confirm their status for the formal submission."),
        ("Challenge theme", "Industry Improvement (as submitted) — confirm vs deck"),
        ("Scope", "Penang Island, Malaysia (committed). Cambodia/ASEAN = future-replication note only."),
        ("Core question", "Where should the next 5G tower go?"),
        ("5G framing", "GREENFIELD: Penang has 0 open-data 5G cells. Use 4G + population to find where the FIRST 5G towers go."),
        ("The 4 steps", "1) See the network  2) Find the gap  3) Spot the strain  4) Pick the site"),
        ("Chosen demo stack", "Python + Folium/Streamlit (MVP). PostGIS / FastAPI / MapLibre = stretch."),
        ("Required deliverable", "Per Secretariat (email 3 Aug): a DIGITAL PROOF OF CONCEPT - web/mobile app, dashboard, interactive map, GeoAI model, decision-support tool, or digital mock-up. Our Streamlit dashboard + interactive map + site-scoring GeoAI model fits directly."),
        ("PRIMARY DELIVERABLE", "A WEB APP: an interactive Streamlit dashboard (browser-based, opens at a URL) that is a DECISION-SUPPORT TOOL built around an INTERACTIVE MAP and powered by a GeoAI SITE-SCORING MODEL. One build = 4 of the 6 categories. NOT a native mobile app (but works in a phone browser)."),
        ("Fallback deliverable", "If the live web app is not fully stable by demo day: a polished screen-recording / clickable mock-up of the same dashboard (task P6-3). Never demo nothing."),
        ("Scope status", "Direction need NOT be final now. Refine scope/methodology/data/approach WITH the mentor after 5 Aug. Greenfield-5G framing = strong working direction, reviewable with mentor."),
        ("TWO PHASES", "Phase 3 VIRTUAL HACKATHON 12-13 Aug = BUILD & PITCH -> finalist SELECTION (build MVP + 5-min pitch). Phase 4 PHYSICAL FINALE 19-22 Sep 2026, Malaysia = SHOWCASE -> finalist refinement + awards. Two finish lines, not one."),
        ("AUGUST vs SEPTEMBER", "AUGUST (get selected) = MVP: Streamlit dashboard reading files, Steps 1->2->4, light demand/backhaul, NO separate backend/DB. SEPTEMBER (finale) = the Stretch items: trained LSTM, GNN, FastAPI, PostGIS, MapLibre, mobile polish, full demand/backhaul. HARD RULE: do not start a September/Stretch item until the August MVP is built AND deployed."),
        ("August MVP detail", "See the dedicated file: AGAIF2026_SiteSense5G_AugustMVP_Plan_KH-002.docx (August critical-path tasks only). Onboarding: TEAM_ONBOARDING_SiteSense5G.docx."),
        ("Hackathon", "Virtual Submission 12 Aug · Pitch & judging 12-13 Aug 2026 (SELECTION) · Physical Finale 19-22 Sep 2026, Malaysia (AWARDS)"),
        ("Plan created", "2026-08-03 · updated 4 Aug (Secretariat email, two-phase split, faithful objective)"),
    ]
    r = 4
    for k, v in info:
        ws.cell(row=r, column=1, value=k).font = BOLD
        ws.cell(row=r, column=1).fill = PatternFill("solid", fgColor=LIGHT)
        ws.cell(row=r, column=1).border = BORDER
        c = ws.cell(row=r, column=2, value=v)
        c.font = CELL_FONT; c.alignment = WRAP; c.border = BORDER
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=6)
        r += 1

    r += 1
    ws.cell(row=r, column=1, value="How to use this workbook").font = BOLD
    r += 1
    notes = [
        "• 'Task Plan' is the master list. Update the Status column daily (Todo / In progress / Done).",
        "• Priority: GATE = mandatory hackathon requirement (do not slip) · Core = MVP · Stretch = if time allows.",
        "• MVP = Steps 1,2,4 in a Streamlit app + updated deck + demo. That alone is a complete, pitchable project.",
        "• Reassign Owner names freely — they are suggestions based on a natural split of work.",
        "• DONE so far: Step 1 tower map, WorldPop population download + Penang clip, Step 2 gap script (drafted).",
    ]
    for n in notes:
        c = ws.cell(row=r, column=1, value=n); c.font = CELL_FONT; c.alignment = WRAP
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=6)
        r += 1

    ws.column_dimensions["A"].width = 20
    for col in "BCDEF":
        ws.column_dimensions[col].width = 20


# =====================================================================
# 2. MILESTONES
# =====================================================================
def sheet_milestones(wb):
    ws = wb.create_sheet("Milestones")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Milestones & Timeline"; ws["A1"].font = TITLE_FONT
    headers = ["Date (2026)", "Milestone", "What it means for us", "Priority"]
    rows = [
        ["Mon 3 Aug", "TODAY — plan locked", "Kick off build. Steps 1-2 already started.", "Core"],
        ["Tue 5 Aug", "Mentor Selection opens", "Pick a GIS/telecom or ML mentor; bring 3 sharp questions.", "Core"],
        ["Fri 8 Aug", "Internal target: data + analysis", "Steps 1-4 producing numbers; dashboard skeleton up.", "Core"],
        ["Sat 9 Aug", "Internal target: assessments done", "Finish Bootcamp assessment early (2-day buffer).", "Gate"],
        ["Mon 11 Aug", "Bootcamp Assessment DEADLINE", "All 3 members must submit in CollabHub. Hard gate.", "Gate"],
        ["Mon 11 Aug", "Internal target: app + deck ready", "Dashboard deployed, deck updated, demo recorded.", "Core"],
        ["Tue 12 Aug", "Virtual Submission", "Upload deck + repo snapshot + demo + datasets + declarations.", "Gate"],
        ["12-13 Aug", "Virtual Hackathon — BUILD & PITCH", "Submit + 5-min pitch -> FINALIST SELECTION. (Rounds: Technical Screening -> Demo Shortlist -> Finalists.)", "Gate"],
        ["19-22 Sep", "Physical Finale — SHOWCASE (Malaysia)", "If selected: finalist refinement, public showcase, judging, AWARDS. This is where Stretch items (LSTM/GNN/FastAPI/PostGIS/MapLibre) land.", "Core"],
    ]
    end = write_table(ws, 3, headers, rows, [14, 32, 55, 12], prio_col=4)
    ws.freeze_panes = "A4"
    ws.auto_filter.ref = f"A3:D{end-1}"


# =====================================================================
# 3. TASK PLAN  (master)
# =====================================================================
def sheet_tasks(wb):
    ws = wb.create_sheet("Task Plan")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Master Task Plan"; ws["A1"].font = TITLE_FONT
    headers = ["ID", "Phase", "Task", "Description / Guide", "Owner",
               "Priority", "Est. days", "Depends on", "Technology & Tools",
               "Deliverable", "Due", "Status"]
    T = [
        # ---- Phase 0: Setup & Admin ----
        ["P0-1", "0 Setup & Admin", "Confirm Project Setup approval",
         "Watch CollabHub; Project Setup is 'Submitted - Pending approval'. Ping Secretariat if not approved by 6 Aug. Project Submission stays LOCKED until approved.",
         "Sroas Thavrith", "Gate", 0.5, "-", "CollabHub", "Approved status", "6 Aug", "Todo"],
        ["P0-2", "0 Setup & Admin", "Bootcamp Certified Assessment (ALL members)",
         "Each member completes & submits all required assessments in CollabHub > Certified Assessment. Do early.",
         "All", "Gate", 1.0, "-", "CollabHub Learning", "All submitted assessments", "9 Aug", "Todo"],
        ["P0-3", "0 Setup & Admin", "Select mentor + prep questions",
         "When roster opens 5 Aug, request a GIS/telecom or ML mentor. Discuss & refine SCOPE, METHODOLOGY, DATA and DEVELOPMENT APPROACH (Secretariat says project may be refined after mentor guidance). Validate greenfield-5G framing. Prep 3-4 questions.",
         "Sroas Thavrith", "Core", 0.5, "P0-7", "CollabHub Consultations", "Mentor booked + notes", "5 Aug", "Todo"],
        ["P0-7", "0 Setup & Admin", "Review submitted Project Setup details",
         "Team review of submitted title, description, problem statement, proposed solution, and required data/tools (Secretariat-recommended pre-work). Align deck + plan; list any refinements to raise with mentor.",
         "All", "Core", 0.3, "-", "CollabHub / docs", "Reviewed + refinement list", "5 Aug", "Todo"],
        ["P0-4", "0 Setup & Admin", "Create CollabHub Git repo + structure",
         "Create repo 'sitesense-5g'. Folders: data/ notebooks/ src/ app/ docs/ outputs/. Add README, .gitignore, requirements.txt.",
         "Sonavin", "Core", 0.5, "-", "Git (geoai-collabhub.com)", "Repo scaffolded", "5 Aug", "Todo"],
        ["P0-5", "0 Setup & Admin", "Set up shared dev environment",
         "Google account for Colab; sign up Google Earth Engine (approval can take a day - do NOW); shared Google Drive folder for big data.",
         "All", "Core", 0.5, "-", "Colab, GEE, Google Drive", "Accounts ready", "5 Aug", "Todo"],
        ["P0-6", "0 Setup & Admin", "Confirm challenge theme vs deck",
         "Team Formation shows 'Cybersecurity' (legacy); Project Setup shows 'Industry Improvement'. Make deck + form consistent.",
         "Sroas Thavrith", "Core", 0.2, "-", "CollabHub / PowerPoint", "Consistent theme", "5 Aug", "Todo"],

        # ---- Phase 1: Data ----
        ["P1-1", "1 Data & Fusion", "Tower data -> Penang (clean)",
         "Load 502.csv, filter bbox lat 5.1-5.6/lon 100.1-100.6, dedupe, cap absurd 'range' values.",
         "Kunvuth Sereyrith", "Core", 0.5, "P0-4", "Python, pandas", "penang_towers.csv", "5 Aug", "Done"],
        ["P1-2", "1 Data & Fusion", "Population raster -> Penang clip",
         "WorldPop MYS 100m 2020 UN-adj; clip to Penang bbox; cache small GeoTIFF.",
         "Kunvuth Sereyrith", "Core", 0.5, "-", "rasterio, WorldPop", "penang_pop_100m.tif", "5 Aug", "Done"],
        ["P1-3", "1 Data & Fusion", "OSM villages/roads/buildings",
         "Extract Penang settlements (kampung/places), roads, buildings via QGIS QuickOSM or Geofabrik extract.",
         "Kunvuth Sereyrith", "Core", 1.0, "P0-5", "QGIS, QuickOSM, geopandas", "penang_places.geojson", "7 Aug", "Todo"],
        ["P1-4", "1 Data & Fusion", "Terrain / elevation (GEE SRTM)",
         "Export SRTM 30m DEM for Penang from Earth Engine; used to refine coverage & as tower-height proxy.",
         "Kunvuth Sereyrith", "Stretch", 1.0, "P0-5", "Google Earth Engine", "penang_dem.tif", "8 Aug", "Todo"],
        ["P1-5", "1 Data & Fusion", "Admin boundaries",
         "GADM Malaysia L2 (Penang districts) OR reuse Bootcamp Place.shp/Asean.shp. Clip everything to Penang.",
         "Hean Mengfong", "Core", 0.5, "-", "GADM, geopandas", "penang_admin.geojson", "7 Aug", "Todo"],
        ["P1-6", "1 Data & Fusion", "Drive-test RSRP prep",
         "Clean raw_dataset.csv (RSRP) for signal calibration + LSTM input.",
         "Kunvuth Sereyrith", "Stretch", 0.5, "-", "pandas", "rsrp_clean.csv", "8 Aug", "Todo"],
        ["P1-7", "1 Data & Fusion", "Data fusion & analysis grid",
         "Align all layers (EPSG:4326 + local metric); build analysis grid; assign canonical tower IDs. #1 telecom-GIS pitfall = CRS mismatch.",
         "Sroas Thavrith", "Core", 1.0, "P1-1,P1-2,P1-3", "geopandas, shapely, numpy", "fused grid", "8 Aug", "Todo"],
        ["P1-8", "1 Data & Fusion", "Load into PostGIS (optional)",
         "Stand up PostgreSQL+PostGIS (Docker or Supabase); load fused layers for spatial queries.",
         "Sonavin", "Stretch", 1.0, "P1-7", "PostgreSQL, PostGIS", "spatial DB", "9 Aug", "Todo"],

        # ---- Phase 2: Analysis / Modeling ----
        ["P2-1", "2 Analysis (4 steps)", "Step 1 - See the network",
         "Interactive map of all towers + coverage circles per radio generation.",
         "Sroas Thavrith", "Core", 0.5, "P1-1", "folium, pandas", "step1 map html", "5 Aug", "Done"],
        ["P2-2", "2 Analysis (4 steps)", "Step 2 - Find the gap",
         "Overlay population on 4G footprint; count unserved people & villages outside coverage.",
         "Sroas Thavrith", "Core", 1.0, "P1-2,P1-7", "rasterio, shapely", "gap map + stats", "6 Aug", "In progress"],
        ["P2-3", "2 Analysis (4 steps)", "Coverage model refinement",
         "Improve footprints beyond raw 'range' using terrain (DEM) + RSRP calibration.",
         "Phylong", "Stretch", 1.5, "P1-4,P1-6", "Python, GEE", "coverage polygons", "9 Aug", "Todo"],
        ["P2-4", "2 Analysis (4 steps)", "Step 3 - Spot the strain",
         "Flag overloaded/weak towers via 'samples' load proxy + RSRP percentiles.",
         "Phylong", "Core", 1.0, "P1-1,P1-6", "pandas, scikit-learn", "strain layer", "8 Aug", "Todo"],
        ["P2-5", "2 Analysis (4 steps)", "LSTM RSRP prediction (optional)",
         "Predict RSRP/signal from drive-test time series (Bootcamp 2 notebook).",
         "Phylong", "Stretch", 1.5, "P1-6", "PyTorch, Colab (GPU)", "LSTM model + plot", "10 Aug", "Todo"],
        ["P2-6", "2 Analysis (4 steps)", "GNN network optimisation (optional)",
         "Model towers as a graph for network optimisation (Bootcamp 2 notebook).",
         "Phylong", "Stretch", 1.5, "P1-7", "PyTorch Geometric, Colab", "GNN demo", "10 Aug", "Todo"],
        ["P2-7", "2 Analysis (4 steps)", "Step 4 - Site-scoring model (MCDA)",
         "Score candidate grid cells: unserved population + demand + backhaul feasibility + terrain. Explainable weights.",
         "Phylong / Sroas Thavrith", "Core", 1.5, "P2-2,P2-4,P2-10", "Python, scikit-learn, numpy", "score raster", "9 Aug", "Todo"],
        ["P2-8", "2 Analysis (4 steps)", "Step 4 - Ranked site shortlist",
         "Output top-N new 5G tower sites with a per-site coverage-gain / score breakdown.",
         "Sroas Thavrith", "Core", 0.5, "P2-7", "Python", "sites.geojson + table", "9 Aug", "Todo"],
        ["P2-9", "2 Analysis (4 steps)", "Validation & sanity check",
         "Compare results to known Penang geography (Georgetown dense, hills empty). Document assumptions/caveats.",
         "All", "Core", 0.5, "P2-8", "QGIS, notebook", "validation notes", "10 Aug", "Todo"],
        ["P2-10", "2 Analysis (4 steps)", "Demand & feature engineering",
         "Derive demand proxy + model features (population density, tower load 'samples', OSM POIs/roads) for the site-scoring model.",
         "Phylong", "Core", 1.0, "P1-7", "pandas, scikit-learn, numpy", "feature table", "8 Aug", "Todo"],
        ["P2-11", "2 Analysis (4 steps)", "Model evaluation & explainability",
         "Sensitivity of scoring weights, feature importance, metrics + charts for the deck.",
         "Phylong", "Core", 0.5, "P2-7", "scikit-learn, matplotlib", "eval report + charts", "10 Aug", "Todo"],

        # ---- Phase 3: Backend ----
        ["P3-1", "3 Backend / API", "Design API contract (optional)",
         "Define endpoints: /towers /gap /strain /sites. MVP Streamlit can read precomputed files instead - only do this if going full web stack.",
         "Sonavin", "Stretch", 0.5, "P2-8", "FastAPI", "API spec", "9 Aug", "Todo"],
        ["P3-2", "3 Backend / API", "Implement FastAPI service (optional)",
         "Serve precomputed GeoJSON/stats via FastAPI + uvicorn.",
         "Sonavin", "Stretch", 1.0, "P3-1", "FastAPI, uvicorn", "running API", "10 Aug", "Todo"],
        ["P3-3", "3 Backend / API", "Deploy backend (optional)",
         "Deploy to Render / Railway / HuggingFace Spaces free tier.",
         "Sonavin", "Stretch", 0.5, "P3-2", "Render/Railway/HF", "public API URL", "11 Aug", "Todo"],
        ["P3-4", "3 Backend / API", "Frontend-backend integration (optional)",
         "Wire the dashboard to the API/PostGIS; env config, CORS, error handling.",
         "Sonavin", "Stretch", 0.5, "P3-2,P4-5", "FastAPI, Streamlit", "integrated app", "11 Aug", "Todo"],

        # ---- Phase 4: Frontend ----
        ["P4-1", "4 Frontend / Dashboard", "Streamlit app skeleton",
         "Layout, sidebar controls, map component, KPI cards, tabs for the 4 steps.",
         "Hean Mengfong / Sonavin", "Core", 1.0, "P2-1", "Streamlit, folium", "app skeleton", "8 Aug", "Todo"],
        ["P4-2", "4 Frontend / Dashboard", "Tab 1 - Network view",
         "Embed Step 1 map; radio-generation toggles.",
         "Hean Mengfong", "Core", 0.5, "P4-1,P2-1", "Streamlit", "tab 1", "9 Aug", "Todo"],
        ["P4-3", "4 Frontend / Dashboard", "Tab 2 - Coverage gap",
         "Population overlay + unserved KPIs + gap map.",
         "Hean Mengfong", "Core", 0.5, "P4-1,P2-2", "Streamlit", "tab 2", "9 Aug", "Todo"],
        ["P4-4", "4 Frontend / Dashboard", "Tab 3 - Tower strain",
         "Show overloaded/weak towers.",
         "Hean Mengfong", "Core", 0.5, "P4-1,P2-4", "Streamlit", "tab 3", "10 Aug", "Todo"],
        ["P4-5", "4 Frontend / Dashboard", "Tab 4 - Recommended sites",
         "Map + table of top-N sites with score breakdown; the money shot for judges.",
         "Hean Mengfong", "Core", 0.5, "P4-1,P2-8", "Streamlit", "tab 4", "10 Aug", "Todo"],
        ["P4-6", "4 Frontend / Dashboard", "Polish & branding",
         "Legend, tooltips, colours, title, responsive layout, SiteSense branding.",
         "Hean Mengfong", "Core", 0.5, "P4-5", "Streamlit, CSS", "polished UI", "11 Aug", "Todo"],
        ["P4-7", "4 Frontend / Dashboard", "MapLibre web frontend (optional)",
         "Richer web map if going beyond Streamlit.",
         "Sonavin", "Stretch", 1.5, "P3-2", "MapLibre GL JS", "web map", "11 Aug", "Todo"],
        ["P4-8", "4 Frontend / Dashboard", "Deploy dashboard",
         "Streamlit Community Cloud or HuggingFace Spaces; get public link for demo.",
         "Sonavin", "Core", 0.5, "P4-6", "Streamlit Cloud / HF", "public app URL", "11 Aug", "Todo"],

        # ---- Phase 5: Integration & QA ----
        ["P5-1", "5 Integration & QA", "End-to-end integration",
         "One flow raw data -> analysis -> dashboard; ensure numbers match across steps.",
         "Sroas Thavrith", "Core", 1.0, "P2-8,P4-5", "Python", "working pipeline", "10 Aug", "Todo"],
        ["P5-2", "5 Integration & QA", "Reproducibility",
         "requirements.txt, README run steps, config/seeds so anyone can reproduce.",
         "Sonavin", "Core", 0.5, "P5-1", "Git, pip", "reproducible repo", "11 Aug", "Todo"],
        ["P5-3", "5 Integration & QA", "QA / bug bash",
         "Test dashboard on fresh machine; verify stats, fix bugs.",
         "All", "Core", 0.5, "P5-1", "-", "QA sign-off", "11 Aug", "Todo"],
        ["P5-4", "5 Integration & QA", "Push repository snapshot",
         "Commit & push everything to CollabHub repo for the submission snapshot.",
         "Sonavin", "Core", 0.3, "P5-2", "Git", "repo snapshot", "11 Aug", "Todo"],
        ["P5-5", "5 Integration & QA", "Deployment & hosting config",
         "requirements/Dockerfile/secrets/hosting for app (+API); keep public links stable for the demo.",
         "Sonavin", "Core", 0.5, "P4-8", "Docker, Streamlit Cloud / HF", "stable hosting", "11 Aug", "Todo"],

        # ---- Phase 6: Pitch & Submission ----
        ["P6-1", "6 Pitch & Submission", "Update pitch deck",
         "Fix 5G-greenfield framing; add data-sources, methodology, results screenshots, market stat (GIS-in-telecom).",
         "Sroas Thavrith", "Core", 1.0, "P2-9,P2-11", "PowerPoint / Slides", "final deck", "11 Aug", "Todo"],
        ["P6-2", "6 Pitch & Submission", "Speaker script / talk track",
         "Per-slide script, ~5 min pitch, roles per member.",
         "Sroas Thavrith", "Core", 0.5, "P6-1", "Docs", "script", "11 Aug", "Todo"],
        ["P6-3", "6 Pitch & Submission", "Record demo video",
         "3-5 min screen recording of the dashboard walking the 4 steps.",
         "Hean Mengfong", "Core", 0.5, "P4-8", "OBS / screen recorder", "demo.mp4", "11 Aug", "Todo"],
        ["P6-4", "6 Pitch & Submission", "Datasets bundle + declarations",
         "List data + licenses (WorldPop CC-BY, OSM ODbL, OpenCelliD). Fill originality/AI-use/data declarations.",
         "Sroas Thavrith", "Core", 0.5, "P5-4", "CollabHub", "declarations", "11 Aug", "Todo"],
        ["P6-5", "6 Pitch & Submission", "Complete Virtual Submission",
         "Upload deck + repo snapshot + demo + datasets + declarations in CollabHub Project Submission.",
         "Sroas Thavrith", "Gate", 0.5, "P6-1,P6-3,P6-4", "CollabHub", "submitted", "12 Aug", "Todo"],
        ["P6-6", "6 Pitch & Submission", "Pitch dry-run",
         "Timed rehearsal; refine cuts.",
         "All", "Core", 0.5, "P6-2", "-", "rehearsed", "12 Aug", "Todo"],
        ["P6-7", "6 Pitch & Submission", "Hackathon pitch & judging",
         "Present; respond to judges across the 3 rounds.",
         "All", "Gate", 1.0, "P6-5,P6-6", "-", "pitched", "12-13 Aug", "Todo"],
    ]
    widths = [7, 20, 26, 52, 22, 9, 8, 16, 26, 22, 10, 12]
    end = write_table(ws, 3, headers, T, widths, prio_col=6, status_col=12)
    ws.freeze_panes = "C4"
    ws.auto_filter.ref = f"A3:L{end-1}"
    for r in range(4, end):
        ws.row_dimensions[r].height = 42


# =====================================================================
# 4. TECH STACK
# =====================================================================
def sheet_stack(wb):
    ws = wb.create_sheet("Tech Stack")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Technology Stack"; ws["A1"].font = TITLE_FONT
    headers = ["Layer", "Technology", "Purpose", "Owner", "Where it runs", "Cost", "Priority"]
    rows = [
        ["Data processing", "Python + pandas / geopandas / rasterio / shapely / numpy",
         "Clean, clip, fuse and analyse all layers; the 4 steps.", "Sroas Thavrith / Kunvuth Sereyrith", "Local + Colab", "Free", "Core"],
        ["Terrain & satellite", "Google Earth Engine",
         "SRTM DEM + land cover for coverage refinement & site scoring.", "Kunvuth Sereyrith", "earthengine.google.com (browser)", "Free (signup, ~1 day approval)", "Stretch"],
        ["Notebooks / compute", "Google Colab",
         "Run ML notebooks (LSTM / GNN) with free GPU; share with team.", "Phylong", "colab.research.google.com", "Free / Pro", "Stretch"],
        ["Machine learning", "scikit-learn / PyTorch / PyTorch Geometric",
         "MCDA site scoring (sklearn/numpy); LSTM RSRP; GNN optimisation.", "Phylong", "Colab", "Free", "Core (sklearn) / Stretch (DL)"],
        ["Spatial database", "PostgreSQL + PostGIS",
         "Store & query fused spatial layers (optional for MVP).", "Sonavin", "Docker local / Supabase", "Free", "Stretch"],
        ["Backend API", "FastAPI + uvicorn",
         "Serve precomputed GeoJSON/stats to the frontend (optional).", "Sonavin", "Render / Railway / HF Spaces", "Free tier", "Stretch"],
        ["Frontend (MVP)", "Streamlit + Folium",
         "Interactive demo dashboard - the chosen path.", "Hean Mengfong", "Streamlit Community Cloud", "Free", "Core"],
        ["Frontend (alt)", "MapLibre GL JS",
         "Richer custom web map if going beyond Streamlit.", "Sonavin", "Vercel / Netlify", "Free", "Stretch"],
        ["Desktop GIS", "QGIS",
         "OSM extraction, data prep, validation, static map exports for deck.", "Hean Mengfong", "Local", "Free", "Core"],
        ["Version control", "Git on CollabHub",
         "Code + collaboration + required repository snapshot.", "Sonavin", "geoai-collabhub.com", "Free", "Core"],
        ["Deck & demo", "PowerPoint / Google Slides + screen recorder",
         "Pitch deck and demo video.", "Sroas Thavrith / Hean Mengfong", "Local", "Free", "Core"],
    ]
    end = write_table(ws, 3, headers, rows, [16, 34, 40, 22, 26, 20, 14], prio_col=7)
    ws.freeze_panes = "A4"
    ws.auto_filter.ref = f"A3:G{end-1}"
    for r in range(4, end):
        ws.row_dimensions[r].height = 40


# =====================================================================
# 5. DATA SOURCES
# =====================================================================
def sheet_data(wb):
    ws = wb.create_sheet("Data Sources")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Data Sources"; ws["A1"].font = TITLE_FONT
    headers = ["Layer", "Dataset", "Source", "License", "Status", "Used in"]
    rows = [
        ["Cell towers", "OpenCelliD 502.csv (MCC 502, Malaysia)", "opencellid.org", "OpenCelliD / CC-BY-SA", "Have (5,284 Penang cells)", "Steps 1-4"],
        ["Population", "WorldPop MYS 100m 2020 UN-adj", "worldpop.org", "CC-BY 4.0", "Downloaded + clipped", "Step 2, 4"],
        ["Villages/roads/buildings", "OpenStreetMap (Penang)", "geofabrik.de / QuickOSM", "ODbL", "To download", "Steps 2, 4"],
        ["Terrain / elevation", "SRTM 30m DEM", "Google Earth Engine", "Public domain", "To export", "Steps 2, 4"],
        ["Admin boundaries", "GADM Malaysia L2 (Penang)", "gadm.org", "Free, non-commercial", "To download", "All (clip/aggregate)"],
        ["Drive-test signal", "raw_dataset.csv (RSRP)", "Bootcamp 2 Day 3", "Provided", "Have", "Step 3, LSTM"],
        ["Places (alt)", "Place.shp / Asean.shp", "Bootcamp 1 CA3", "Provided", "Have", "Village points, boundary"],
        ["Speed tiles (optional)", "Ookla Open Data", "github.com/teamookla/ookla-open-data", "CC-BY-NC-SA", "Optional", "Validation"],
    ]
    end = write_table(ws, 3, headers, rows, [22, 34, 30, 22, 22, 18], status_col=5)
    ws.freeze_panes = "A4"
    ws.auto_filter.ref = f"A3:F{end-1}"
    for r in range(4, end):
        ws.row_dimensions[r].height = 30


# =====================================================================
# 6. SUBMISSION CHECKLIST
# =====================================================================
def sheet_checklist(wb):
    ws = wb.create_sheet("Submission Checklist")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "CollabHub Virtual Submission — Checklist"; ws["A1"].font = TITLE_FONT
    ws["A2"] = "Everything below must be ready by 12 Aug. Project Submission unlocks only after Project Setup is approved."
    ws["A2"].font = SUB_FONT; ws.merge_cells("A2:D2")
    headers = ["Deliverable", "Detail", "Owner", "Done?"]
    rows = [
        ["Project Setup approved", "Secretariat approval unlocks submission.", "Sroas Thavrith", ""],
        ["Bootcamp assessments", "All registered members submitted (due 11 Aug).", "All", ""],
        ["Pitch deck", "Updated framing, methodology, results, data-sources slide.", "Sroas Thavrith", ""],
        ["Repository snapshot", "Code + notebooks pushed to CollabHub repo.", "Sonavin", ""],
        ["Demo", "3-5 min video + live dashboard link.", "Hean Mengfong", ""],
        ["Datasets", "Data files or links + provenance.", "Sroas Thavrith", ""],
        ["Declarations", "Originality, data licenses, AI-use as required.", "Sroas Thavrith", ""],
        ["Speaker script", "Talk track for the pitch.", "Sroas Thavrith", ""],
    ]
    end = write_table(ws, 3, headers, rows, [26, 48, 18, 10])
    ws.freeze_panes = "A4"
    for r in range(4, end):
        ws.row_dimensions[r].height = 26


def main():
    wb = Workbook()
    sheet_overview(wb)
    sheet_milestones(wb)
    sheet_tasks(wb)
    sheet_stack(wb)
    sheet_data(wb)
    sheet_checklist(wb)
    wb.save(OUT)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
