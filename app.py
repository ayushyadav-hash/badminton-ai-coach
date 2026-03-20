#!/usr/bin/env python3
"""
Badminton Analysis Web App
Upload video → Extract frames → Claude AI Analysis → PDF Report
"""

import os
import cv2
import base64
import json
import math
import shutil
import hashlib
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template
from anthropic import Anthropic
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether, Image as RLImage
)
from reportlab.graphics.shapes import Drawing, Rect, Line, String
from reportlab.graphics import renderPDF

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

UPLOAD_DIR  = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), 'outputs')
FRAMES_DIR  = os.path.join(os.path.dirname(__file__), 'frames')
CACHE_DIR   = os.path.join(os.path.dirname(__file__), 'cache')
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'settings.json')

for d in [UPLOAD_DIR, OUTPUT_DIR, FRAMES_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return {}


def save_settings(data):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def get_api_key():
    # Priority: env var → settings file
    key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not key:
        key = load_settings().get('api_key', '')
    return key


def get_cache_key(video_name, sport, player_name):
    raw = f"{video_name}:{sport}:{player_name}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_cache(cache_key):
    path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_cache(cache_key, analysis):
    path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    with open(path, 'w') as f:
        json.dump(analysis, f, indent=2)

# ── Colour Palette ──────────────────────────────────────────────────────────────
C_NAVY   = colors.HexColor("#0D1B2A")
C_BLUE   = colors.HexColor("#1565C0")
C_GREEN  = colors.HexColor("#2E7D32")
C_ORANGE = colors.HexColor("#E65100")
C_AMBER  = colors.HexColor("#F57F17")
C_RED    = colors.HexColor("#B71C1C")
C_LGRAY  = colors.HexColor("#F4F6F8")
C_BODY   = colors.HexColor("#212121")
C_MGRAY  = colors.HexColor("#757575")
C_BORDER = colors.HexColor("#CFD8DC")
C_WHITE  = colors.white
C_GOLD   = colors.HexColor("#FFD600")
C_COURT  = colors.HexColor("#00897B")
W, H = A4
MARGIN = 16 * mm


def to_hex(col):
    return f"{int(col.red*255):02x}{int(col.green*255):02x}{int(col.blue*255):02x}"


def score_color(s):
    if s >= 8: return C_GREEN
    if s >= 6: return C_BLUE
    if s >= 4: return C_AMBER
    return C_RED


def score_label(s):
    if s >= 9: return "Excellent"
    if s >= 7: return "Good"
    if s >= 5: return "Average"
    if s >= 3: return "Needs Work"
    return "Critical"


def make_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=24, textColor=C_WHITE, leading=30, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"],
            fontName="Helvetica", fontSize=11, textColor=colors.HexColor("#90CAF9"), leading=15, spaceAfter=3),
        "meta": ParagraphStyle("meta", parent=base["Normal"],
            fontName="Helvetica", fontSize=8.5, textColor=colors.HexColor("#B0BEC5"), leading=12),
        "section": ParagraphStyle("section", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=13, textColor=C_NAVY, leading=17, spaceBefore=12, spaceAfter=7),
        "subsection": ParagraphStyle("subsection", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=10.5, textColor=C_BLUE, leading=14, spaceBefore=7, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["Normal"],
            fontName="Helvetica", fontSize=9.5, textColor=C_BODY, leading=15, spaceAfter=5, alignment=TA_JUSTIFY),
        "body_sm": ParagraphStyle("body_sm", parent=base["Normal"],
            fontName="Helvetica", fontSize=8.5, textColor=C_BODY, leading=13, spaceAfter=4),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"],
            fontName="Helvetica", fontSize=9, textColor=C_BODY, leading=13, leftIndent=14, spaceAfter=3),
        "caption": ParagraphStyle("caption", parent=base["Normal"],
            fontName="Helvetica", fontSize=7.5, textColor=C_MGRAY, leading=10, alignment=TA_CENTER),
        "th": ParagraphStyle("th", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=8.5, textColor=C_WHITE, leading=11),
        "td": ParagraphStyle("td", parent=base["Normal"],
            fontName="Helvetica", fontSize=8.5, textColor=C_BODY, leading=12),
        "td_bold": ParagraphStyle("td_bold", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=8.5, textColor=C_BODY, leading=12),
        "quote": ParagraphStyle("quote", parent=base["Normal"],
            fontName="Helvetica-Oblique", fontSize=9.5, textColor=C_NAVY,
            leading=15, leftIndent=14, rightIndent=14, spaceAfter=5),
        "footer": ParagraphStyle("footer", parent=base["Normal"],
            fontName="Helvetica", fontSize=7, textColor=C_MGRAY, alignment=TA_CENTER),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: Home
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/settings', methods=['GET'])
def get_settings_route():
    s = load_settings()
    key = s.get('api_key', '')
    masked = (key[:8] + '...' + key[-4:]) if len(key) > 12 else ('set' if key else '')
    return jsonify({'has_key': bool(key), 'masked': masked})


@app.route('/settings', methods=['POST'])
def save_settings_route():
    data = request.json
    s = load_settings()
    if 'api_key' in data:
        s['api_key'] = data['api_key']
    save_settings(s)
    return jsonify({'ok': True})


@app.route('/cache-list', methods=['GET'])
def cache_list():
    files = []
    for f in os.listdir(CACHE_DIR):
        if f.endswith('.json'):
            path = os.path.join(CACHE_DIR, f)
            with open(path) as fp:
                d = json.load(fp)
            files.append({
                'key': f.replace('.json', ''),
                'player': d.get('player_name', '?'),
                'sport': d.get('sport', '?'),
                'score': d.get('overall_score', '?'),
                'date': d.get('cached_at', '?'),
            })
    return jsonify({'cache': files})


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: Extract Frames
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/extract', methods=['POST'])
def extract_frames():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400

    video = request.files['video']
    interval = int(request.form.get('interval', 5))

    # Clear previous frames
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    video_path = os.path.join(UPLOAD_DIR, video.filename)
    video.save(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({'error': 'Could not open video'}), 400

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    frame_interval = max(1, int(fps * interval))

    frames = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            timestamp = int(frame_num / fps)
            filename = f"frame_{timestamp:04d}s.jpg"
            path = os.path.join(FRAMES_DIR, filename)
            cv2.imwrite(path, frame)

            # Base64 encode for preview
            with open(path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()

            frames.append({
                'filename': filename,
                'timestamp': timestamp,
                'b64': b64,
                'path': path,
            })
        frame_num += 1

    cap.release()

    return jsonify({
        'frames': [{'filename': f['filename'], 'timestamp': f['timestamp'], 'b64': f['b64']} for f in frames],
        'total': len(frames),
        'duration': round(duration, 1),
        'fps': round(fps, 1),
        'video_name': video.filename,
    })


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: Analyse with Claude
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/analyse', methods=['POST'])
def analyse():
    data = request.json
    sport = data.get('sport', 'badminton')
    player_name = data.get('player_name', 'Player')
    custom_prompt = data.get('custom_prompt', '')
    video_name = data.get('video_name', 'video')
    force_refresh = data.get('force_refresh', False)

    # Check cache first
    cache_key = get_cache_key(video_name, sport, player_name)
    if not force_refresh:
        cached = load_cache(cache_key)
        if cached:
            return jsonify({'analysis': cached, 'from_cache': True})

    # Validate API key
    api_key = get_api_key()
    if not api_key:
        return jsonify({'error': 'No API key set. Click the ⚙️ Settings button and enter your Anthropic API key.'}), 400

    # Get all frames
    frame_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
    if not frame_files:
        return jsonify({'error': 'No frames found. Extract frames first.'}), 400

    try:
        client = Anthropic(api_key=api_key)

        content = []
        for fname in frame_files:
            path = os.path.join(FRAMES_DIR, fname)
            with open(path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
            ts = fname.replace('frame_', '').replace('s.jpg', '')
            content.append({"type": "text", "text": f"Frame at {ts} seconds:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})

        base_prompt = f"""You are an expert {sport} coach with 20+ years of experience.

Analyse these {len(frame_files)} video frames of {player_name} playing {sport}.

Respond ONLY with valid JSON (no markdown, no explanation):

{{
  "player_name": "{player_name}",
  "sport": "{sport}",
  "overall_score": <1-10>,
  "overall_grade": "<A/B/C/D/F>",
  "executive_summary": "<3-4 sentence assessment>",
  "skills": [
    {{"name": "<skill>", "score": <1-10>, "observation": "<detail>", "improvement": "<tip>"}}
  ],
  "frame_analysis": [
    {{"timestamp": "<Xs>", "title": "<what is happening>", "priority": "<CRITICAL/HIGH/MEDIUM/LOW>",
      "observations": ["<obs 1>", "<obs 2>"], "pro_reference": "<player name and lesson>"}}
  ],
  "top_fixes": ["<fix 1>", "<fix 2>", "<fix 3>"],
  "drills": [
    {{"name": "<name>", "duration": "<time>", "description": "<how to>", "targets": "<skill>"}}
  ],
  "strengths": ["<s1>", "<s2>", "<s3>"],
  "coach_quote": "<honest inspiring quote about this player>"
}}

{custom_prompt}

Reference world-class players. Be specific and detailed."""

        content.append({"type": "text", "text": base_prompt})

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4000,
            messages=[{"role": "user", "content": content}]
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        analysis = json.loads(raw)
        analysis['cached_at'] = datetime.now().strftime('%Y-%m-%d %H:%M')
        analysis['cache_key'] = cache_key

        # Save to cache + output
        save_cache(cache_key, analysis)
        with open(os.path.join(OUTPUT_DIR, 'analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2)

        return jsonify({'analysis': analysis, 'from_cache': False})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/load-cache/<cache_key>', methods=['GET'])
def load_cache_route(cache_key):
    cached = load_cache(cache_key)
    if cached:
        return jsonify({'analysis': cached})
    return jsonify({'error': 'Not found'}), 404


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: Generate PDF
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    data = request.json
    analysis = data.get('analysis')
    if not analysis:
        analysis_path = os.path.join(OUTPUT_DIR, 'analysis.json')
        if os.path.exists(analysis_path):
            with open(analysis_path) as f:
                analysis = json.load(f)
        else:
            return jsonify({'error': 'No analysis data'}), 400

    output_path = os.path.join(OUTPUT_DIR, 'BADMINTON-ANALYSIS-REPORT.pdf')
    styles = make_styles()

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=20 * mm,
        title=f"{analysis.get('sport','Badminton')} Analysis Report",
        author="AI Coaching Analysis",
    )

    story = []
    story += pdf_cover(analysis, styles)
    story += pdf_scorecard(analysis, styles)
    story += pdf_frame_analysis(analysis, styles)
    story += pdf_drills(analysis, styles)
    story += pdf_final(analysis, styles)

    doc.build(story, onFirstPage=pdf_footer, onLaterPages=pdf_footer)
    return send_file(output_path, as_attachment=True, download_name='Badminton-Analysis-Report.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# PDF Sections
# ─────────────────────────────────────────────────────────────────────────────
def pdf_cover(analysis, styles):
    story = []
    banner = Drawing(W - 2 * MARGIN, 100)
    bg = Rect(0, 0, W - 2 * MARGIN, 100, fillColor=C_NAVY, strokeColor=None)
    banner.add(bg)
    accent = Rect(0, 0, 6, 100, fillColor=C_COURT, strokeColor=None)
    banner.add(accent)
    story.append(banner)
    story.append(Spacer(1, -100))
    story.append(Spacer(1, 12))

    sport = analysis.get('sport', 'Badminton').upper()
    story.append(Paragraph(f"{sport} PLAYER", styles["title"]))
    story.append(Paragraph("TECHNICAL ANALYSIS REPORT", styles["title"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Professional AI Coaching Assessment — Video Frame Analysis", styles["subtitle"]))
    story.append(Paragraph(f"Player: {analysis.get('player_name','Player')}  |  Date: {datetime.now().strftime('%B %d, %Y')}", styles["meta"]))
    story.append(Spacer(1, 60))

    score = analysis.get('overall_score', 5)
    grade = analysis.get('overall_grade', 'C')
    sc = score_color(score)

    info_data = [
        [Paragraph("<b>PLAYER</b>", styles["th"]),
         Paragraph("<b>SPORT</b>", styles["th"]),
         Paragraph("<b>OVERALL SCORE</b>", styles["th"]),
         Paragraph("<b>GRADE</b>", styles["th"])],
        [Paragraph(analysis.get('player_name', 'Player'), styles["td"]),
         Paragraph(analysis.get('sport', 'Badminton'), styles["td"]),
         Paragraph(f'<b><font color="#{to_hex(sc)}">{score}/10</font></b>', styles["td_bold"]),
         Paragraph(f'<b><font color="#{to_hex(sc)}">{grade}</font></b>', styles["td_bold"])],
    ]
    cw = (W - 2 * MARGIN) / 4
    t = Table(info_data, colWidths=[cw] * 4)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_BLUE),
        ("BACKGROUND", (0, 1), (-1, 1), C_LGRAY),
        ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    summary = Table([[Paragraph(
        f"<b>Coach's Summary:</b> {analysis.get('executive_summary', '')}",
        styles["body"]
    )]], colWidths=[W - 2 * MARGIN])
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E3F2FD")),
        ("BOX", (0, 0), (-1, -1), 1.5, C_BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(summary)
    story.append(Spacer(1, 12))

    # Strengths
    strengths = analysis.get('strengths', [])
    if strengths:
        story.append(Paragraph("<b>Key Strengths:</b>", styles["body"]))
        for s in strengths:
            story.append(Paragraph(f"✓ {s}", styles["bullet"]))

    story.append(PageBreak())
    return story


def pdf_scorecard(analysis, styles):
    story = []
    story.append(Paragraph("SKILLS SCORECARD", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY))
    story.append(Spacer(1, 8))

    skills = analysis.get('skills', [])
    if not skills:
        story.append(Paragraph("No skills data.", styles["body"]))
        story.append(PageBreak())
        return story

    header = [
        Paragraph("<b>Skill Area</b>", styles["th"]),
        Paragraph("<b>Score</b>", styles["th"]),
        Paragraph("<b>Level</b>", styles["th"]),
        Paragraph("<b>Observation</b>", styles["th"]),
        Paragraph("<b>How to Improve</b>", styles["th"]),
    ]
    rows = [header]
    for sk in skills:
        s = sk.get('score', 5)
        col = score_color(s)
        rows.append([
            Paragraph(sk.get('name', ''), styles["td"]),
            Paragraph(f'<b><font color="#{to_hex(col)}">{s}/10</font></b>', styles["td_bold"]),
            Paragraph(f'<font color="#{to_hex(col)}">{score_label(s)}</font>', styles["td"]),
            Paragraph(sk.get('observation', ''), styles["td"]),
            Paragraph(sk.get('improvement', ''), styles["td"]),
        ])

    cw_total = W - 2 * MARGIN
    col_w = [100, 40, 60, cw_total - 100 - 40 - 60 - 100, 100]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)

    # Top fixes
    top_fixes = analysis.get('top_fixes', [])
    if top_fixes:
        story.append(Spacer(1, 12))
        story.append(Paragraph("TOP PRIORITY FIXES", styles["section"]))
        story.append(HRFlowable(width="100%", thickness=1, color=C_RED))
        story.append(Spacer(1, 6))
        for i, fix in enumerate(top_fixes, 1):
            fix_row = Table([[
                Paragraph(f"<b>{i}</b>", ParagraphStyle("fn", fontName="Helvetica-Bold",
                    fontSize=13, textColor=C_WHITE, leading=16, alignment=TA_CENTER)),
                Paragraph(fix, styles["body"]),
            ]], colWidths=[30, W - 2 * MARGIN - 30])
            fix_row.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), C_RED),
                ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#FFF3E0")),
                ("BOX", (0, 0), (-1, -1), 0.5, C_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(fix_row)
            story.append(Spacer(1, 4))

    story.append(PageBreak())
    return story


def pdf_frame_analysis(analysis, styles):
    story = []
    story.append(Paragraph("FRAME-BY-FRAME TECHNICAL ANALYSIS", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY))
    story.append(Spacer(1, 8))

    pri_colors = {"CRITICAL": C_RED, "HIGH": C_AMBER, "MEDIUM": C_BLUE, "LOW": C_GREEN}
    frames = analysis.get('frame_analysis', [])

    for frame in frames:
        pri = frame.get('priority', 'MEDIUM')
        pc = pri_colors.get(pri, C_BLUE)

        header = Table([[
            Paragraph(f'<b>{frame.get("timestamp","")}: {frame.get("title","")}</b>',
                ParagraphStyle("fh", fontName="Helvetica-Bold", fontSize=9.5, textColor=C_WHITE, leading=12)),
            Paragraph(f'<b>{pri}</b>',
                ParagraphStyle("fp", fontName="Helvetica-Bold", fontSize=8.5, textColor=C_WHITE, leading=12)),
        ]], colWidths=[W - 2 * MARGIN - 65, 65])
        header.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), C_NAVY),
            ("BACKGROUND", (1, 0), (1, 0), pc),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("ALIGN", (1, 0), (1, 0), "CENTER"),
        ]))
        story.append(KeepTogether([header, Spacer(1, 3)]))

        # Frame image if exists
        fname = f"frame_{frame.get('timestamp','0').replace('s','').zfill(4)}s.jpg"
        fpath = os.path.join(FRAMES_DIR, fname)
        if os.path.exists(fpath):
            img = RLImage(fpath, width=100, height=70)
            obs_text = "\n".join([f"• {o}" for o in frame.get('observations', [])])
            img_row = Table([[
                img,
                Table([[Paragraph(f"• {o}", styles["bullet"])] for o in frame.get('observations', [])],
                    colWidths=[W - 2 * MARGIN - 115]),
            ]], colWidths=[110, W - 2 * MARGIN - 110])
            img_row.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(img_row)
        else:
            for obs in frame.get('observations', []):
                story.append(Paragraph(f"• {obs}", styles["bullet"]))

        pro_ref = frame.get('pro_reference', '')
        if pro_ref:
            ref = Table([[Paragraph(f'<b>Pro Reference:</b> {pro_ref}', styles["body_sm"])]],
                colWidths=[W - 2 * MARGIN])
            ref.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF8E1")),
                ("BOX", (0, 0), (-1, -1), 1, C_AMBER),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(ref)
        story.append(Spacer(1, 8))

    story.append(PageBreak())
    return story


def pdf_drills(analysis, styles):
    story = []
    story.append(Paragraph("PERSONALISED TRAINING DRILL PLAN", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY))
    story.append(Spacer(1, 8))

    drills = analysis.get('drills', [])
    if not drills:
        story.append(Paragraph("No drills data.", styles["body"]))
        story.append(PageBreak())
        return story

    header = [
        Paragraph("<b>Drill</b>", styles["th"]),
        Paragraph("<b>Duration</b>", styles["th"]),
        Paragraph("<b>How to Execute</b>", styles["th"]),
        Paragraph("<b>Targets</b>", styles["th"]),
    ]
    rows = [header]
    for d in drills:
        rows.append([
            Paragraph(f"<b>{d.get('name','')}</b>", styles["td_bold"]),
            Paragraph(d.get('duration', ''), styles["td"]),
            Paragraph(d.get('description', ''), styles["td"]),
            Paragraph(d.get('targets', ''), styles["td"]),
        ])

    cw_total = W - 2 * MARGIN
    col_w = [110, 60, cw_total - 110 - 60 - 90, 90]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(PageBreak())
    return story


def pdf_final(analysis, styles):
    story = []
    story.append(Paragraph("COACH'S FINAL ASSESSMENT", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY))
    story.append(Spacer(1, 10))

    quote = analysis.get('coach_quote', '')
    if quote:
        q = Table([[Paragraph(f'"{quote}"', styles["quote"])]],
            colWidths=[W - 2 * MARGIN])
        q.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E8F5E9")),
            ("BOX", (0, 0), (-1, -1), 2, C_COURT),
            ("LEFTPADDING", (0, 0), (-1, -1), 16),
            ("RIGHTPADDING", (0, 0), (-1, -1), 16),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ]))
        story.append(q)
        story.append(Spacer(1, 16))

    sig = Table([[
        Paragraph(
            f"<b>Player:</b> {analysis.get('player_name','Player')}<br/>"
            f"<b>Sport:</b> {analysis.get('sport','Badminton')}<br/>"
            f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y')}<br/>"
            f"<b>Overall:</b> {analysis.get('overall_score',0)}/10 — Grade {analysis.get('overall_grade','C')}",
            styles["body_sm"]
        ),
        Paragraph(
            "<b>Analysis by:</b> Claude AI Coaching System<br/>"
            "<b>Method:</b> Video Frame Analysis<br/>"
            "<b>Model:</b> Claude Opus 4.6<br/>"
            "<b>Report Version:</b> 1.0",
            styles["body_sm"]
        ),
    ]], colWidths=[(W - 2 * MARGIN) * 0.5, (W - 2 * MARGIN) * 0.5])
    sig.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, -1), C_WHITE),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LINEBEFORE", (1, 0), (1, 0), 1, C_COURT),
    ]))
    story.append(sig)
    return story


def pdf_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(C_MGRAY)
    canvas.drawRightString(W - MARGIN, 10 * mm, f"Page {doc.page}")
    canvas.drawString(MARGIN, 10 * mm, "Sports Technical Analysis Report — AI Coaching System")
    canvas.setStrokeColor(C_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 13 * mm, W - MARGIN, 13 * mm)
    canvas.restoreState()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)
