import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import streamlit.components.v1 as components
import time
import re

st.set_page_config(page_title="CineScope · Sentiment AI", page_icon="🎬", layout="centered")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400;1,700&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --gold:#C9A84C; --gold-dim:rgba(201,168,76,0.18);
    --cream:#F5ECD7; --dark:#0D0C0A; --panel:#151310; --muted:#6B6560;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0D0C0A !important;
    color: #F5ECD7 !important;
    font-family: 'DM Sans', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container { max-width: 760px !important; padding: 0 2rem 4rem !important; position: relative; z-index: 1; }

/* ── Textarea ── */
[data-testid="stTextArea"] textarea {
    background: #0A0908 !important; border: 1px solid rgba(201,168,76,0.2) !important;
    border-radius: 3px !important; color: #F5ECD7 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.92rem !important;
    line-height: 1.7 !important; caret-color: #C9A84C !important;
    padding: 1rem 1.1rem !important; resize: vertical !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #C9A84C !important; box-shadow: 0 0 0 3px rgba(201,168,76,0.08) !important; outline: none !important;
}
[data-testid="stTextArea"] label { display: none !important; }

/* ── Button ── */
[data-testid="stButton"] > button {
    width: 100%; background: transparent !important; border: 1px solid #C9A84C !important;
    border-radius: 3px !important; color: #C9A84C !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.72rem !important;
    letter-spacing: 0.2em !important; text-transform: uppercase !important;
    padding: 0.85rem 2rem !important; cursor: pointer !important;
    transition: background 0.22s, color 0.22s !important; margin-top: 0.6rem;
}
[data-testid="stButton"] > button:hover { background: #C9A84C !important; color: #0D0C0A !important; }

/* ── Input card ── */
.input-card {
    background: #151310; border: 1px solid rgba(201,168,76,0.18);
    border-radius: 4px; padding: 2rem 2rem 1.6rem; margin-bottom: 1.4rem;
    box-shadow: 0 4px 40px rgba(0,0,0,0.55), inset 0 1px 0 rgba(201,168,76,0.06);
}
.card-label {
    font-family: 'DM Mono', monospace; font-size: 0.62rem; letter-spacing: 0.22em;
    text-transform: uppercase; color: #C9A84C; margin-bottom: 0.85rem;
}

/* ── Footer ── */
.footer { text-align: center; margin-top: 3.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(201,168,76,0.12); }
.footer-text { font-family: 'DM Mono', monospace; font-size: 0.58rem; letter-spacing: 0.18em; text-transform: uppercase; color: #6B6560; opacity: 0.45; }
</style>
""", unsafe_allow_html=True)


# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    word_idx = imdb.get_word_index()
    rev_idx  = {v: k for k, v in word_idx.items()}
    mdl      = load_model('simple_rnn_imdb.h5')
    return word_idx, rev_idx, mdl

word_index, reverse_word_index, model = load_resources()


# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess_text(text: str):
    words   = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]
    return sequence.pad_sequences([encoded], maxlen=500)

def score_to_stars(score): return round(1 + score * 9, 1)

def confidence_info(score):
    d = abs(score - 0.5)
    if d > 0.30: return "HIGH",   "#4CAF84", "rgba(76,175,132,0.12)", "rgba(76,175,132,0.3)"
    if d > 0.15: return "MEDIUM", "#C9A84C", "rgba(201,168,76,0.12)", "rgba(201,168,76,0.3)"
    return              "LOW",    "#C95A4C", "rgba(201,90,76,0.12)",  "rgba(201,90,76,0.3)"

def tone_breakdown(score):
    if   score >= 0.90: return [("Euphoric",88),("Enthusiastic",75),("Uplifting",60)]
    elif score >= 0.75: return [("Positive",82),("Appreciative",68),("Warm",55)]
    elif score >= 0.60: return [("Favourable",70),("Mild Praise",55),("Neutral",30)]
    elif score >= 0.50: return [("Mixed",60),("Hesitant",50),("Cautious",35)]
    elif score >= 0.35: return [("Disappointed",72),("Underwhelmed",60),("Flat",40)]
    elif score >= 0.20: return [("Critical",80),("Harsh",65),("Dismissive",50)]
    else:               return [("Scathing",90),("Contemptuous",75),("Hostile",62)]

def star_label(stars):
    if   stars >= 9.0: return "Masterpiece"
    elif stars >= 8.0: return "Excellent"
    elif stars >= 7.0: return "Very Good"
    elif stars >= 6.0: return "Good"
    elif stars >= 5.0: return "Average"
    elif stars >= 4.0: return "Below Average"
    elif stars >= 3.0: return "Poor"
    elif stars >= 2.0: return "Bad"
    else:              return "Terrible"

def text_stats(text):
    words    = text.strip().split()
    sents    = max(1, len(re.split(r'[.!?]+', text.strip())))
    avg_wps  = round(len(words) / sents, 1)
    unique   = len(set(w.lower().strip(".,!?;:") for w in words))
    lexical  = round(unique / max(1, len(words)) * 100, 1)
    return len(words), sents, avg_wps, lexical

def result_card_html(score, text):
    positive   = score > 0.5
    sentiment  = "Positive" if positive else "Negative"
    stars      = score_to_stars(score)
    label      = star_label(stars)
    vc         = "#4CAF84" if positive else "#C95A4C"
    bar_color  = vc
    bg_grad    = "linear-gradient(135deg,#0C1A12,#0D0C0A)" if positive else "linear-gradient(135deg,#1A0C0C,#0D0C0A)"
    border_c   = "rgba(76,175,132,0.28)" if positive else "rgba(201,90,76,0.28)"
    bar_pct    = round(score*100,1) if positive else round((1-score)*100,1)
    ct, cc, cb, cbd = confidence_info(score)
    tones      = tone_breakdown(score)
    wc, sents, avg_wps, lexical = text_stats(text)
    full       = int(stars); partial = stars - full
    star_html  = ""
    for i in range(1, 11):
        if i <= full:
            star_html += f'<span class="star filled" style="animation-delay:{(i-1)*0.07}s">★</span>'
        elif i == full+1 and partial >= 0.3:
            star_html += f'<span class="star half" style="animation-delay:{(i-1)*0.07}s">★</span>'
        else:
            star_html += '<span class="star empty">★</span>'
    tone_html = ""
    for tn, tp in tones:
        tone_html += f"""<div class="tone-row"><span class="tone-name">{tn}</span><div class="tone-bar-bg"><div class="tone-bar-fill" style="width:0%;background:{vc}80;" data-target="{tp}"></div></div><span class="tone-pct">{tp}%</span></div>"""

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:transparent;font-family:'DM Sans',sans-serif;padding:0 0 8px 0;}}
.card{{background:{bg_grad};border:1px solid {border_c};border-radius:4px;padding:1.8rem 2rem 1.6rem;position:relative;overflow:hidden;animation:fadeUp 0.5s cubic-bezier(.22,.68,0,1.2) both;margin-bottom:10px;}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:translateY(0)}}}}
.top-row{{display:flex;justify-content:space-between;align-items:flex-start;}}
.eyebrow{{font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.24em;text-transform:uppercase;color:#6B6560;margin-bottom:0.35rem;}}
.verdict{{font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:700;color:{vc};line-height:1;margin-bottom:0.6rem;}}
.chips{{display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1.2rem;}}
.chip{{font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.1em;padding:0.25rem 0.65rem;border-radius:2px;text-transform:uppercase;}}
.chip-conf{{background:{cb};color:{cc};border:1px solid {cbd};}}
.chip-words{{background:rgba(255,255,255,0.04);color:#6B6560;border:1px solid rgba(255,255,255,0.08);}}
.star-eyebrow{{font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.2em;text-transform:uppercase;color:#6B6560;margin-bottom:0.3rem;}}
.star-number{{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:{vc};line-height:1;}}
.star-label{{font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.12em;text-transform:uppercase;color:{vc};opacity:0.7;margin-top:0.2rem;}}
.stars{{display:flex;gap:2px;margin-top:0.5rem;justify-content:flex-end;}}
.star{{font-size:1.1rem;}}
.star.filled{{color:{vc};animation:starPop 0.4s cubic-bezier(.22,.68,0,1.2) both;}}
.star.half{{color:{vc};opacity:0.5;animation:starPop 0.4s cubic-bezier(.22,.68,0,1.2) both;}}
.star.empty{{color:#2A2520;}}
@keyframes starPop{{from{{transform:scale(0.4);opacity:0}}to{{transform:scale(1);opacity:1}}}}
.divider{{height:1px;background:rgba(255,255,255,0.06);margin:1.2rem 0;}}
.bar-label{{font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.2em;text-transform:uppercase;color:#6B6560;margin-bottom:0.5rem;}}
.bar-bg{{width:100%;height:4px;background:rgba(255,255,255,0.07);border-radius:2px;overflow:hidden;}}
.bar-fill{{height:100%;border-radius:2px;background:{bar_color};animation:growBar 1s cubic-bezier(.22,.68,0,1.2) 0.25s both;width:0;}}
@keyframes growBar{{from{{width:0}}to{{width:{bar_pct}%}}}}
.score-row{{font-family:'DM Mono',monospace;font-size:0.68rem;color:#6B6560;margin-top:0.5rem;display:flex;justify-content:space-between;}}
.score-pct{{color:{vc};font-weight:500;}}
.panel{{background:#111009;border:1px solid rgba(201,168,76,0.1);border-radius:4px;padding:1.3rem 1.5rem;margin-bottom:10px;animation:fadeUp 0.5s cubic-bezier(.22,.68,0,1.2) 0.15s both;opacity:0;}}
.panel-title{{font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.22em;text-transform:uppercase;color:#C9A84C;margin-bottom:1rem;}}
.tone-row{{display:flex;align-items:center;gap:0.7rem;margin-bottom:0.65rem;}}
.tone-name{{font-family:'DM Mono',monospace;font-size:0.65rem;color:#F5ECD7;width:110px;flex-shrink:0;letter-spacing:0.06em;}}
.tone-bar-bg{{flex:1;height:3px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;}}
.tone-bar-fill{{height:100%;border-radius:2px;transition:width 0.9s cubic-bezier(.22,.68,0,1.2);}}
.tone-pct{{font-family:'DM Mono',monospace;font-size:0.62rem;color:#6B6560;width:32px;text-align:right;}}
.stats-grid{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;}}
.stat-box{{background:#0D0C0A;border:1px solid rgba(201,168,76,0.1);border-radius:3px;padding:0.8rem 0.7rem;text-align:center;}}
.stat-val{{font-family:'Playfair Display',serif;font-size:1.3rem;font-weight:700;color:#F5ECD7;line-height:1;}}
.stat-key{{font-family:'DM Mono',monospace;font-size:0.52rem;letter-spacing:0.14em;text-transform:uppercase;color:#6B6560;margin-top:0.3rem;}}
</style></head><body>
<div class="card">
  <div class="top-row">
    <div class="left">
      <div class="eyebrow">Sentiment Verdict</div>
      <div class="verdict">{sentiment}</div>
      <div class="chips">
        <span class="chip chip-conf">Confidence: {ct}</span>
        <span class="chip chip-words">{wc} words</span>
      </div>
    </div>
    <div class="right">
      <div class="star-eyebrow">Star Rating</div>
      <div class="star-number">{stars}<span style="font-size:1rem;opacity:0.5;font-family:'DM Mono',monospace"> /10</span></div>
      <div class="star-label">{label}</div>
      <div class="stars">{star_html}</div>
    </div>
  </div>
  <div class="divider"></div>
  <div class="bar-label">Sentiment Strength</div>
  <div class="bar-bg"><div class="bar-fill"></div></div>
  <div class="score-row">
    <span><span class="score-pct">{bar_pct}%</span> {sentiment.lower()} signal</span>
    <span>raw score {score:.4f}</span>
  </div>
</div>
<div class="panel" id="tonePanel">
  <div class="panel-title">◈ Tonal Breakdown</div>{tone_html}
</div>
<div class="panel" style="animation-delay:0.25s;">
  <div class="panel-title">◈ Review Analytics</div>
  <div class="stats-grid">
    <div class="stat-box"><div class="stat-val">{wc}</div><div class="stat-key">Words</div></div>
    <div class="stat-box"><div class="stat-val">{sents}</div><div class="stat-key">Sentences</div></div>
    <div class="stat-box"><div class="stat-val">{avg_wps}</div><div class="stat-key">Words/Sent</div></div>
    <div class="stat-box"><div class="stat-val">{lexical}%</div><div class="stat-key">Lexical Rich</div></div>
  </div>
</div>
<script>window.addEventListener('load',function(){{setTimeout(function(){{document.querySelectorAll('.tone-bar-fill').forEach(function(el){{el.style.width=el.dataset.target+'%';}});}},400);}});</script>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# HERO — rendered via components.html to bypass Streamlit sanitizer
# ══════════════════════════════════════════════════════════════════════════════
HERO_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400;1,700&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:transparent;font-family:'DM Sans',sans-serif;overflow:hidden;}

/* ── Film strip top ── */
.filmstrip{width:100%;height:36px;background:#0A0908;border-bottom:1px solid rgba(201,168,76,0.12);display:flex;align-items:center;overflow:hidden;position:relative;}
.perfs{display:flex;gap:0px;align-items:center;height:100%;padding:0 8px;}
.perf{width:18px;height:24px;border-radius:3px;border:1.5px solid rgba(201,168,76,0.22);margin-right:10px;flex-shrink:0;}
.strip-text{position:absolute;left:50%;transform:translateX(-50%);font-family:'DM Mono',monospace;font-size:0.52rem;letter-spacing:0.3em;text-transform:uppercase;color:rgba(201,168,76,0.35);white-space:nowrap;}

/* ── Hero wrapper ── */
.hero{
  padding: 3rem 2rem 2.2rem;
  position:relative;
  text-align:center;
  background: #0D0C0A;
}

/* ── Spotlight cone ── */
.spotlight{
  position:absolute;top:0;left:50%;transform:translateX(-50%);
  width:420px;height:260px;
  background:radial-gradient(ellipse 200px 180px at 50% 0%, rgba(201,168,76,0.07) 0%, transparent 70%);
  pointer-events:none;
}

/* ── Floating review cards ── */
.float-card{
  position:absolute;
  background:#151310;
  border:1px solid rgba(201,168,76,0.14);
  border-radius:4px;
  padding:0.7rem 0.9rem;
  font-family:'DM Sans',sans-serif;
  font-size:0.72rem;
  color:rgba(245,236,215,0.55);
  line-height:1.5;
  max-width:170px;
  animation:floatIn 1s cubic-bezier(.22,.68,0,1.2) both;
}
.float-card .stars{color:#C9A84C;font-size:0.65rem;margin-bottom:0.3rem;letter-spacing:2px;}
.float-card .quote{font-style:italic;color:rgba(245,236,215,0.4);}

.fc-left{left:0;top:56px;animation-delay:0.6s;transform:rotate(-3deg);}
.fc-right{right:0;top:56px;animation-delay:0.8s;transform:rotate(2.5deg);}
.fc-left2{left:12px;bottom:20px;animation-delay:1s;transform:rotate(1.5deg);}
.fc-right2{right:12px;bottom:20px;animation-delay:1.1s;transform:rotate(-2deg);}

@keyframes floatIn{from{opacity:0;transform:translateY(12px) rotate(var(--r,0deg))}to{opacity:1;}}

/* ── Badge row ── */
.badge-row{display:flex;justify-content:center;gap:0.6rem;margin-bottom:1.4rem;flex-wrap:wrap;}
.badge{
  font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.18em;
  text-transform:uppercase;padding:0.3rem 0.75rem;border-radius:2px;
}
.badge-gold{background:rgba(201,168,76,0.1);color:#C9A84C;border:1px solid rgba(201,168,76,0.25);}
.badge-dim{background:rgba(255,255,255,0.03);color:#6B6560;border:1px solid rgba(255,255,255,0.07);}

/* ── Title ── */
.eyebrow{
  font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.32em;
  text-transform:uppercase;color:rgba(201,168,76,0.6);margin-bottom:1rem;
  animation:fadeUp 0.7s ease both;
}
.title-wrap{margin-bottom:0.8rem;animation:fadeUp 0.7s ease 0.1s both;opacity:0;}
.title-main{
  font-family:'Playfair Display',serif;
  font-size:clamp(3rem,8vw,4.8rem);
  font-weight:900;
  line-height:0.95;
  color:#F5ECD7;
  display:block;
  letter-spacing:-0.01em;
}
.title-em{
  font-family:'Playfair Display',serif;
  font-style:italic;
  font-size:clamp(3rem,8vw,4.8rem);
  font-weight:700;
  color:#C9A84C;
  display:block;
  margin-top:-0.05em;
}

/* ── Horizontal rule ── */
.rule-wrap{display:flex;align-items:center;justify-content:center;gap:1rem;margin:1.2rem 0 1rem;animation:fadeUp 0.7s ease 0.2s both;opacity:0;}
.rule-line{flex:1;max-width:80px;height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.3));}
.rule-line.right{background:linear-gradient(90deg,rgba(201,168,76,0.3),transparent);}
.rule-dot{width:4px;height:4px;border-radius:50%;background:#C9A84C;opacity:0.6;}

/* ── Subtitle ── */
.subtitle{
  font-size:0.9rem;color:#6B6560;line-height:1.75;letter-spacing:0.02em;
  max-width:440px;margin:0 auto 1.8rem;
  animation:fadeUp 0.7s ease 0.25s both;opacity:0;
}

/* ── Stat pills ── */
.stat-row{display:flex;justify-content:center;gap:2rem;margin-bottom:0;animation:fadeUp 0.7s ease 0.35s both;opacity:0;}
.stat-pill{text-align:center;}
.stat-pill .num{font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:700;color:#F5ECD7;line-height:1;}
.stat-pill .lbl{font-family:'DM Mono',monospace;font-size:0.55rem;letter-spacing:0.18em;text-transform:uppercase;color:#6B6560;margin-top:0.2rem;}
.stat-sep{width:1px;background:rgba(201,168,76,0.15);align-self:stretch;margin:0.2rem 0;}

/* ── Film strip bottom ── */
.filmstrip-bottom{width:100%;height:36px;background:#0A0908;border-top:1px solid rgba(201,168,76,0.12);display:flex;align-items:center;overflow:hidden;margin-top:2.2rem;}

/* ── Scroll hint ── */
.scroll-hint{display:flex;flex-direction:column;align-items:center;gap:0.4rem;margin-top:1.4rem;opacity:0;animation:fadeUp 0.7s ease 0.55s both;}
.scroll-hint span{font-family:'DM Mono',monospace;font-size:0.55rem;letter-spacing:0.2em;text-transform:uppercase;color:#6B6560;}
.scroll-arrow{width:1px;height:24px;background:linear-gradient(180deg,rgba(201,168,76,0.4),transparent);animation:scrollPulse 1.8s ease-in-out infinite;}
@keyframes scrollPulse{0%,100%{opacity:0.3;transform:scaleY(0.8)}50%{opacity:1;transform:scaleY(1)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

/* ── Horizontal ticker ── */
.ticker-wrap{width:100%;overflow:hidden;height:28px;display:flex;align-items:center;background:#0A0908;}
.ticker{display:flex;gap:0;white-space:nowrap;animation:ticker 22s linear infinite;}
@keyframes ticker{from{transform:translateX(0)}to{transform:translateX(-50%)}}
.tick-item{font-family:'DM Mono',monospace;font-size:0.55rem;letter-spacing:0.22em;text-transform:uppercase;color:rgba(201,168,76,0.28);padding:0 2rem;}
.tick-sep{color:rgba(201,168,76,0.15);}
</style>
</head><body>

<!-- Film strip top -->
<div class="filmstrip">
  <div class="perfs">
    <div class="perf"></div><div class="perf"></div><div class="perf"></div>
    <div class="perf"></div><div class="perf"></div><div class="perf"></div>
    <div class="perf"></div><div class="perf"></div><div class="perf"></div>
    <div class="perf"></div><div class="perf"></div><div class="perf"></div>
    <div class="perf"></div><div class="perf"></div><div class="perf"></div>
    <div class="perf"></div><div class="perf"></div><div class="perf"></div>
    <div class="perf"></div><div class="perf"></div><div class="perf"></div>
    <div class="perf"></div>
  </div>
  <div class="strip-text">35mm · Color · Dolby Digital · 2025</div>
  <div class="perfs" style="margin-left:auto">
    <div class="perf"></div><div class="perf"></div><div class="perf"></div>
    <div class="perf"></div><div class="perf"></div><div class="perf"></div>
    <div class="perf"></div><div class="perf"></div><div class="perf"></div>
    <div class="perf"></div><div class="perf"></div>
  </div>
</div>

<!-- Hero body -->
<div class="hero">
  <div class="spotlight"></div>

  <!-- Floating ghost review cards -->
  <div class="float-card fc-left">
    <div class="stars">★★★★★</div>
    <div class="quote">"A masterwork of modern cinema…"</div>
  </div>
  <div class="float-card fc-right">
    <div class="stars">★★★★☆</div>
    <div class="quote">"Visually stunning, emotionally raw."</div>
  </div>
  <div class="float-card fc-left2" style="opacity:0.5">
    <div class="stars">★★☆☆☆</div>
    <div class="quote">"Predictable and disappointing."</div>
  </div>
  <div class="float-card fc-right2" style="opacity:0.5">
    <div class="stars">★★★★★</div>
    <div class="quote">"Absolutely breathtaking."</div>
  </div>

  <div class="eyebrow">✦ Neural Sentiment Analysis ✦</div>

  <div class="title-wrap">
    <span class="title-main">Cine</span>
    <span class="title-em">Scope</span>
  </div>

  <div class="rule-wrap">
    <div class="rule-line"></div>
    <div class="rule-dot"></div>
    <div class="rule-dot" style="opacity:0.35;"></div>
    <div class="rule-dot" style="opacity:0.18;"></div>
    <div class="rule-line right"></div>
  </div>

  <p class="subtitle">
    A recurrent neural network trained on 50,000 IMDB reviews<br>
    decodes the emotional undertone of any film critique.
  </p>

  <div class="badge-row">
    <span class="badge badge-gold">RNN Model</span>
    <span class="badge badge-dim">50k Reviews</span>
    <span class="badge badge-gold">Star Rating</span>
    <span class="badge badge-dim">Tone Analysis</span>
  </div>

  <div class="stat-row">
    <div class="stat-pill">
      <div class="num">50k</div>
      <div class="lbl">Training Reviews</div>
    </div>
    <div class="stat-sep"></div>
    <div class="stat-pill">
      <div class="num">500</div>
      <div class="lbl">Token Window</div>
    </div>
    <div class="stat-sep"></div>
    <div class="stat-pill">
      <div class="num">1–10</div>
      <div class="lbl">Star Scale</div>
    </div>
    <div class="stat-sep"></div>
    <div class="stat-pill">
      <div class="num">3</div>
      <div class="lbl">Tone Signals</div>
    </div>
  </div>

  <div class="scroll-hint">
    <span>Enter your review below</span>
    <div class="scroll-arrow"></div>
  </div>
</div>

<!-- Film strip bottom / ticker -->
<div class="ticker-wrap">
  <div class="ticker">
    <span class="tick-item">Sentiment Analysis</span><span class="tick-sep">◆</span>
    <span class="tick-item">Star Rating Prediction</span><span class="tick-sep">◆</span>
    <span class="tick-item">Tonal Breakdown</span><span class="tick-sep">◆</span>
    <span class="tick-item">Lexical Analysis</span><span class="tick-sep">◆</span>
    <span class="tick-item">Neural RNN Model</span><span class="tick-sep">◆</span>
    <span class="tick-item">IMDB Dataset</span><span class="tick-sep">◆</span>
    <span class="tick-item">Sentiment Analysis</span><span class="tick-sep">◆</span>
    <span class="tick-item">Star Rating Prediction</span><span class="tick-sep">◆</span>
    <span class="tick-item">Tonal Breakdown</span><span class="tick-sep">◆</span>
    <span class="tick-item">Lexical Analysis</span><span class="tick-sep">◆</span>
    <span class="tick-item">Neural RNN Model</span><span class="tick-sep">◆</span>
    <span class="tick-item">IMDB Dataset</span><span class="tick-sep">◆</span>
  </div>
</div>

</body></html>"""

components.html(HERO_HTML, height=560)

# ── Input card ────────────────────────────────────────────────────────────────
st.markdown('<div class="input-card"><div class="card-label">◈ Your Review</div>', unsafe_allow_html=True)
user_input = st.text_area("", placeholder="Paste or type a movie review here…", height=160, key="review_input")
classify   = st.button("Analyse Sentiment →")
st.markdown('</div>', unsafe_allow_html=True)

# ── Result ────────────────────────────────────────────────────────────────────
if classify:
    if not user_input.strip():
        st.markdown("""
        <div style="text-align:center;padding:1.2rem;font-family:'DM Mono',monospace;
             font-size:0.72rem;letter-spacing:0.14em;color:#C9A84C;
             border:1px solid rgba(201,168,76,0.18);border-radius:3px;margin-top:1rem;
             text-transform:uppercase;">⚠ Please enter a review before analysing.</div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Reading between the lines…"):
            time.sleep(0.3)
            preprocessed = preprocess_text(user_input)
            prediction   = model.predict(preprocessed, verbose=0)
            score        = float(prediction[0][0])
        components.html(result_card_html(score, user_input), height=590)
else:
    # How it works
    st.markdown("""
    <div class="input-card" style="margin-top:0;">
      <div class="card-label">◈ How It Works</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-top:0.5rem;">
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(201,168,76,0.14);border-radius:3px;padding:0.75rem 1rem;font-size:0.78rem;color:#6B6560;line-height:1.5;">
          <strong style="display:block;color:#C9A84C;font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.25rem;">Model</strong>
          Simple RNN trained on 50k IMDB reviews, 500-token context window.
        </div>
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(201,168,76,0.14);border-radius:3px;padding:0.75rem 1rem;font-size:0.78rem;color:#6B6560;line-height:1.5;">
          <strong style="display:block;color:#C9A84C;font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.25rem;">Star Rating</strong>
          Sentiment score (0–1) mapped to a 1–10 star scale in real time.
        </div>
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(201,168,76,0.14);border-radius:3px;padding:0.75rem 1rem;font-size:0.78rem;color:#6B6560;line-height:1.5;">
          <strong style="display:block;color:#C9A84C;font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.25rem;">Tonal Breakdown</strong>
          Descriptive tone tags calibrated to score ranges.
        </div>
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(201,168,76,0.14);border-radius:3px;padding:0.75rem 1rem;font-size:0.78rem;color:#6B6560;line-height:1.5;">
          <strong style="display:block;color:#C9A84C;font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.25rem;">Limitation</strong>
          Sarcasm and complex irony may fool the sentiment classifier.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div class="footer-text">CineScope · Powered by TensorFlow & Streamlit · IMDB Dataset</div>
</div>
""", unsafe_allow_html=True)