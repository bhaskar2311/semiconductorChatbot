"""
PE Knowledge Graph Chatbot — Streamlit Web App
===============================================
Upload your files in the sidebar, enter your API key, and start asking
questions about semiconductor plasma etching physics.

Files to upload:
  REQUIRED  merged_triplets.json        (Phase 6 output)
  optional  equations_database.json     (Phase 2b output)
  optional  pe.json                     (opb/pe/pe.json)
  optional  OPB.json                    (opb/physics/OPB.json)
  optional  pe_interactive.html         (Phase 7 PyVis graph)

Run:
  pip install streamlit openai pandas networkx python-dotenv
  streamlit run pe_streamlit_app.py
"""

import json
import re
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PE Knowledge Graph Chatbot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
LLM_MODEL       = "gpt-4o-mini"
LLM_MAX_TOKENS  = 1600
LLM_TEMPERATURE = 0.3
MAX_RETRIEVAL   = 25
FUZZY_THRESHOLD = 0.55

CATEGORY_COLORS = {
    "Parameter": "#4ECDC4",
    "Process":   "#FF6B6B",
    "Equipment": "#45B7D1",
    "Outcome":   "#96CEB4",
    "Unknown":   "#FECA57",
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
.block-container { padding-top: 1.5rem; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.app-header h1 { color: #e0e0e0; margin: 0; font-size: 1.9rem; }
.app-header p  { color: #a0a0b0; margin: 0.4rem 0 0; font-size: 0.95rem; }

/* ── Stat card ── */
.stat-card {
    background: #f7f8fc;
    border-left: 4px solid #0f3460;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
}

/* ── Equation card ── */
.eq-card {
    background: #fffbf0;
    border: 1px solid #f0c040;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin: 0.7rem 0;
}
.eq-card .eq-id   { font-size: 0.78rem; color: #888; font-weight: 600; }
.eq-card .eq-type { font-size: 0.78rem; color: #b07800; background: #fff3cd;
                    padding: 1px 6px; border-radius: 4px; }
.eq-card .eq-plain { font-size: 0.88rem; color: #444; margin-top: 0.4rem; }

/* ── Trace card ── */
.trace-card {
    background: #f0f4ff;
    border: 1px solid #c0d0ff;
    border-left: 4px solid #3b5bdb;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
}
.trace-card h4 { color: #1a3a8f; margin: 0 0 0.5rem; }
.trace-card .score-row { display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 0.4rem 0; }
.score-badge {
    background: #3b5bdb;
    color: white;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.entity-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 2px 3px;
}

/* ── Citation badge ── */
.citation {
    display: inline-block;
    background: #e8f4fd;
    color: #0066cc;
    border: 1px solid #b0d4f0;
    border-radius: 4px;
    padding: 0 5px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0 1px;
}

/* ── Source badge ── */
.source-badge {
    display: inline-block;
    background: #e8f5e9;
    color: #2e7d32;
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 0.78rem;
    margin: 1px 2px;
}

/* ── Sidebar upload box ── */
.upload-hint {
    font-size: 0.78rem;
    color: #888;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND CLASSES  (adapted from phase8_pe_chatbot.py to use in-memory data)
# ─────────────────────────────────────────────────────────────────────────────

class OntologyLoader:
    """Loads pe.json + OPB.json concepts from parsed dicts."""

    def __init__(self, pe_data: Optional[dict] = None, opb_data: Optional[dict] = None):
        self.pe_concepts:  Dict[str, Dict] = {}
        self.opb_concepts: Dict[str, Dict] = {}
        if pe_data:
            for label, c in pe_data.get("concepts", {}).items():
                self.pe_concepts[label.lower().strip()] = c
        if opb_data:
            for label, c in opb_data.get("concepts", {}).items():
                self.opb_concepts[label.lower().strip()] = c

    def define(self, term: str) -> Optional[Dict]:
        key = term.lower().strip()
        for concepts, source in [
            (self.pe_concepts,  "PE ontology"),
            (self.opb_concepts, "OPB (physics) ontology"),
        ]:
            if key in concepts:
                c = concepts[key]
                return {
                    "label":      c.get("label", term),
                    "definition": c.get("definition", "No definition available"),
                    "category":   c.get("category",   "Unknown"),
                    "synonyms":   c.get("synonyms",    []),
                    "source":     source,
                }
            for label, c in concepts.items():
                if key in label or label in key:
                    return {
                        "label":      c.get("label", label),
                        "definition": c.get("definition", "No definition available"),
                        "category":   c.get("category", "Unknown"),
                        "synonyms":   c.get("synonyms", []),
                        "source":     source,
                    }
        return None


class EquationsLoader:
    """Loads equations_database.json from a parsed dict."""

    def __init__(self, eq_data: Optional[dict] = None):
        self.equations: Dict[str, Dict] = {}
        if eq_data:
            for eq in eq_data.get("equations", []):
                eq_id = eq.get("equation_id", "")
                if eq_id:
                    self.equations[eq_id] = eq
        self.total = len(self.equations)

    def get(self, equation_id: str) -> Optional[Dict]:
        return self.equations.get(equation_id)

    def all_equations(self) -> List[Dict]:
        return list(self.equations.values())


class GraphRetriever:
    """Loads merged_triplets.json from a parsed dict and indexes everything."""

    def __init__(self, merged_data: dict):
        self.graph_meta = {k: v for k, v in merged_data.items()
                           if k != "merged_triplets"}
        self.triplets:        List[Dict]        = merged_data.get("merged_triplets", [])
        self.nx_graph:        nx.DiGraph        = nx.DiGraph()
        self.merged_id_index: Dict[str, int]    = {}
        self.entity_index:    Dict[str, List]   = defaultdict(list)
        self.relation_index:  Dict[str, List]   = defaultdict(list)
        self.all_entities:    set               = set()
        self.all_relations:   set               = set()
        self._index()

    def _index(self):
        for idx, t in enumerate(self.triplets):
            e1  = t.get("entity1",  "").strip()
            e2  = t.get("entity2",  "").strip()
            rel = t.get("relation", "").strip()
            mid = t.get("merged_id", "")
            if not (e1 and e2 and rel):
                continue
            if mid:
                self.merged_id_index[mid.upper()] = idx
            self.entity_index[e1.lower()].append(idx)
            self.entity_index[e2.lower()].append(idx)
            self.relation_index[rel.lower()].append(idx)
            self.all_entities.add(e1)
            self.all_entities.add(e2)
            self.all_relations.add(rel)
            self.nx_graph.add_edge(e1, e2,
                relation        = rel,
                consensus_score = t.get("consensus_score", 0.5),
                paper_count     = t.get("paper_count", 1),
                merged_id       = mid,
            )

    # ── Fuzzy matching ────────────────────────────────────────────────────
    def _fuzzy_match(self, query: str) -> List[str]:
        q = query.lower().strip()
        hits: List[Tuple[str, float]] = []
        for entity in self.all_entities:
            el = entity.lower()
            if q == el:
                hits.append((entity, 1.0)); continue
            if q in el or el in q:
                hits.append((entity, 0.90)); continue
            q_w = set(q.split()); el_w = set(el.split())
            ov = q_w & el_w
            if ov:
                sc = len(ov) / max(len(q_w), len(el_w))
                if sc >= 0.45:
                    hits.append((entity, 0.70 + sc * 0.20)); continue
            ratio = SequenceMatcher(None, q, el).ratio()
            if ratio >= FUZZY_THRESHOLD:
                hits.append((entity, ratio * 0.75))
        hits.sort(key=lambda x: x[1], reverse=True)
        return [h[0] for h in hits[:12]]

    # ── Main search ───────────────────────────────────────────────────────
    def search(self, query: str, max_results: int = MAX_RETRIEVAL) -> List[Dict]:
        q_words = query.lower().strip().split()
        idx_set: set = set()
        for word in q_words:
            if len(word) > 2:
                for ent in self._fuzzy_match(word):
                    idx_set.update(self.entity_index[ent.lower()])
        for i in range(len(q_words)):
            for j in range(i + 2, min(i + 5, len(q_words) + 1)):
                for ent in self._fuzzy_match(" ".join(q_words[i:j])):
                    idx_set.update(self.entity_index[ent.lower()])
        for word in q_words:
            if len(word) > 3:
                for rel in self.all_relations:
                    if word in rel.lower():
                        idx_set.update(self.relation_index[rel.lower()])
        results = [self.triplets[i] for i in idx_set if i < len(self.triplets)]
        results.sort(key=lambda t: t.get("consensus_score", 0) *
                                    t.get("paper_count", 1), reverse=True)
        return results[:max_results]

    # ── Trace by M-ID ─────────────────────────────────────────────────────
    def get_by_merged_id(self, mid: str) -> Optional[Dict]:
        raw = mid.strip().upper()
        digits = raw.lstrip("M")
        key = f"M{digits.zfill(5)}" if digits.isdigit() else raw
        idx = self.merged_id_index.get(key)
        return None if idx is None else self.triplets[idx]


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE  (retrieval + LLM)
# ─────────────────────────────────────────────────────────────────────────────

class PEEngine:
    """
    Wraps retrieval + LLM into one object.
    Returns structured dicts so Streamlit can render each part correctly.
    """

    SYSTEM_PROMPT = """You are an expert assistant specialising in semiconductor plasma etching physics. \
You have access to a knowledge graph built from peer-reviewed research papers.

YOUR KNOWLEDGE SOURCES:
1. Knowledge graph triplets — specific physics relationships extracted from papers
2. Mathematical equations — formulas backing those relationships, with physical quantities
3. Ontology definitions — precise domain definitions for every entity
4. General plasma physics and semiconductor engineering knowledge

HOW TO ANSWER:
- PRIMARY SOURCE: use the provided knowledge graph data when relevant
- EQUATIONS: reference specific equations when explaining mechanisms (write them as $$latex$$)
- REASONING: explain WHY relationships exist using physics principles
- INFERENCE: make logical inferences beyond what is explicitly stated

CRITICAL CITATION RULES:
- Cite merged triplets using their Merged ID in square brackets: [M00001]
- Multiple citations: [M00001][M00003]
- Place citations at the END of the sentence they support
- Every factual claim from the graph MUST be cited

FORMATTING RULES — VERY IMPORTANT:
- Use ## for main section headers (NOT **bold**)
- Use ### for sub-headers
- Use bullet points with - for lists
- Write inline LaTeX equations as $$equation$$ so they render properly
- Example header: ## Key Relationships  (NOT **Key Relationships:**)

ANSWER STRUCTURE:
## Introduction
Brief 1-2 sentence overview.

## [Topic 1]
Details with citations [M00001][M00002].

## [Topic 2]
Details with citations [M00005].

## Mathematical Perspective
Equations and quantities involved. Write equations as $$...$$

## Summary
Key conclusions."""

    def __init__(self, retriever: GraphRetriever, eq_loader: EquationsLoader,
                 ontology: OntologyLoader, api_key: str):
        self.retriever  = retriever
        self.eq_loader  = eq_loader
        self.ontology   = ontology
        self.client     = OpenAI(api_key=api_key)
        self.history:   List[Dict] = []

    # ── Build LLM context ─────────────────────────────────────────────────
    def _build_context(self, triplets: List[Dict]) -> str:
        if not triplets:
            return "No relevant triplets found in the knowledge graph."
        lines = []

        # Section 1 — relationships
        lines.append("PHYSICS RELATIONSHIPS (from research papers):\n")
        for i, t in enumerate(triplets, 1):
            mid   = t.get("merged_id", "?")
            e1    = t.get("entity1",   "?")
            rel   = t.get("relation",  "?")
            e2    = t.get("entity2",   "?")
            score = t.get("consensus_score", 0.0)
            papers= t.get("paper_count", 1)
            lines.append(f"[{i}][{mid}]  {e1}  --[{rel}]-->  {e2}")
            lines.append(f"         Consensus: {score:.2f}  |  Papers: {papers}")
        lines.append("")

        # Section 2 — equations
        eq_lines = []
        for t in triplets:
            if not t.get("has_equation", False):
                continue
            mid = t.get("merged_id", "?")
            for eq in t.get("equation_refs", []):
                latex = eq.get("equation_latex", eq.get("equation_raw", ""))
                plain = eq.get("plain_english", "")
                qty   = eq.get("physical_quantities", [])
                entry = f"  [{mid}] {eq.get('equation_id','')} ({eq.get('equation_type','')}): {latex}"
                if plain:
                    entry += f"\n         Meaning: {plain}"
                if qty:
                    syms = ", ".join(
                        f"{q.get('symbol','?')} = {q.get('name','?')} [{q.get('unit','')}]"
                        for q in qty[:4])
                    entry += f"\n         Quantities: {syms}"
                eq_lines.append(entry)
        if eq_lines:
            lines.append("MATHEMATICAL FOUNDATIONS:\n")
            lines.extend(eq_lines)
            lines.append("")

        # Section 3 — ontology
        seen = set()
        ont_lines = []
        for t in triplets:
            for side in ("entity1", "entity2"):
                ent  = t.get(side, "")
                cat  = t.get(f"{side}_category", "Unknown")
                defn = t.get(f"{side}_definition", "")
                if ent and ent not in seen:
                    seen.add(ent)
                    ont_lines.append(f"  • {ent} [{cat}]: {defn}" if defn
                                     else f"  • {ent} [{cat}]")
        if ont_lines:
            lines.append("ONTOLOGY CONTEXT:\n")
            lines.extend(ont_lines)
            lines.append("")

        # Section 4 — provenance
        lines.append("PAPER PROVENANCE:\n")
        for i, t in enumerate(triplets, 1):
            mid     = t.get("merged_id", "?")
            t_ids   = t.get("contributing_triplet_ids", [])
            sources = t.get("source_files", [])
            src_str = ", ".join(Path(s).stem for s in sources) or "unknown"
            lines.append(f"  [{i}][{mid}]  sources: {src_str}")
            if t_ids:
                t_str = ", ".join(t_ids[:6]) + (" ..." if len(t_ids) > 6 else "")
                lines.append(f"              T-IDs: {t_str}")
        lines.append("")

        return "\n".join(lines)

    # ── Main chat call ────────────────────────────────────────────────────
    def chat(self, user_query: str) -> Dict:
        """
        Returns dict:
          answer   : str   (LLM markdown answer)
          triplets : list  (retrieved merged triplets)
          eq_cards : list  (equation dicts for rendering)
          error    : str   (empty if OK)
        """
        relevant = self.retriever.search(user_query, max_results=MAX_RETRIEVAL)
        context  = self._build_context(relevant)

        user_msg = (
            f"USER QUESTION:\n{user_query}\n\n"
            f"{context}\n"
            "Answer using ## section headers and [M-ID] citations. "
            "Write LaTeX equations as $$equation$$."
        )

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        for ex in self.history[-4:]:
            messages.append({"role": "user",      "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": user_msg})

        try:
            resp   = self.client.chat.completions.create(
                model=LLM_MODEL, messages=messages,
                max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            return {"answer": "", "triplets": relevant,
                    "eq_cards": [], "error": str(e)}

        # Collect equation cards for all equation-backed triplets
        eq_cards = []
        seen_eq_ids = set()
        for t in relevant:
            if not t.get("has_equation", False):
                continue
            for eq_ref in t.get("equation_refs", []):
                eq_id = eq_ref.get("equation_id", "")
                if eq_id and eq_id not in seen_eq_ids:
                    seen_eq_ids.add(eq_id)
                    eq_cards.append({
                        "eq_id":    eq_id,
                        "eq_type":  eq_ref.get("equation_type", "other"),
                        "latex":    eq_ref.get("equation_latex",
                                               eq_ref.get("equation_raw", "")),
                        "plain":    eq_ref.get("plain_english", ""),
                        "qty":      eq_ref.get("physical_quantities", []),
                        "sources":  eq_ref.get("source_papers", []),
                        "mid":      t.get("merged_id", ""),
                    })

        self.history.append({
            "user":      user_query,
            "assistant": answer,
            "timestamp": datetime.now().isoformat(),
        })

        return {"answer": answer, "triplets": relevant,
                "eq_cards": eq_cards, "error": ""}

    # ── Define ────────────────────────────────────────────────────────────
    def define_term(self, term: str) -> Optional[Dict]:
        term_lower = term.lower().strip()
        for t in self.retriever.triplets:
            for side in ("entity1", "entity2"):
                if t.get(side, "").lower() == term_lower:
                    defn = t.get(f"{side}_definition", "")
                    if defn:
                        return {
                            "label":      t[side],
                            "definition": defn,
                            "category":   t.get(f"{side}_category", "Unknown"),
                            "synonyms":   t.get(f"{side}_synonyms", []),
                            "source":     "Knowledge graph (Phase 6 embedded)",
                        }
        return self.ontology.define(term)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_eq_card(eq: Dict):
    """Render one equation card with st.latex() for the formula."""
    eq_id   = eq.get("eq_id",   "")
    eq_type = eq.get("eq_type", "other")
    latex   = eq.get("latex",   "")
    plain   = eq.get("plain",   "")
    qty     = eq.get("qty",     [])
    mid     = eq.get("mid",     "")
    sources = eq.get("sources", [])

    # Pre-compute mid badge outside f-string (backslash not allowed in f-expr < py3.12)
    mid_badge   = f'&nbsp;&nbsp;<span class="eq-id">&#8594; {mid}</span>' if mid else ""
    eq_type_fmt = eq_type.replace("_", " ").title()
    st.markdown(
        f'<div class="eq-card">'
        f'<span class="eq-id">{eq_id}</span>&nbsp;&nbsp;'
        f'<span class="eq-type">{eq_type_fmt}</span>'
        f'{mid_badge}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Render LaTeX
    clean_latex = latex.strip()
    # Remove surrounding $$ if already present
    clean_latex = re.sub(r"^\$\$", "", clean_latex)
    clean_latex = re.sub(r"\$\$$", "", clean_latex)
    clean_latex = clean_latex.strip()
    if clean_latex and clean_latex.lower() not in ("null", "none", ""):
        try:
            st.latex(clean_latex)
        except Exception:
            st.code(clean_latex, language=None)

    if plain:
        st.markdown(f'<p class="eq-plain">📝 {plain}</p>', unsafe_allow_html=True)

    if qty:
        rows = [{"Symbol": q.get("symbol",""), "Quantity": q.get("name",""),
                 "Unit": q.get("unit","")} for q in qty]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=False)

    if sources:
        src_str = " ".join(
            f'<span class="source-badge">{Path(s).stem}</span>'
            for s in sources[:3])
        st.markdown(src_str, unsafe_allow_html=True)


def render_triplet_table(triplets: List[Dict]):
    """Render retrieved triplets as a styled dataframe."""
    if not triplets:
        return

    # Deduplicate by merged_id
    seen = set()
    unique = []
    for t in triplets:
        mid = t.get("merged_id", "")
        if mid not in seen:
            seen.add(mid); unique.append(t)

    rows = []
    for t in unique:
        t_ids   = t.get("contributing_triplet_ids", [])
        sources = t.get("source_files", [])
        rows.append({
            "Merged ID":  t.get("merged_id",        ""),
            "Entity 1":   t.get("entity1",           ""),
            "Relation":   t.get("relation",          ""),
            "Entity 2":   t.get("entity2",           ""),
            "Papers":     t.get("paper_count",        1),
            "Consensus":  round(t.get("consensus_score", 0.0), 3),
            "Eq?":        "✓" if t.get("has_equation", False) else "✗",
            "T-IDs":      ", ".join(t_ids[:5]) + (" …" if len(t_ids) > 5 else ""),
            "Sources":    ", ".join(Path(s).name for s in sources[:2]),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"{len(unique)} unique merged triplets used · "
               f"{len(triplets)} retrieved before deduplication")


def render_trace_card(t: Dict):
    """Render a full provenance card for one merged triplet."""
    mid   = t.get("merged_id",            "?")
    e1    = t.get("entity1",              "?")
    rel   = t.get("relation",             "?")
    e2    = t.get("entity2",              "?")
    score = t.get("consensus_score",      0.0)
    cv    = t.get("cross_validation_score", 0.0)
    ont   = t.get("ontology_score",       0.0)
    conf  = t.get("avg_confidence",       0.0)
    size  = t.get("cluster_size",         1)
    papers= t.get("paper_count",          1)

    e1_cat  = t.get("entity1_category",  "Unknown")
    e2_cat  = t.get("entity2_category",  "Unknown")
    e1_defn = t.get("entity1_definition", "")
    e2_defn = t.get("entity2_definition", "")
    e1_syns = t.get("entity1_synonyms",  [])
    e2_syns = t.get("entity2_synonyms",  [])
    t_ids   = t.get("contributing_triplet_ids", [])
    sources = t.get("source_files",      [])
    eq_refs = t.get("equation_refs",     [])

    c1_color = CATEGORY_COLORS.get(e1_cat, "#FECA57")
    c2_color = CATEGORY_COLORS.get(e2_cat, "#FECA57")

    st.markdown(f"### 🔎 Trace: `{mid}`")

    # Relationship row
    st.markdown(
        f'<span class="entity-pill" style="background:{c1_color}20;'
        f'color:{c1_color};border:1px solid {c1_color};">{e1} [{e1_cat}]</span>'
        f'&nbsp; <b>──[ {rel} ]──▶</b> &nbsp;'
        f'<span class="entity-pill" style="background:{c2_color}20;'
        f'color:{c2_color};border:1px solid {c2_color};">{e2} [{e2_cat}]</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Score row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Consensus",       f"{score:.3f}")
    col2.metric("Cross-Validation",f"{cv:.3f}")
    col3.metric("Ontology Score",  f"{ont:.3f}")
    col4.metric("Avg Confidence",  f"{conf:.3f}")

    st.markdown("")

    # Entity details
    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown(f"**Entity 1: {e1}**")
        st.markdown(f"- **Category:** {e1_cat}")
        if e1_defn:
            st.markdown(f"- **Definition:** {e1_defn}")
        if e1_syns:
            st.markdown(f"- **Synonyms:** {', '.join(e1_syns)}")
    with ec2:
        st.markdown(f"**Entity 2: {e2}**")
        st.markdown(f"- **Category:** {e2_cat}")
        if e2_defn:
            st.markdown(f"- **Definition:** {e2_defn}")
        if e2_syns:
            st.markdown(f"- **Synonyms:** {', '.join(e2_syns)}")

    st.markdown("---")

    # Provenance
    pc1, pc2, pc3 = st.columns(3)
    pc1.markdown(f"**Cluster size:** {size}")
    pc2.markdown(f"**Papers:** {papers}")
    pc3.markdown(f"**T-IDs:** {len(t_ids)}")

    if t_ids:
        st.markdown("**Contributing T-IDs:**")
        st.code(", ".join(t_ids), language=None)

    if sources:
        st.markdown("**Source papers:**")
        for s in sources:
            st.markdown(f"- {Path(s).name}")

    # Equations
    if eq_refs:
        st.markdown("---")
        st.markdown("**⚗️ Equations backing this relationship:**")
        for eq in eq_refs:
            render_eq_card({
                "eq_id":   eq.get("equation_id",   ""),
                "eq_type": eq.get("equation_type", "other"),
                "latex":   eq.get("equation_latex", eq.get("equation_raw", "")),
                "plain":   eq.get("plain_english", ""),
                "qty":     eq.get("physical_quantities", []),
                "sources": eq.get("source_papers", []),
                "mid":     mid,
            })


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "engine":        None,
        "retriever":     None,
        "eq_loader":     None,
        "initialized":   False,
        "chat_history":  [],   # list of {role, content, triplets, eq_cards}
        "graph_html":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Setup")

    api_key = st.text_input("OpenAI API Key", type="password",
                             placeholder="sk-...",
                             help="Your key is never stored.")

    st.markdown("---")
    st.markdown("### 📂 Upload Files")

    merged_file = st.file_uploader(
        "merged_triplets.json ⭐ required",
        type=["json"], key="up_merged",
    )
    st.markdown('<p class="upload-hint">Phase 6 output → outputs/phase6_merged_triplets/</p>',
                unsafe_allow_html=True)

    eq_file = st.file_uploader(
        "equations_database.json (optional)",
        type=["json"], key="up_eq",
    )
    st.markdown('<p class="upload-hint">Phase 2b output → outputs/phase2b_equations/</p>',
                unsafe_allow_html=True)

    pe_file = st.file_uploader(
        "pe.json (optional)",
        type=["json"], key="up_pe",
    )
    st.markdown('<p class="upload-hint">opb/pe/pe.json</p>', unsafe_allow_html=True)

    opb_file = st.file_uploader(
        "OPB.json (optional)",
        type=["json"], key="up_opb",
    )
    st.markdown('<p class="upload-hint">opb/physics/OPB.json</p>', unsafe_allow_html=True)

    graph_html_file = st.file_uploader(
        "pe_interactive.html (optional)",
        type=["html"], key="up_html",
        help="Phase 7 PyVis graph for the Graph tab",
    )
    st.markdown('<p class="upload-hint">outputs/phase7_graph/pe_interactive.html</p>',
                unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🚀 Initialize", use_container_width=True,
                 disabled=(not merged_file or not api_key)):
        with st.spinner("Loading knowledge graph …"):
            try:
                merged_data = json.load(merged_file)
                eq_data     = json.load(eq_file)     if eq_file  else None
                pe_data     = json.load(pe_file)     if pe_file  else None
                opb_data    = json.load(opb_file)    if opb_file else None
                graph_html  = (graph_html_file.read().decode("utf-8")
                               if graph_html_file else None)

                retriever = GraphRetriever(merged_data)
                eq_loader = EquationsLoader(eq_data)
                ontology  = OntologyLoader(pe_data, opb_data)
                engine    = PEEngine(retriever, eq_loader, ontology, api_key)

                st.session_state.engine      = engine
                st.session_state.retriever   = retriever
                st.session_state.eq_loader   = eq_loader
                st.session_state.initialized = True
                st.session_state.chat_history = []
                st.session_state.graph_html  = graph_html

                st.success(f"✅ Ready! {len(retriever.triplets)} triplets · "
                           f"{eq_loader.total} equations")
            except Exception as e:
                st.error(f"❌ {e}")

    # ── Stats panel ──────────────────────────────────────────────────────
    if st.session_state.initialized:
        r = st.session_state.retriever
        el = st.session_state.eq_loader
        st.markdown("---")
        st.markdown("### 📊 Graph Stats")

        n_eq  = sum(1 for t in r.triplets if t.get("has_equation", False))
        n_multi = sum(1 for t in r.triplets if t.get("paper_count", 1) > 1)
        avg_sc  = (sum(t.get("consensus_score", 0) for t in r.triplets)
                   / len(r.triplets)) if r.triplets else 0

        st.markdown(f"""
<div class="stat-card">
🏷️ <b>Entities</b>: {len(r.all_entities)}<br>
🔗 <b>Relation types</b>: {len(r.all_relations)}<br>
📊 <b>Merged triplets</b>: {len(r.triplets)}<br>
⚗️ <b>Equation-backed</b>: {n_eq}<br>
📄 <b>Multi-paper</b>: {n_multi}<br>
🎯 <b>Avg consensus</b>: {avg_sc:.3f}<br>
🧮 <b>Equations in DB</b>: {el.total}
</div>""", unsafe_allow_html=True)

        sources = r.graph_meta.get("source_files", [])
        if sources:
            st.markdown("**Source papers:**")
            for s in sources:
                st.markdown(f"- {Path(s).name}")

        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.engine.history = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🔬 Semiconductor Plasma Etching — Knowledge Graph Chatbot by BHASKAR</h1>
  <p>Physics-Aware Knowledge Graph · LLM Reasoning · Multi-Paper Consensus</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.initialized:
    st.info("👈 Upload **merged_triplets.json**, enter your **OpenAI API key**, "
            "then click **Initialize** to start.")

    st.markdown("### 💡 What you can ask:")
    examples = [
        "What factors affect etch rate in plasma etching?",
        "How does RF power influence ion flux?",
        "Explain the transport equation for electron density",
        "What is the relationship between electron density and sheath potential?",
        "How does a PINN solve plasma equations?",
        "Which parameters control selectivity?",
        "What do multiple papers agree on regarding electron density?",
    ]
    for ex in examples:
        st.markdown(f"- {ex}")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_search, tab_eq, tab_trace, tab_graph, tab_explore = st.tabs([
    "💬 Chat", "🔍 Search", "⚗️ Equations", "🔎 Trace", "🕸️ Graph", "📈 Explore"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### Ask anything about semiconductor plasma etching physics")

    # ── Render previous messages ─────────────────────────────────────────
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🔬"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])

                # Equation cards
                eq_cards = msg.get("eq_cards", [])
                if eq_cards:
                    with st.expander(f"⚗️ {len(eq_cards)} equation(s) referenced",
                                     expanded=False):
                        for eq in eq_cards:
                            render_eq_card(eq)
                            st.markdown("---")

                # Triplet table
                triplets = msg.get("triplets", [])
                if triplets:
                    with st.expander(
                        f"📊 {len(set(t.get('merged_id','') for t in triplets))} "
                        f"knowledge graph triplets used", expanded=False
                    ):
                        render_triplet_table(triplets)

    # ── Chat input ────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask a question about plasma etching physics …")

    if user_input:
        # Show user message immediately
        with st.chat_message("user", avatar="🔬"):
            st.markdown(user_input)
        st.session_state.chat_history.append({
            "role": "user", "content": user_input,
        })

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Searching graph · 🧠 Generating response …"):
                result = st.session_state.engine.chat(user_input)

            if result["error"]:
                st.error(f"❌ LLM error: {result['error']}")
            else:
                st.markdown(result["answer"])

                eq_cards = result.get("eq_cards", [])
                if eq_cards:
                    with st.expander(f"⚗️ {len(eq_cards)} equation(s) referenced",
                                     expanded=True):
                        for eq in eq_cards:
                            render_eq_card(eq)
                            st.markdown("---")

                triplets = result.get("triplets", [])
                if triplets:
                    with st.expander(
                        f"📊 {len(set(t.get('merged_id','') for t in triplets))} "
                        f"knowledge graph triplets used", expanded=False
                    ):
                        render_triplet_table(triplets)

                st.session_state.chat_history.append({
                    "role":     "assistant",
                    "content":  result["answer"],
                    "triplets": triplets,
                    "eq_cards": eq_cards,
                })


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEARCH
# ═════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.markdown("### 🔍 Search the Knowledge Graph")

    col1, col2 = st.columns([4, 1])
    with col1:
        search_term = st.text_input("Search term:", placeholder="e.g. electron density, etch rate")
    with col2:
        max_res = st.number_input("Max results", min_value=5, max_value=50, value=20)

    if st.button("Search", use_container_width=True, key="btn_search"):
        if search_term.strip():
            results = st.session_state.retriever.search(search_term, max_results=max_res)
            if results:
                st.success(f"Found **{len(results)}** results for '{search_term}'")
                rows = []
                for t in results:
                    rows.append({
                        "Merged ID":  t.get("merged_id", ""),
                        "Entity 1":   t.get("entity1",   ""),
                        "Relation":   t.get("relation",  ""),
                        "Entity 2":   t.get("entity2",   ""),
                        "Papers":     t.get("paper_count", 1),
                        "Consensus":  round(t.get("consensus_score", 0), 3),
                        "Eq?":        "✓" if t.get("has_equation", False) else "✗",
                        "Sources":    ", ".join(
                            Path(s).name for s in t.get("source_files", [])[:2]),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)
            else:
                st.warning(f"No results for '{search_term}'")

    # ── Define term ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📖 Define a Term")
    define_term = st.text_input("Look up definition:", placeholder="e.g. electron density")
    if st.button("Define", key="btn_define"):
        if define_term.strip():
            result = st.session_state.engine.define_term(define_term)
            if result:
                c1, c2 = st.columns([2, 1])
                c1.markdown(f"**{result['label']}**")
                c2.markdown(
                    f'<span class="entity-pill" style="background:'
                    f'{CATEGORY_COLORS.get(result["category"],"#FECA57")}20;'
                    f'color:{CATEGORY_COLORS.get(result["category"],"#888")};">'
                    f'{result["category"]}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Definition:** {result['definition']}")
                if result.get("synonyms"):
                    st.markdown(f"**Synonyms:** {', '.join(result['synonyms'])}")
                st.caption(f"Source: {result['source']}")
            else:
                st.warning(f"'{define_term}' not found in graph, PE ontology, or OPB.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — EQUATIONS
# ═════════════════════════════════════════════════════════════════════════════
with tab_eq:
    st.markdown("### ⚗️ Equations Database")

    eq_loader = st.session_state.eq_loader
    all_eqs   = eq_loader.all_equations()

    if not all_eqs:
        st.info("No equations database loaded. Upload equations_database.json to see equations here.")
    else:
        # Group by type
        by_type: Dict[str, List] = defaultdict(list)
        for eq in all_eqs:
            by_type[eq.get("equation_type", "other")].append(eq)

        st.markdown(f"**{len(all_eqs)} equations across {len(by_type)} types**")

        # Filter
        all_types = ["All"] + sorted(by_type.keys())
        selected_type = st.selectbox("Filter by type:", all_types)

        if selected_type == "All":
            to_show = all_eqs
        else:
            to_show = by_type[selected_type]

        st.markdown(f"Showing **{len(to_show)}** equations")
        st.markdown("---")

        for eq in to_show:
            render_eq_card({
                "eq_id":   eq.get("equation_id",   ""),
                "eq_type": eq.get("equation_type", "other"),
                "latex":   eq.get("equation_latex", eq.get("equation_raw", "")),
                "plain":   eq.get("plain_english", ""),
                "qty":     eq.get("physical_quantities", []),
                "sources": eq.get("source_papers", []),
                "mid":     "",
            })
            st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRACE
# ═════════════════════════════════════════════════════════════════════════════
with tab_trace:
    st.markdown("### 🔎 Trace a Merged Triplet")
    st.markdown("Enter any Merged ID to see its full provenance — "
                "entity definitions, contributing T-IDs, source papers, and equations.")

    mid_input = st.text_input(
        "Merged ID:", placeholder="e.g. M00001  or  1  or  m00001"
    )

    if st.button("Trace", use_container_width=False, key="btn_trace"):
        if mid_input.strip():
            t = st.session_state.retriever.get_by_merged_id(mid_input.strip())
            if t:
                render_trace_card(t)
            else:
                st.error(f"Merged ID `{mid_input.strip().upper()}` not found. "
                         f"Use the Search tab to find valid IDs.")

    # ── Quick browse ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Or browse by entity:**")
    all_ents = sorted(st.session_state.retriever.all_entities)
    selected_ent = st.selectbox("Select entity:", [""] + all_ents)
    if selected_ent:
        hits = st.session_state.retriever.search(selected_ent, max_results=10)
        if hits:
            st.markdown(f"**Triplets involving '{selected_ent}':**")
            rows = [{"Merged ID": t.get("merged_id",""),
                     "Entity 1":  t.get("entity1",""),
                     "Relation":  t.get("relation",""),
                     "Entity 2":  t.get("entity2",""),
                     "Consensus": round(t.get("consensus_score",0),3),
                     "Eq?":       "✓" if t.get("has_equation",False) else "✗"}
                    for t in hits]
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)
            st.caption("Click a Merged ID above and paste it into the Trace box to see full details.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — GRAPH
# ═════════════════════════════════════════════════════════════════════════════
with tab_graph:
    st.markdown("### 🕸️ Interactive Knowledge Graph")

    graph_html = st.session_state.graph_html

    if graph_html:
        st.markdown("Drag nodes · Hover for tooltips · Scroll to zoom")
        components.html(graph_html, height=700, scrolling=False)
    else:
        st.info(
            "**No graph file uploaded.**\n\n"
            "To see the interactive graph:\n"
            "1. Run `phase7_pe_graph_builder.py` to generate the graph\n"
            "2. Upload `outputs/phase7_graph/pe_interactive.html` in the sidebar\n"
            "3. Click **Initialize** again\n\n"
            "The graph shows all physics relationships as nodes and edges, "
            "with gold edges for equation-backed relationships."
        )
        # Show a simple NetworkX summary as fallback
        r = st.session_state.retriever
        st.markdown("---")
        st.markdown("**Graph summary (from loaded triplets):**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Nodes", r.nx_graph.number_of_nodes())
        m2.metric("Edges", r.nx_graph.number_of_edges())
        m3.metric("Components",
                  nx.number_weakly_connected_components(r.nx_graph))


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — EXPLORE
# ═════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("### 📈 Explore the Knowledge Graph")

    r  = st.session_state.retriever
    el = st.session_state.eq_loader

    # ── Top central nodes ─────────────────────────────────────────────────
    st.markdown("#### 🏆 Most Connected Entities")
    try:
        centrality = nx.degree_centrality(r.nx_graph)
        top_nodes  = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:15]
        cent_rows  = [{"Entity": n, "Centrality": round(c, 4),
                       "Degree": r.nx_graph.degree(n)} for n, c in top_nodes]
        st.dataframe(pd.DataFrame(cent_rows), hide_index=True, use_container_width=True)
    except Exception:
        st.warning("Could not compute centrality.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔗 All Relation Types")
        rels = sorted(r.all_relations)
        rel_counts = {rel: len(r.relation_index[rel.lower()]) for rel in rels}
        rel_rows   = [{"Relation": rel, "Triplet count": rel_counts[rel]}
                      for rel in sorted(rels, key=lambda x: rel_counts[x], reverse=True)]
        st.dataframe(pd.DataFrame(rel_rows), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### 🏷️ Category Distribution")
        cat_counts: Dict[str, int] = defaultdict(int)
        for t in r.triplets:
            cat_counts[t.get("entity1_category", "Unknown")] += 1
            cat_counts[t.get("entity2_category", "Unknown")] += 1
        cat_rows = [{"Category": cat, "Count": cnt}
                    for cat, cnt in sorted(cat_counts.items(),
                                           key=lambda x: x[1], reverse=True)]
        st.dataframe(pd.DataFrame(cat_rows), hide_index=True, use_container_width=True)

    st.markdown("---")

    # ── High-consensus triplets ───────────────────────────────────────────
    st.markdown("#### 🎯 Highest-Confidence Triplets")
    top_triplets = sorted(r.triplets,
                          key=lambda t: t.get("consensus_score", 0), reverse=True)[:20]
    top_rows = [{"Merged ID": t.get("merged_id",""),
                 "Entity 1":  t.get("entity1",""),
                 "Relation":  t.get("relation",""),
                 "Entity 2":  t.get("entity2",""),
                 "Consensus": round(t.get("consensus_score",0),3),
                 "Papers":    t.get("paper_count",1),
                 "Eq?":       "✓" if t.get("has_equation",False) else "✗"}
                for t in top_triplets]
    st.dataframe(pd.DataFrame(top_rows), hide_index=True, use_container_width=True)

    # ── Equation-backed triplets ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚗️ Equation-Backed Triplets")
    eq_triplets = [t for t in r.triplets if t.get("has_equation", False)]
    if eq_triplets:
        eq_rows = [{"Merged ID": t.get("merged_id",""),
                    "Entity 1":  t.get("entity1",""),
                    "Relation":  t.get("relation",""),
                    "Entity 2":  t.get("entity2",""),
                    "Consensus": round(t.get("consensus_score",0),3),
                    "# Equations": len(t.get("equation_refs",[]))}
                   for t in eq_triplets]
        st.dataframe(pd.DataFrame(eq_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No equation-backed triplets found.")
