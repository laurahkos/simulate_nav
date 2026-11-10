"""Simulate an LLM agent answering a student's question by browsing liverpool.ac.uk.

GPT-2 measures surprisal on every page and selects findings; in standard mode it
also ranks links by perplexity. An instruct model writes the answer and, with
--agentic, chooses each link. Outputs trace.txt and trace.json in --out-dir.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import re
import textwrap
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from urllib.parse import parse_qsl, urldefrag, urlencode, urlparse

import html2text
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    set_seed,
)

SEED_URLS = {
    "sociology": "https://www.liverpool.ac.uk/sociology-social-policy-and-criminology/",
    "geography": "https://www.liverpool.ac.uk/geography-and-planning/",
}
PROBE_MODEL = "gpt2"
ANSWER_MODEL = "microsoft/Phi-3-mini-4k-instruct"

MAX_PAGE_WORDS = 600
PROFILE_WORDS = 400
LINK_OPTIONS = 15  # links shown to the agentic model
LINKS_SCORED = 20  # per group (department / other) in standard mode
RUNNERS_PER_PAGE = 3  # non-chosen links kept for backtracking
BACKTRACK_MAX = 20
MIN_RELEVANCE = 0.5
KW_BONUS = 20
EVIDENCE_SENTENCES = 6
MIN_SOURCE_OVERLAP = 0.34  # below this, an answer sentence has no close evidence

SKIP_LINK_RE = re.compile(
    r"(?i)(cookie|privacy|accessibility|sitemap|login|sign.in|basket|cart|javascript:|"
    r"mailto:|tel:|#$|\.pdf$|\.docx?$|twitter\.com|facebook\.com|linkedin\.com|"
    r"instagram|youtube|rss)"
)
PG_KEYWORDS = re.compile(
    r"\b(postgraduate|phd|doctoral|pgr|pgt|master|msc|mres|mphil|training|workshop|"
    r"seminar|course|programme|opportunity|supervision|funding|bursary|scholarship|"
    r"career|skill|researcher|development|support|student)\b",
    re.I,
)
CHROME_LINE_RE = re.compile(
    r"(?i)^(skip(\s+(to\s+)?(main\s+)?content|\s+navigation)?|select\s+campus:?|"
    r"(necessary|analytical|marketing|functional|functionality)(\s+cookies?)?|"
    r"i\s+(accept|do\s+not\s+accept)\s*cookies?.*|cookies?\s+(enhance|are\s+essential|"
    r"information)|uk|china|india|online|on|off|analytics?|marketing|home|menu|study|"
    r"student\s+life|research(\s+and\s+innovation)?|about\s+us|staff|students|global|"
    r"search\s+the\s+site)\s*$"
)
NAV_PHRASE_RE = re.compile(
    r"(?i)(getting\s+to\s+liverpool|getting\s+around|find\s+accommodation|life\s+in\s+halls|"
    r"guild\s+of\s+students|sport\s+at\s+liverpool|mental\s+health\s+and\s+wellbeing|"
    r"visas\s+and\s+immigration|open\s+(study|student(\s+life)?|research|global)\s+menu|"
    r"select\s+campus|home\s+menu\s+menu|alternatively\s+use\s+our|what\s+types\s+of\s+page|"
    r"for\s+more\s+information.*cookie)"
)
PROSE_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|offers?|provides?|includes?|enables?|allows?|"
    r"supports?|gives?|helps?|covers?|consists?|focuses?|aims?|develops?|trains?|"
    r"prepares?|equips?|funds?|available|designed|open|required|based|run|held|taught)\b",
    re.I,
)
ANSWER_STOP_RE = re.compile(
    r"(###\s*(Exercise|Instruction|Question)|Exercise\s+\d+:|Question\s+\d+:"
    r"|\nA student asked:|\nBased only on the following)",
    re.I,
)
WORD_RE = re.compile(r"[A-Za-z]{4,}")
# fmt: off
REFINE_STOPWORDS = {
    "about", "being", "could", "other", "their", "there", "these", "those", "through",
    "where", "which", "while", "within", "would",
    "liverpool", "university", "department", "students",
}
# fmt: on

STRIP_CHROME_JS = """() => {
    ['nav', 'header', 'footer', '[role="banner"]', '[role="navigation"]', '.cookie-notice',
     '.cookie-banner', '#cookie-consent', '.site-header', '.site-footer', '.breadcrumb',
     '[aria-label="breadcrumb"]']
        .forEach(s => document.querySelectorAll(s).forEach(e => e.remove()));
    return (document.querySelector('main') || document.querySelector('[role="main"]')
            || document.querySelector('#main-content') || document.body).outerHTML;
}"""


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# GPT-2 probe


def load_probe(name: str):
    tok = GPT2Tokenizer.from_pretrained(name)
    model = GPT2LMHeadModel.from_pretrained(name)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return tok, model


def token_surprisals(tok, model, context: str, text: str) -> list[tuple[str, float]]:
    """Surprisal (bits) of each token of `text` given `context`."""
    window = model.config.n_positions
    ctx = tok.encode(context, add_special_tokens=False) or [tok.bos_token_id]
    ids = tok.encode(text, add_special_tokens=False)[: window - 1]
    if not ids:
        return []
    ctx = ctx[-(window - len(ids)) :]
    inp = torch.tensor([ctx + ids], device=model.device)
    with torch.no_grad():
        logprobs = torch.log_softmax(model(inp).logits[0, len(ctx) - 1 : -1].float(), dim=-1)
    target = torch.tensor(ids, device=model.device).unsqueeze(1)
    bits = (-logprobs.gather(1, target).squeeze(1) / math.log(2)).tolist()
    return [(tok.decode([t]), b) for t, b in zip(ids, bits)]


def perplexity(tok, model, context: str, text: str) -> float:
    bits = [b for _, b in token_surprisals(tok, model, context, " " + text.strip())]
    return 2 ** mean(bits) if bits else float("inf")


# Fetching


class FetchError(Exception):
    pass


def to_markdown(html: str, base_url: str = "") -> str:
    h = html2text.HTML2Text(baseurl=base_url)
    h.ignore_links, h.ignore_images, h.body_width = False, True, 0
    return h.handle(html)


def canonical(url: str) -> str:
    """Drop the fragment and utm_* tracking parameters."""
    parts = urlparse(urldefrag(url)[0])
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.startswith("utm_")]
    return parts._replace(query=urlencode(query)).geturl()


def url_key(url: str) -> str:
    """Identity of a page: canonical URL, trailing slash ignored."""
    parts = urlparse(canonical(url))
    return parts._replace(path=parts.path.rstrip("/")).geturl()


def extract_links(full_html: str, page_url: str) -> list[tuple[str, str]]:
    host, seen, links = urlparse(page_url).netloc, set(), []
    md = to_markdown(full_html, base_url=page_url)
    for text, href in re.findall(r"\[([^\]]{3,80})\]\((https?://[^\)]+)\)", md):
        href = canonical(href.strip())
        junk = SKIP_LINK_RE.search(href) or SKIP_LINK_RE.search(text)
        if junk or urlparse(href).netloc != host or url_key(href) in seen:
            continue
        seen.add(url_key(href))
        links.append((text.strip(), href))
    return links


def looks_like_prose(text: str) -> bool:
    return len(text) >= 40 and len(text.split()) >= 8 and bool(PROSE_RE.search(text))


def is_chrome(line: str) -> bool:
    return bool(CHROME_LINE_RE.match(line.strip()) or NAV_PHRASE_RE.search(line))


def extract_text(main_html: str) -> str:
    """Main content as plain text, from the first prose paragraph on."""
    md = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", to_markdown(main_html))
    md = re.sub(r"[#*_`>|\\]", "", md)
    lines = [line for line in md.splitlines() if not line.strip() or not is_chrome(line)]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", "\n".join(lines)) if p.strip()]
    start = next((i for i, p in enumerate(paragraphs) if looks_like_prose(p)), 0)
    text = "\n\n".join(paragraphs[start:])
    words = text.split()
    return " ".join(words[:MAX_PAGE_WORDS]) if len(words) > MAX_PAGE_WORDS else text


def render(url: str) -> tuple[str, str]:
    """(main_html, full_html). Waits for the document, then up to 5 s for the network to
    settle; analytics requests can keep it busy indefinitely, so that wait is best effort."""
    from playwright.sync_api import Error, sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent="Mozilla/5.0 Chrome/122")
            page.goto(url, timeout=30_000, wait_until="domcontentloaded")
            with contextlib.suppress(Error):
                page.wait_for_load_state("networkidle", timeout=5_000)
            return page.evaluate(STRIP_CHROME_JS), page.content()
        finally:
            browser.close()


def fetch(url: str, attempts: int = 2) -> tuple[str, list[tuple[str, str]]]:
    for attempt in range(attempts):
        try:
            main_html, full_html = render(url)
            break
        except Exception as e:
            if attempt == attempts - 1:
                raise FetchError(f"{type(e).__name__}: {e}") from e
    if len(main_html or "") < 50:
        raise FetchError("empty page")
    return extract_text(main_html), extract_links(full_html, url)


# Page analysis


def prose_sentences(text: str) -> list[str]:
    return [
        s
        for para in text.splitlines()
        if not NAV_PHRASE_RE.search(para)
        for s in re.split(r"(?<=[.!?])\s+", para.strip())
        if 6 <= len(s.split()) <= 80 and PROSE_RE.search(s) and not s.endswith("?")
    ]


def analyse_page(tok, model, prompt: str, page_text: str) -> dict:
    context = f"A postgraduate student asked: '{prompt}'\nRelevant information:\n"
    sentences = prose_sentences(page_text)
    prose = " ".join(sentences)
    profile = token_surprisals(
        tok, model, context, " ".join((prose or page_text).split()[:PROFILE_WORDS])
    )

    findings = []
    for i, sent in enumerate(sentences):
        kw = len(PG_KEYWORDS.findall(sent))
        if not kw:
            continue
        preceding = " ".join(sentences[max(0, i - 2) : i])  # never the sentence itself
        text = f" {sent}" if preceding else sent
        surp = mean(b for _, b in token_surprisals(tok, model, context + preceding, text))
        findings.append(
            {
                "sentence": sent,
                "kw_count": kw,
                "mean_surprisal": round(surp, 3),
                "interest": round(kw * min(surp, 8), 3),
            }
        )
    findings.sort(key=lambda f: -f["interest"])
    return {
        "relevance_score": round(
            100 * len(PG_KEYWORDS.findall(prose)) / max(len(prose.split()), 1), 2
        ),
        "mean_surprisal": round(mean(b for _, b in profile), 3),
        "findings": findings[:5],
        "surprisal_profile": [(t, round(b, 3)) for t, b in profile[:80]],
    }


# Navigation


def candidate_links(links, visited, start_url: str) -> list[dict]:
    """Unvisited links, department first (seed slug anywhere in the path), in page order."""
    slug = urlparse(start_url).path.strip("/").split("/")[0]
    seen = {url_key(v) for v in visited}
    cands = [
        {"text": t, "url": u, "dept": bool(slug) and slug in urlparse(u).path.split("/")}
        for t, u in links
        if url_key(u) not in seen
    ]
    return sorted(cands, key=lambda c: not c["dept"])


def score_links(tok, model, prompt: str, cands: list[dict]) -> list[dict]:
    context = (
        f"A postgraduate student is looking for: {prompt}\n"
        f"The most relevant page to visit next would be about:"
    )
    ranked = []
    for is_dept in (True, False):
        scored = []
        for c in [c for c in cands if c["dept"] == is_dept][:LINKS_SCORED]:
            perp = perplexity(tok, model, context, c["text"])
            kw = bool(PG_KEYWORDS.search(c["text"]))
            scored.append(
                {
                    **c,
                    "perplexity": round(perp, 2),
                    "kw": kw,
                    "score": round(perp - KW_BONUS * kw, 2),
                }
            )
        ranked += sorted(scored, key=lambda s: s["score"])
    return ranked


def pop_unvisited(stack: list[str], visited) -> str | None:
    seen = {url_key(v) for v in visited}
    while stack:
        url = stack.pop()
        if url_key(url) not in seen:
            return url
    return None


def push_runners(stack: list[str], ranked: list[dict], chosen: str | None):
    runners = [c["url"] for c in ranked if c["url"] != chosen][:RUNNERS_PER_PAGE]
    stack[:] = [u for u in stack if u not in runners] + runners[::-1]
    del stack[:-BACKTRACK_MAX]


def decide_standard(ranked, relevance, stack, visited) -> tuple[str | None, str]:
    best = ranked[0] if ranked else None
    if best and (relevance >= MIN_RELEVANCE or not stack):
        return best[
            "url"
        ], f"Relevance={relevance:.2f}. Following '{best['text']}' (perp={best['perplexity']:.1f})"
    url = pop_unvisited(stack, visited)
    why = "No links" if not best else f"Low relevance ({relevance:.2f})"
    return url, f"{why}. " + (f"Backtracking to {url}" if url else "Nothing left; stopping.")


# Instruct model


def load_instruct(name: str):
    if torch.cuda.device_count():
        dtype, device = torch.bfloat16, "auto" if torch.cuda.device_count() > 1 else "cuda:0"
    elif torch.backends.mps.is_available():
        dtype, device = torch.float16, "mps"
    else:
        dtype, device = torch.float32, "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device)
    return tok, model.eval()


def generate(
    tok, model, message: str, max_new_tokens: int, temperature=0.0, top_p=1.0, logits=False
):
    """Returns (prompt_text, new_token_ids, raw per-step logits or None)."""
    if getattr(tok, "chat_template", None):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True
        )
        ids = tok.encode(prompt, add_special_tokens=False, return_tensors="pt")  # template has BOS
    else:
        prompt = f"### Instruction:\n{message}\n\n### Response:\n"
        ids = tok.encode(prompt, return_tensors="pt")
    ids = ids.to(model.device)
    sampling = (
        {"do_sample": True, "temperature": temperature, "top_p": top_p}
        if temperature
        else {"do_sample": False}
    )
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            return_dict_in_generate=True,
            output_logits=logits,
            **sampling,
        )
    return prompt, out.sequences[0, ids.shape[1] :], out.logits if logits else None


def choose_link(tok, model, prompt, page_text, options, findings, page_num) -> dict:
    found = "\n".join(
        f"  - {f['sentence'][:120]}" for f in sorted(findings, key=lambda f: -f["interest"])[:4]
    )
    menu = "\n".join(f"  [{i}] {o['text']}  →  {o['url']}" for i, o in enumerate(options, 1))
    message = (
        f'You are helping a postgraduate student find information about: "{prompt}"\n\n'
        f"You are currently on page {page_num} of a university website. "
        f"The page contains the following relevant content:\n{page_text[:600]}\n\n"
        f"Information found so far:\n{found or '  (nothing relevant found yet)'}\n\n"
        f"Available links to follow:\n{menu}\n\n"
        f"Which link should be followed next to best answer the student's question? "
        f"Consider: does this page already answer the question? Which link is most "
        f"likely to lead to specific information about postgraduate provision, "
        f"training, or support?\n\n"
        f"Respond in exactly this format:\n"
        f"CHOICE: [number]\n"
        f"REASON: [one sentence explaining why]\n"
        f"STOP: [YES if the question is already answered, NO if more pages needed]"
    )
    _, ids, _ = generate(tok, model, message, max_new_tokens=120)
    raw = tok.decode(ids, skip_special_tokens=True).strip()
    choice = re.search(r"CHOICE\s*:\s*\[?(\d+)\]?", raw, re.I)
    reason = re.search(r"REASON\s*:\s*(.+?)(?:\n|STOP|$)", raw, re.I | re.S)
    stop = bool(re.search(r"STOP\s*:\s*YES", raw, re.I))
    idx = int(choice.group(1)) - 1 if choice else -1
    parse_ok = 0 <= idx < len(options)
    return {
        "url": options[idx]["url"] if parse_ok and not stop else None,
        "reasoning": reason.group(1).strip() if reason else raw,
        "raw": raw,
        "stop": stop,
        "parse_ok": parse_ok,
    }


def answer_question(tok, model, prompt: str, findings: list[dict]) -> dict:
    evidence = sorted(findings, key=lambda f: -f["interest"])[:EVIDENCE_SENTENCES]
    if not evidence:
        return {
            "text": "[No findings]",
            "generation_prompt": "",
            "token_log": [],
            "evidence_used": [],
            "sentence_sources": [],
            "grounding_stats": grounding_stats([], []),
        }
    message = (
        f'A student asked: "{prompt}"\n\n'
        f"Based only on the following information found on the university website, "
        f"write a clear and specific answer. Do not add information not present below.\n\n"
        f"Evidence:\n" + "\n".join(f"[{i}] {f['sentence']}" for i, f in enumerate(evidence, 1))
    )
    prompt_text, ids, step_logits = generate(
        tok, model, message, 300, temperature=0.3, top_p=0.9, logits=True
    )
    full = tok.decode(ids, skip_special_tokens=True)
    stop = ANSWER_STOP_RE.search(full)
    cut = stop.start() if stop else len(full)

    evidence_words = {w.lower() for f in evidence for w in WORD_RE.findall(f["sentence"])}
    words = [
        (m.start(), m.end(), m.group().lower(), m.group().lower() in evidence_words)
        for m in WORD_RE.finditer(full[:cut])
    ]

    token_log, end = [], 0
    for step, (tid, logit) in enumerate(zip(ids.tolist(), step_logits)):
        start = end  # character span of this token; special tokens are zero-width
        end = max(end, min(len(tok.decode(ids[: step + 1], skip_special_tokens=True)), len(full)))
        hits = [w for w in words if w[0] < end and start < w[1]]
        probs = torch.softmax(logit[0].float(), dim=-1)
        p = probs[tid].item()
        top_p, top_i = torch.topk(probs, 3)
        token_log.append(
            {
                "step": step,
                "token": tok.decode([tid]),
                "prob": round(p, 5),
                "surprisal": round(-math.log2(p + 1e-12), 3),
                "alternatives": [
                    {"token": tok.decode([i]), "prob": round(q, 4)}
                    for q, i in zip(top_p.tolist(), top_i.tolist())
                ],
                "in_answer": start < min(end, cut),
                "content": bool(hits),
                "grounded": bool(hits) and all(w[3] for w in hits),
            }
        )
    return {
        "text": full[:cut].strip(),
        "generation_prompt": prompt_text,
        "token_log": token_log,
        "grounding_stats": grounding_stats(words, [t for t in token_log if t["in_answer"]]),
        "sentence_sources": sentence_sources(full[:cut], evidence),
        "evidence_used": [f["sentence"] for f in evidence],
    }


def sentence_sources(answer: str, evidence: list[dict]) -> list[dict]:
    """For each answer sentence, the evidence item sharing most of its content words."""
    ev_words = [{w.lower() for w in WORD_RE.findall(f["sentence"])} for f in evidence]
    sources = []
    for sent in re.split(r"(?<=[.!?])\s+", answer.strip()):
        words = {w.lower() for w in WORD_RE.findall(sent)}
        if not words:
            continue
        overlaps = [len(words & ev) / len(words) for ev in ev_words]
        best = max(range(len(overlaps)), key=overlaps.__getitem__)
        sources.append(
            {"sentence": sent, "evidence": best + 1, "overlap": round(overlaps[best], 2)}
        )
    return sources


def is_risky(t: dict) -> bool:
    return t["content"] and not t["grounded"] and t["prob"] < 0.05


def grounding_stats(words, tokens) -> dict:
    """Grounding = share of content words (>= 4 letters) found verbatim in the evidence."""
    spans = []
    for risky, run in groupby(tokens, key=is_risky):
        run = list(run)
        if risky and len(run) >= 3:
            spans.append(
                {
                    "start": run[0]["step"],
                    "end": run[-1]["step"],
                    "text": "".join(t["token"] for t in run).strip(),
                    "mean_prob": round(mean(t["prob"] for t in run), 4),
                }
            )
    grounded = sum(w[3] for w in words)
    return {
        "ungrounded_words": list(dict.fromkeys(w[2] for w in words if not w[3])),
        "n_content_words": len(words),
        "n_grounded_words": grounded,
        "grounding_pct": round(100 * grounded / max(len(words), 1), 1),
        "n_answer_tokens": len(tokens),
        "mean_prob": round(mean(t["prob"] for t in tokens), 4),
        "mean_surprisal": round(mean(t["surprisal"] for t in tokens), 3),
        "hallucination_spans": spans[:5],
    }


# Trace


class Trace:
    def __init__(self, path: Path):
        self.path, self.lines = path, []

    def __call__(self, line: str = ""):
        self.lines.append(line)
        print(line, flush=True)

    def banner(self, title: str, char: str = "="):
        for line in ("", char * 70, title, char * 70):
            self(line)

    def wrapped(self, text: str, indent: str, width: int = 66, first: str | None = None):
        for i, line in enumerate(textwrap.wrap(text, width=width) or [""]):
            self((first if i == 0 and first is not None else indent) + line)

    def save(self):
        self.path.write_text("\n".join(self.lines), encoding="utf-8")


def log_analysis(log: Trace, a: dict):
    log()
    log(f"  Relevance (keyword density × 100): {a['relevance_score']}")
    log(f"  Mean surprisal: {a['mean_surprisal']} bits")
    for f in a["findings"][:3]:
        log.wrapped(f["sentence"], "      ", 60, first="    › ")
        log(f"      [kw={f['kw_count']}  surp={f['mean_surprisal']}  interest={f['interest']}]")
    words = [
        (t, s)
        for t, s in a["surprisal_profile"]
        if len(t.strip()) >= 3 and re.search(r"[a-zA-Z]{2,}", t)
    ]
    if words:
        log("  Highest-surprisal tokens:")
        for t, s in sorted(words, key=lambda x: -x[1])[:10]:
            log(f"    {s:5.2f}  {'█' * min(int(s), 15):<15}  {t!r}")


def log_links(log: Trace, ranked: list[dict], chosen: str | None):
    scored = any("score" in c for c in ranked)
    log()
    log("        #  section     " + ("perplexity  score  " if scored else "") + "link text")
    for i, c in enumerate(ranked, 1):
        arrow = "→" if c["url"] == chosen else " "
        section = "department" if c["dept"] else "university"
        cols = f"{c['perplexity']:10.1f}  {c['score']:5.1f}  " if scored else ""
        log(f"  {arrow} {i:>4}  {section:<10}  {cols}{c['text'][:60]}")
    if scored:
        log(f"  score = perplexity − {KW_BONUS} if the text has a PG keyword")
        log("  department links rank first; lowest score first within each section")


def log_answer(log: Trace, answer: dict, model_name: str):
    gs = answer["grounding_stats"]
    log.banner(f"GENERATED ANSWER  [{model_name}]")
    log("  Evidence given to the model:")
    for i, sent in enumerate(answer["evidence_used"], 1):
        log.wrapped(sent, "        ", 62, first=f"    [{i}] ")
    log()
    log("  Answer:")
    log.wrapped(answer["text"], "    ")
    log()
    log(
        f"  Grounding: {gs['n_grounded_words']} of {gs['n_content_words']} content words "
        f"({gs['grounding_pct']}%) appear in the evidence."
    )
    if gs["ungrounded_words"]:
        log.wrapped(", ".join(gs["ungrounded_words"]), "    ", 66, first="  Not in evidence: ")
    log()
    log("  Closest evidence for each answer sentence (share of its content words):")
    for i, s in enumerate(answer["sentence_sources"], 1):
        source = (
            f"evidence [{s['evidence']}]"
            if s["overlap"] >= MIN_SOURCE_OVERLAP
            else "no close evidence"
        )
        log(f"    {i}. {source}, {s['overlap']:.0%}")
        log.wrapped(s["sentence"], "       ", 62)
    log()
    log(f"  Mean token probability: {gs['mean_prob']}  ({gs['mean_surprisal']} bits)")
    if gs["hallucination_spans"]:
        phrases = ", ".join(
            f'"{s["text"]}" (p={s["mean_prob"]})' for s in gs["hallucination_spans"]
        )
        log.wrapped(phrases, "    ", 66, first="  Low-confidence, not in evidence: ")


# Main loop


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run(
    prompt,
    start_url,
    probe_name,
    answer_name,
    max_pages,
    out_dir: Path,
    generate_answer=True,
    agentic=False,
    seed=0,
    shuffle_links=False,
) -> dict:
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Trace(out_dir / "trace.txt")
    log.banner("NAVIGATION + ANSWER SIMULATION")
    log(f"PROMPT       : {prompt}")
    log(f"START URL    : {start_url}")
    log(f"NAVIGATION   : {answer_name + ' (agentic)' if agentic else 'GPT-2 perplexity'}")
    log(f"ANSWER MODEL : {answer_name if generate_answer else 'disabled'}")
    log(f"SEED         : {seed}" + ("  (link order shuffled)" if shuffle_links else ""))
    log(f"TIME         : {now()}")

    probe = load_probe(probe_name)
    instruct = load_instruct(answer_name) if agentic else None
    visited, stack, findings, pages, skipped, seen_texts = [], [], [], [], [], set()
    url = canonical(start_url)
    # The budget counts pages actually read; failed fetches and duplicates don't use it.
    while url and len(pages) < max_pages and len(visited) < 3 * max_pages:
        visited.append(url)
        depth = len(pages) + 1
        log.banner(f"PAGE {depth}  {url}", "─")
        t0 = time.time()
        try:
            page_text, links = fetch(url)
            reason = "duplicate of an earlier page" if page_text in seen_texts else None
        except FetchError as e:
            reason = f"fetch failed ({e})"
        if reason:
            skipped.append({"url": url, "reason": reason})
            url = pop_unvisited(stack, visited)
            log(f"  Skipped: {reason}. Backtracking to: {url or 'nothing; stopping'}")
            continue
        seen_texts.add(page_text)
        log(f"  {len(page_text.split())} words · {len(links)} links")

        analysis = analyse_page(*probe, prompt, page_text)
        log_analysis(log, analysis)
        known = {f["sentence"] for f in findings}
        findings += [
            {**f, "source_url": url, "source_depth": depth}
            for f in analysis["findings"]
            if f["sentence"] not in known
        ]

        cands = candidate_links(links, visited, start_url)
        if agentic:
            ranked = cands[:LINK_OPTIONS]
            if shuffle_links:  # control for position effects; reproducible per seed and page
                ranked = random.Random(f"{seed}-{depth}").sample(ranked, len(ranked))
            decision = choose_link(*instruct, prompt, page_text, ranked, findings, depth)
            next_url = decision["url"]
            push_runners(stack, ranked, next_url)  # backtracking follows the order shown
            if next_url is None and not decision["stop"]:
                next_url = pop_unvisited(stack, visited)
            rationale = decision["reasoning"]
            if decision["stop"]:
                rationale += " [model: question answered]"
            elif not decision["parse_ok"]:
                rationale += " [unparseable choice → backtrack]"
            log_links(log, ranked, decision["url"])
            log("  Raw model output:")
            for line in decision["raw"][:300].splitlines():
                log(f"    | {line}")
        else:
            ranked = score_links(*probe, prompt, cands)
            push_runners(stack, ranked, ranked[0]["url"] if ranked else None)
            next_url, rationale = decide_standard(
                ranked, analysis["relevance_score"], stack, visited
            )
            decision = None
            log_links(log, ranked[:8], next_url)
        log()
        log(f"  Decision: {rationale}")
        log(f"  → {next_url or 'STOP'}   ({time.time() - t0:.1f}s)")

        pages.append(
            {
                "depth": depth,
                "url": url,
                "n_words": len(page_text.split()),
                "n_links": len(links),
                "links": [{**c, "chosen": c["url"] == next_url} for c in ranked],
                "agentic_decision": decision,
                "analysis": analysis,
                "decision": {"rationale": rationale, "next_url": next_url},
            }
        )
        url = next_url

    findings.sort(key=lambda f: -f["interest"])
    log.banner("NAVIGATION SUMMARY")
    read = [p["url"] for p in pages]
    log(f"Pages read: {len(read)}")
    for i, u in enumerate(read, 1):
        log(f"  {i}. {u}")
    for s in skipped:
        log(f"  skipped: {s['url']} ({s['reason'][:60]})")
    log()
    log("Top findings (interest = kw_count × surprisal):")
    for i, f in enumerate(findings[:10], 1):
        log(f"  [{i}] interest={f['interest']:.2f}  {f['source_url']}")
        log.wrapped(f["sentence"], "      ", 64)

    answer = None
    if generate_answer:
        if instruct is None:
            del probe
            torch.cuda.empty_cache()
            instruct = load_instruct(answer_name)
        answer = answer_question(*instruct, prompt, findings)
        log_answer(log, answer, answer_name)
    log.save()

    output = {
        "meta": {
            "prompt": prompt,
            "start_url": start_url,
            "agentic": agentic,
            "seed": seed,
            "shuffle_links": shuffle_links,
            "probe_model": probe_name,
            "answer_model": answer_name if generate_answer else None,
            "max_pages": max_pages,
            "pages_visited": len(read),
            "timestamp": now(),
        },
        "visited_urls": read,
        "skipped": skipped,
        "all_findings": findings[:20],
        "answer": answer,
        "pages": pages,
    }
    (out_dir / "trace.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


# Query refinement


def refinement_terms(findings: list[dict], exclude: set[str], k: int = 3) -> list[str]:
    """Top-k words by summed interest of the findings containing them, excluding the
    words used to select findings (every finding contains those)."""
    scores = Counter()
    for f in findings:
        for w in set(WORD_RE.findall(f["sentence"].lower())):
            if (
                len(w) >= 5
                and w not in exclude
                and not (PG_KEYWORDS.fullmatch(w) or PROSE_RE.fullmatch(w))
            ):
                scores[w] += f["interest"]
    return [w for w, _ in scores.most_common(k)]


def run_passes(prompt: str, max_passes: int, out_dir: Path, **run_kwargs) -> list[dict]:
    """Re-run with the query narrowed by surfaced terms until a pass finds nothing new."""
    terms, passes, seen_findings, seen_pages = [], [], set(), set()
    exclude = {w.lower() for w in WORD_RE.findall(prompt)} | REFINE_STOPWORDS
    for n in range(1, max_passes + 1):
        query = f"{prompt} (specifically: {', '.join(terms)})" if terms else prompt
        result = run(query, out_dir=out_dir / f"pass{n}", **run_kwargs)
        found = {f["sentence"] for f in result["all_findings"]}
        new_terms = refinement_terms(result["all_findings"], exclude | set(terms))
        answer = result["answer"]
        passes.append(
            {
                "pass": n,
                "query": query,
                "pages_visited": len(result["visited_urls"]),
                "new_pages": sorted(set(result["visited_urls"]) - seen_pages),
                "new_findings": len(found - seen_findings),
                "grounding_pct": answer["grounding_stats"]["grounding_pct"] if answer else None,
                "next_terms": new_terms,
            }
        )
        seen_findings |= found
        seen_pages |= set(result["visited_urls"])
        terms += new_terms
        torch.cuda.empty_cache()
        if not passes[-1]["new_findings"] or not new_terms:
            break
    (out_dir / "passes.json").write_text(json.dumps(passes, indent=2), encoding="utf-8")
    return passes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="what does the department offer for postgraduate students?")
    ap.add_argument("--dept", choices=SEED_URLS, default="sociology")
    ap.add_argument("--start-url", help="overrides --dept")
    ap.add_argument("--max-pages", type=int, default=8)
    ap.add_argument("--nav-model", default=PROBE_MODEL)
    ap.add_argument("--answer-model", default=ANSWER_MODEL)
    ap.add_argument("--agentic", action="store_true")
    ap.add_argument("--no-answer", action="store_true")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-links", action="store_true", help="randomise link order (agentic)")
    ap.add_argument("--out-dir", default="./run")
    ap.add_argument(
        "--dump-fetch", action="store_true", help="print the extracted start page and exit"
    )
    args = ap.parse_args()
    start = args.start_url or SEED_URLS[args.dept]

    if args.dump_fetch:
        text, links = fetch(start)
        print(text, *(f"{t[:50]:<50}  {u}" for t, u in links), sep="\n")
        return

    kwargs = dict(
        start_url=start,
        probe_name=args.nav_model,
        answer_name=args.answer_model,
        max_pages=args.max_pages,
        generate_answer=not args.no_answer,
        agentic=args.agentic,
        seed=args.seed,
        shuffle_links=args.shuffle_links,
    )
    if args.passes > 1:
        run_passes(args.prompt, args.passes, Path(args.out_dir), **kwargs)
    else:
        run(args.prompt, out_dir=Path(args.out_dir), **kwargs)


if __name__ == "__main__":
    main()
