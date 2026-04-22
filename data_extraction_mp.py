import os, re, json, time, pandas as pd
from datetime import datetime
from collections import OrderedDict

from dotenv import load_dotenv
load_dotenv()

# ───  INSTALL DEPENDENCIES FIRST  ────────────────────────────
# pip install -U transformers kernels torch

# ───  CONFIG  ────────────────────────────────────────────────
INPUT_DIR    = "input_dir/txt"
JSON_OUT     = "results/marches_output.json"
XLSX_OUT     = "results/marches_output.xlsx"

MODEL_ID     = "openai/gpt-oss-20b"

# Reasoning level injected into the system prompt: "low" | "medium" | "high"
# - low    → fast, general dialogue
# - medium → balanced (recommended for extraction tasks)
# - high   → deep analysis, slower
REASONING_LEVEL = os.environ.get("REASONING_LEVEL", "medium")

MAX_NEW_TOKENS  = 4096   # extraction output can be long for many competitors

# ───  PROMPT TEMPLATE  ────────────────────────────────────────
SYSTEM_PROMPT = f"""You are an expert assistant for extracting structured JSON data from French public procurement reports.
Reasoning: {REASONING_LEVEL}"""

USER_TEMPLATE = r"""Your task is to extract **only** the fields listed below from the document. Do not add, rename, or remove any field. Do not explain anything. Respect the field order exactly.

Return only a valid JSON object with this exact structure:

{{
  "Date de publication au portail des marchés publics": "string — extract only the date mentioned after phrases like 'Apparu au portail', 'Sites électroniques de publication de l'avis', or 'www.marchespublics.gov.ma du'. Do not include the full sentence, only the date. If not found, return null",
  "Date d'ouverture des plis": "string — always extract it exactly as written in the document",
  "Date d'achèvement des travaux de la commission": "string — extract it from the bottom of the page or final section of the document",
  "Journaux de publications": "string — not a list. Combine all journal entries in one string, separate them with ' | ' (pipe). If not found, return null. Do not include URL",
  "Référence du marché": "string — extract from lines after: 'APPEL D'OFFRES OUVERT', 'offres de prix', 'L'APPEL D'OFFRES OUVERT NATIONAL', or 'AO n°', such as 'N° 36/2024', 'n° 09/CNDP/2024', or 'N° 147-24-AOO'",
  "Objet de marché": "string — extract from the line starting with 'Objet :' or 'OBJET :' or 'Objet de l'appel d'offre :'",
  "Maître d'ouvrage": "string — extract from lines like 'Maître d'ouvrage' or 'Maitre d'ouvrage Délégué'",
  "Liste des concurrents": [
    {{
      "Noms-Raisons sociales": "string — full name as written in the document",
      "Montant TTC": "string — full amount as written in the document. If not found, return null",
      "Admissible": true or false,
      "Retenu": true or false
    }}
  ],
  "Le marché est infructueux": true or false — return true only if the document explicitly states that the tender was unsuccessful
}}

Instructions:
- Only include the fields defined above, and always respect the order.
- Return a **valid JSON object** — do not add explanations, comments, or extra text.
- All competitors mentioned in the document must be included — even if they were not admitted or retained.
- If a competitor is not mentioned in the admitted/retained section, default:
  + "Montant TTC": null
  + "Admissible": false
  + "Retenu": false
- Do not convert numbers. Keep amounts as strings.
- If any field is missing, return it as null.
- Dates must be returned exactly as they appear in the document.
- "Journaux de publications" should be a single string with all journal references joined using ' | '.
- Do not include other fields like 'Concurrent retenu', 'Attributaire', or 'Justification du choix'.

Here is the text:
\"\"\"{text_content}\"\"\""""

FIELD_ORDER = [
    "Date de publication au portail des marchés publics",
    "Date d'ouverture des plis",
    "Date d'achèvement des travaux de la commission",
    "Journaux de publications",
    "Référence du marché",
    "Objet de marché",
    "Maître d'ouvrage",
    "Liste des concurrents",
    "Le marché est infructueux",
]

# ───  LOGGING  ───────────────────────────────────────────────
def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

# ───  LOAD MODEL (once, at startup)  ─────────────────────────
def load_pipeline():
    """
    Loads the gpt-oss-20b pipeline using the official Transformers API.
    torch_dtype='auto' will pick bf16 on capable hardware.
    device_map='auto' spreads the model across available GPUs (or CPU).
    """
    from transformers import pipeline
    import torch

    log(f"Loading model: {MODEL_ID}  (this may take a few minutes the first time)…")
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        torch_dtype="auto",   # bf16 on GPU, fp32 on CPU
        device_map="auto",    # multi-GPU or CPU fallback
    )
    log("✅ Model loaded.")
    return pipe

# ───  INFERENCE  ─────────────────────────────────────────────
def query_model(pipe, text_content: str) -> str | None:
    """
    Builds the harmony-format message list and runs inference.
    gpt-oss-20b MUST use the harmony response format — the Transformers
    chat template handles this automatically when you pass a messages list.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.replace("{text_content}", text_content)},
    ]

    try:
        outputs = pipe(
            messages,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,        # greedy — deterministic extraction
            temperature=None,       # must be None when do_sample=False
            top_p=None,
        )
        # The pipeline returns the full conversation; the last message is the reply
        return outputs[0]["generated_text"][-1]["content"]
    except Exception as e:
        log(f"❌ Inference error: {e}")
        return None

# ───  JSON CLEANER / PARSER  ─────────────────────────────────
def parse_json(txt: str | None):
    if not txt:
        return None

    txt = txt.strip()
    # Strip ```json ... ``` fences if present
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$",          "", txt)

    # The model may include chain-of-thought before the JSON block;
    # grab only the first {...} blob.
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None

    try:
        return json.loads(m.group(0))
    except Exception as e:
        log(f"⚠️  JSON parse error: {e}")
        return None

def reorder(d: dict) -> OrderedDict:
    return OrderedDict((k, d.get(k)) for k in FIELD_ORDER)

# ───  EXCEL FLATTENERS  ──────────────────────────────────────
def flatten_for_excel(marches: list[dict]) -> pd.DataFrame:
    rows = []
    for m in marches:
        base = {k: m.get(k) for k in FIELD_ORDER if k != "Liste des concurrents"}
        for c in m.get("Liste des concurrents", []) or [{}]:
            rows.append(base | {
                "Noms-Raisons sociales": c.get("Noms-Raisons sociales"),
                "Montant TTC":           c.get("Montant TTC"),
                "Admissible":            c.get("Admissible"),
                "Retenu":                c.get("Retenu"),
            })
    return pd.DataFrame(rows)

def flatten_retenu_for_excel(marches: list[dict]) -> pd.DataFrame:
    rows = []
    for m in marches:
        base = {k: m.get(k) for k in FIELD_ORDER if k != "Liste des concurrents"}
        for c in m.get("Liste des concurrents", []) or [{}]:
            if c.get("Retenu"):
                rows.append(base | {
                    "Noms-Raisons sociales": c.get("Noms-Raisons sociales"),
                    "Montant TTC":           c.get("Montant TTC"),
                    "Admissible":            c.get("Admissible"),
                    "Retenu":                c.get("Retenu"),
                })
    return pd.DataFrame(rows)

# ───  PROCESS ALL FILES  ─────────────────────────────────────
def run_all(pipe, folder: str) -> list:
    """
    Single-process loop — Transformers already parallelises internally
    across GPUs via device_map='auto', so multiprocessing is not needed.
    """
    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt", ".md")],
        key=os.path.getsize,
    )
    log(f"🚀 Processing {len(files)} file(s) with {MODEL_ID}")

    results = []
    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        log(f"⚙️  [{i}/{len(files)}] {name}")

        try:
            with open(path, encoding="utf-8") as f:
                txt = f.read()

            raw = query_model(pipe, txt)
            res = parse_json(raw)

            if res is None:
                log(f"⚠️  {name}: could not parse JSON — skipping")
                continue

            res.setdefault("Le marché est infructueux", False)
            results.append(reorder(res))

        except Exception as e:
            log(f"❌ {name}: {e}")

    return results

# ───  EXCEL WRITER  ──────────────────────────────────────────
def build_excel(records: list[dict]):
    clean = [r for r in records if isinstance(r, dict)]

    df_main   = pd.DataFrame([{k: v for k, v in r.items() if k != "Liste des concurrents"} for r in clean])
    df_flat   = flatten_for_excel(clean)
    df_retenu = flatten_retenu_for_excel(clean)

    with pd.ExcelWriter(XLSX_OUT, engine="xlsxwriter") as w:
        df_main.to_excel(w,   sheet_name="Marches",   index=False)
        df_flat.to_excel(w,   sheet_name="Flattened", index=False)
        df_retenu.to_excel(w, sheet_name="Retenu",    index=False)

    log(f"📊 Excel saved → {XLSX_OUT}")

# ───  MAIN  ──────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(JSON_OUT),  exist_ok=True)
    os.makedirs(os.path.dirname(XLSX_OUT),  exist_ok=True)

    pipe    = load_pipeline()
    records = run_all(pipe, INPUT_DIR)

    if not records:
        log("Nothing extracted.")
        return

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump({"Liste des marchés publics": records}, f, indent=2, ensure_ascii=False)
    log(f"📝 JSON → {JSON_OUT}")

    build_excel(records)
    log(f"✅ Done: {len(records)} marchés extracted.")

if __name__ == "__main__":
    main()