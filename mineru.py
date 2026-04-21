import subprocess
from pathlib import Path

input_dir = Path("./pdfs")
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

pdf_files = list(input_dir.glob("*.pdf"))
print(f"Found {len(pdf_files)} PDF files")

for pdf in pdf_files:
    print(f"Processing: {pdf.name}")
    result = subprocess.run(
        ["mineru", "-p", str(pdf), "-o", str(output_dir), "-b", "pipeline"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"✅ Done: {pdf.name}")
    else:
        print(f"❌ Failed: {pdf.name} → {result.stderr}")



from mineru.cli.main import do_parse
from pathlib import Path

input_dir = Path("./pdfs")
output_dir = "./output"

pdf_files = list(input_dir.glob("*.pdf"))

pdf_names = [f.name for f in pdf_files]
pdf_bytes  = [open(f, "rb").read() for f in pdf_files]

do_parse(
    output_dir=output_dir,
    pdf_file_names=pdf_names,
    pdf_bytes_list=pdf_bytes,
    parse_method="pipeline"  # CPU-safe for OpenShift
)



from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess

input_dir = Path("./pdfs")
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

def process_pdf(pdf_path):
    result = subprocess.run(
        ["mineru", "-p", str(pdf_path), "-o", str(output_dir), "-b", "pipeline"],
        capture_output=True, text=True
    )
    return pdf_path.name, result.returncode == 0

pdf_files = list(input_dir.glob("*.pdf"))

# Run 4 files in parallel (adjust based on your pod's CPU)
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(process_pdf, f): f for f in pdf_files}
    for future in as_completed(futures):
        name, success = future.result()
        print(f"{'✅' if success else '❌'} {name}")





"""
MinerU - Full Pipeline
- Auto-detects GPU and selects best backend
- Processes multiple PDFs with live output
- Reads and parses the markdown output
- Handles errors and timeouts gracefully
"""

import subprocess
import sys
import torch
import psutil
from pathlib import Path


# ─────────────────────────────────────────
# CONFIG — change these to your paths
# ─────────────────────────────────────────
INPUT_DIR   = Path("./pdfs")       # folder with your PDF files
OUTPUT_DIR  = Path("./output")     # where MinerU saves results
TIMEOUT     = 600                  # max seconds per file (10 min)
CHUNK_SIZE  = 30                   # pages per chunk if PDF is too large


# ─────────────────────────────────────────
# STEP 1 — System Check
# ─────────────────────────────────────────
def system_check():
    print("=" * 50)
    print("🖥️  SYSTEM CHECK")
    print("=" * 50)

    gpu_available = torch.cuda.is_available()
    print(f"GPU Available : {gpu_available}")

    if gpu_available:
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        vram_free  = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
        print(f"GPU Name      : {torch.cuda.get_device_name(0)}")
        print(f"VRAM Total    : {vram_total:.1f} GB")
        print(f"VRAM Free     : {vram_free:.1f} GB")
    else:
        vram_total = 0

    ram_free = psutil.virtual_memory().available / 1024**3
    disk_free = psutil.disk_usage(".").free / 1024**3
    print(f"RAM Free      : {ram_free:.1f} GB")
    print(f"Disk Free     : {disk_free:.1f} GB")
    print("=" * 50)

    return vram_total


# ─────────────────────────────────────────
# STEP 2 — Auto-detect Best Backend
# ─────────────────────────────────────────
def get_best_backend(vram_gb):
    print("\n🔍 Selecting best backend...")

    if not torch.cuda.is_available():
        print("⚠️  No GPU found — using CPU pipeline")
        return "pipeline"

    if vram_gb >= 10:
        backend = "vlm-auto-engine"
        print(f"✅ VRAM {vram_gb:.1f}GB >= 10GB → using: {backend} (best accuracy)")
    elif vram_gb >= 8:
        backend = "hybrid-auto-engine"
        print(f"✅ VRAM {vram_gb:.1f}GB >= 8GB → using: {backend} (balanced)")
    elif vram_gb >= 6:
        backend = "pipeline"
        print(f"✅ VRAM {vram_gb:.1f}GB >= 6GB → using: {backend} (GPU pipeline)")
    else:
        backend = "pipeline"
        print(f"⚠️  VRAM {vram_gb:.1f}GB < 6GB → using: {backend} (CPU fallback)")

    return backend


# ─────────────────────────────────────────
# STEP 3 — Split Large PDFs
# ─────────────────────────────────────────
def split_pdf_if_needed(pdf_path, chunk_size=30):
    try:
        import PyPDF2
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "PyPDF2", "-q"])
        import PyPDF2

    reader   = PyPDF2.PdfReader(str(pdf_path))
    total    = len(reader.pages)
    print(f"   📄 Pages: {total}")

    if total <= chunk_size:
        return [pdf_path]  # no split needed

    print(f"   ✂️  Splitting into chunks of {chunk_size} pages...")
    chunks     = []
    chunk_dir  = pdf_path.parent / "chunks"
    chunk_dir.mkdir(exist_ok=True)

    for i in range(0, total, chunk_size):
        writer = PyPDF2.PdfWriter()
        for j in range(i, min(i + chunk_size, total)):
            writer.add_page(reader.pages[j])

        chunk_path = chunk_dir / f"{pdf_path.stem}_chunk_{i//chunk_size + 1}.pdf"
        with open(chunk_path, "wb") as f:
            writer.write(f)
        chunks.append(chunk_path)
        print(f"   → {chunk_path.name} (pages {i+1}–{min(i+chunk_size, total)})")

    return chunks


# ─────────────────────────────────────────
# STEP 4 — Process a Single PDF with Live Output
# ─────────────────────────────────────────
def process_pdf(pdf_path, output_dir, backend, timeout=600):
    print(f"\n⏳ Processing: {pdf_path.name}")

    process = subprocess.Popen(
        ["mineru", "-p", str(pdf_path), "-o", str(output_dir), "-b", backend],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    try:
        for line in process.stdout:
            print(f"   {line}", end="", flush=True)
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"\n⏰ Timeout! {pdf_path.name} exceeded {timeout}s — skipping.")
        return False

    if process.returncode == 0:
        print(f"✅ Done: {pdf_path.name}")
        return True
    else:
        print(f"❌ Failed: {pdf_path.name}")
        return False


# ─────────────────────────────────────────
# STEP 5 — Process All PDFs in Folder
# ─────────────────────────────────────────
def process_all(input_dir, output_dir, backend, timeout=600):
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDF files found in {input_dir}")
        return

    print(f"\n📂 Found {len(pdf_files)} PDF file(s) in {input_dir}")
    print("=" * 50)

    results = {"success": [], "failed": []}

    for pdf in pdf_files:
        # Split if too large
        chunks = split_pdf_if_needed(pdf, chunk_size=CHUNK_SIZE)

        for chunk in chunks:
            ok = process_pdf(chunk, output_dir, backend, timeout)
            if ok:
                results["success"].append(chunk.name)
            else:
                results["failed"].append(chunk.name)

    # Summary
    print("\n" + "=" * 50)
    print("📊 PROCESSING SUMMARY")
    print("=" * 50)
    print(f"✅ Success : {len(results['success'])} file(s)")
    print(f"❌ Failed  : {len(results['failed'])} file(s)")
    if results["failed"]:
        for f in results["failed"]:
            print(f"   - {f}")

    return results


# ─────────────────────────────────────────
# STEP 6 — Read & Parse Markdown Output
# ─────────────────────────────────────────
def read_markdown_outputs(output_dir):
    print("\n" + "=" * 50)
    print("📖 READING MARKDOWN OUTPUTS")
    print("=" * 50)

    md_files = list(output_dir.rglob("*.md"))

    if not md_files:
        print("⚠️  No markdown files found in output directory.")
        return {}

    print(f"Found {len(md_files)} markdown file(s)\n")

    parsed_results = {}

    for md_file in md_files:
        print(f"📄 File: {md_file.name}")
        print("-" * 40)

        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Basic parsing
        lines    = content.splitlines()
        headings = [l for l in lines if l.startswith("#")]
        tables   = [l for l in lines if l.startswith("|")]
        images   = [l for l in lines if "![" in l]
        formulas = [l for l in lines if "$$" in l or "$" in l]
        words    = len(content.split())
        chars    = len(content)

        print(f"   📝 Words       : {words}")
        print(f"   🔤 Characters  : {chars}")
        print(f"   📌 Headings    : {len(headings)}")
        print(f"   📊 Table rows  : {len(tables)}")
        print(f"   🖼️  Images      : {len(images)}")
        print(f"   🔢 Formulas    : {len(formulas)}")

        if headings:
            print(f"\n   📌 Headings found:")
            for h in headings[:10]:  # show first 10 headings
                print(f"      {h}")
            if len(headings) > 10:
                print(f"      ... and {len(headings) - 10} more")

        print(f"\n   📃 Preview (first 500 chars):")
        print("   " + "-" * 36)
        preview = content[:500].replace("\n", "\n   ")
        print(f"   {preview}")
        print("   " + "-" * 36)

        parsed_results[md_file.name] = {
            "path"     : str(md_file),
            "content"  : content,
            "words"    : words,
            "headings" : headings,
            "tables"   : tables,
            "images"   : images,
            "formulas" : formulas,
        }

        print()

    return parsed_results


# ─────────────────────────────────────────
# STEP 7 — Search Inside Markdown Content
# ─────────────────────────────────────────
def search_in_results(parsed_results, keyword):
    print(f"\n🔎 Searching for: '{keyword}'")
    print("=" * 50)

    found = False
    for filename, data in parsed_results.items():
        content = data["content"]
        if keyword.lower() in content.lower():
            found = True
            print(f"✅ Found in: {filename}")
            # Show context around the keyword
            idx = content.lower().find(keyword.lower())
            start = max(0, idx - 100)
            end   = min(len(content), idx + 200)
            print(f"   ...{content[start:end]}...")
            print()

    if not found:
        print(f"❌ '{keyword}' not found in any output file.")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    # Step 1 — System check
    vram = system_check()

    # Step 2 — Auto pick backend
    backend = get_best_backend(vram)

    # Step 3 & 4 & 5 — Process all PDFs
    process_all(INPUT_DIR, OUTPUT_DIR, backend, timeout=TIMEOUT)

    # Step 6 — Read and parse all markdown outputs
    parsed = read_markdown_outputs(OUTPUT_DIR)

    # Step 7 — (Optional) Search for a keyword across all outputs
    # search_in_results(parsed, "introduction")

    print("\n🎉 Pipeline complete!")
    print(f"📁 Results saved in: {OUTPUT_DIR.resolve()}")












    # In your notebook — no API server needed
import subprocess
from pathlib import Path

INPUT_DIR  = Path("./pdfs")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

for pdf in INPUT_DIR.glob("*.pdf"):
    print(f"⏳ Processing: {pdf.name}")
    result = subprocess.Popen(
        ["mineru", "-p", str(pdf), "-o", str(OUTPUT_DIR), "-b", "pipeline"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    for line in result.stdout:
        print(line, end="", flush=True)
    result.wait()
    print(f"✅ Done: {pdf.name}" if result.returncode == 0 else f"❌ Failed: {pdf.name}")