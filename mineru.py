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





import subprocess
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Auto-detect best backend
def get_best_backend():
    if not torch.cuda.is_available():
        return "pipeline"  # CPU fallback
    
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")
    
    if vram_gb >= 10:
        return "vlm-auto-engine"
    elif vram_gb >= 8:
        return "hybrid-auto-engine"
    elif vram_gb >= 6:
        return "pipeline"
    else:
        return "pipeline"  # CPU fallback for low VRAM

backend = get_best_backend()
print(f"Using backend: {backend}")

# Process files
input_dir = Path("./pdfs")
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

pdf_files = list(input_dir.glob("*.pdf"))
print(f"Found {len(pdf_files)} PDF files")

for pdf in pdf_files:
    print(f"\n⏳ Processing: {pdf.name}")
    result = subprocess.run(
        ["mineru", "-p", str(pdf), "-o", str(output_dir), "-b", backend],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"✅ Done: {pdf.name}")
    else:
        print(f"❌ Failed: {pdf.name}")
        print(result.stderr)