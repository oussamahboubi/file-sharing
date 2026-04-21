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