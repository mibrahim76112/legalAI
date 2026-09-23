/**
 * Uploaded file -> plain contract text. Port of inference/extract.py.
 *
 * The line-unwrapping matters: PDFs keep visual line breaks mid-sentence, and
 * the model was trained on paragraph text. Without this it quotes one fragment
 * per line, and on the test NDA it got a verdict wrong until the wraps were
 * rejoined.
 */

const ENDS_SENTENCE = /[.;:!?]["'”’)]*$/;
const SECTION_START = /^(\d+(\.\d+)*\.?|§\s*\d+|[A-Z]\.|\([a-z0-9]{1,4}\))\s/;
const CONTINUES = /^[,;:)a-z]/;

export function unwrap(text: string): string {
  const out: string[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    const prev = out.length ? out[out.length - 1] : null;
    if (prev && line) {
      const heading = prev.length < 60 && (SECTION_START.test(prev) || prev === prev.toUpperCase());
      const joins = CONTINUES.test(line) ||
        !(ENDS_SENTENCE.test(prev) || heading || SECTION_START.test(line));
      if (joins) {
        out[out.length - 1] = prev + (",;:)".includes(line[0]) ? "" : " ") + line;
        continue;
      }
    }
    out.push(line);
  }
  return out.join("\n");
}

async function fromPdf(data: Uint8Array): Promise<string> {
  const { extractText, getDocumentProxy } = await import("unpdf");
  const pdf = await getDocumentProxy(data);
  const { text } = await extractText(pdf, { mergePages: false });
  const pages = (Array.isArray(text) ? text : [text]).map((p) => p.replace(/[ \t]+/g, " "));
  return unwrap(pages.join("\n"));
}

async function fromDocx(data: Uint8Array): Promise<string> {
  const mammoth = (await import("mammoth")).default ?? (await import("mammoth"));
  const { value } = await mammoth.extractRawText({ buffer: Buffer.from(data) });
  return value;
}

export async function extractText(filename: string, data: Uint8Array): Promise<string> {
  const name = filename.toLowerCase();
  let text: string;
  if (name.endsWith(".pdf")) text = await fromPdf(data);
  else if (name.endsWith(".docx")) text = await fromDocx(data);
  else if (name.endsWith(".txt")) text = new TextDecoder().decode(data);
  else throw new Error("Unsupported file type; use PDF, DOCX or TXT");

  text = text.replace(/\r\n?/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  if (!text) throw new Error("No text found in the file (a scanned PDF needs OCR first)");
  return text;
}
