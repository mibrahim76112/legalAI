/**
 * The vLLM endpoint serving the contract-review LoRA.
 *
 * Two things this must never get wrong:
 * 1. Call the LoRA module (e.g. "contract"), not the base model. The endpoint
 *    serves both, and the base returns fluent untuned output that looks fine.
 * 2. Pin temperature 0. The served generation_config defaults to 0.6, so
 *    without this every run differs and none match how the model was evaluated.
 */

const MAX_NEW_TOKENS = 1024;
export const MAX_CONTEXT = 16384;
export const PROMPT_BUDGET = MAX_CONTEXT - MAX_NEW_TOKENS;

function config() {
  const base = (process.env.LEGALAI_ENDPOINT || "").replace(/\/$/, "");
  const token = process.env.HF_TOKEN || "";
  if (!base || !token) {
    throw new Error("Model endpoint not configured: set LEGALAI_ENDPOINT and HF_TOKEN.");
  }
  return { base: base.endsWith("/v1") ? base : `${base}/v1`, token };
}

async function call<T>(path: string, body?: unknown, root = false): Promise<T> {
  const { base, token } = config();
  const url = root ? base.replace(/\/v1$/, "") : base;
  let r: Response;
  try {
    r = await fetch(url + path, {
      method: body === undefined ? "GET" : "POST",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (e) {
    throw new Error(`Cannot reach the model endpoint: ${(e as Error).message}`);
  }
  if (r.status === 503 || r.status === 502) {
    throw new Error("The endpoint is waking up (it scales to zero). Try again in a minute.");
  }
  if (!r.ok) {
    throw new Error(`Endpoint ${r.status}: ${(await r.text()).slice(0, 300)}`);
  }
  return (await r.json()) as T;
}

let modelName: Promise<string> | null = null;

/** The LoRA module's name; the base model repo id is the other entry. */
export function loraModel(): Promise<string> {
  if (!modelName) {
    modelName = call<{ data: { id: string }[] }>("/models").then(({ data }) => {
      const ids = data.map((m) => m.id);
      const lora = ids.find((i) => !i.includes("/"));
      if (!lora) {
        throw new Error(`No LoRA adapter is served here, only ${ids.join(", ")}. ` +
          "Start vLLM with --enable-lora --lora-modules <name>=<repo>.");
      }
      return lora;
    }).catch((e) => { modelName = null; throw e; });
  }
  return modelName;
}

export interface Msg { role: "system" | "user"; content: string }

export async function ask(messages: Msg[]): Promise<string> {
  const model = await loraModel();
  const r = await call<{ choices: { message: { content: string } }[] }>("/chat/completions", {
    model, messages,
    temperature: 0, top_p: 1, max_tokens: MAX_NEW_TOKENS,
    chat_template_kwargs: { enable_thinking: false },  // Qwen3 only; Llama ignores it
  });
  return r.choices[0]?.message?.content ?? "";
}

/** Token count from the server's own tokenizer, so windowing matches the model.
 *  vLLM serves /tokenize at the root, not under /v1. */
export async function countTokens(text: string): Promise<number | null> {
  try {
    const model = await loraModel();
    const r = await call<{ count: number }>("/tokenize", { model, prompt: text }, true);
    return typeof r.count === "number" ? r.count : null;
  } catch {
    return null;   // caller falls back to the character estimate
  }
}

export async function health(): Promise<{ model: string }> {
  return { model: await loraModel() };
}
