"use client";

import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api } from "@/lib/api";

export default function Login() {
  return <Suspense><Form /></Suspense>;
}

function Form() {
  const router = useRouter();
  const next = useSearchParams().get("next") || "/reviews";
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true); setErr(null);
    try {
      await api.login(password);
      router.replace(next);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  return (
    <div className="login">
      <form className="card" onSubmit={submit}>
        <h1>Contract Review</h1>
        <p className="muted">This demo runs a GPU endpoint, so it asks for a password.</p>
        <input type="password" value={password} autoFocus placeholder="Password"
               onChange={(e) => setPassword(e.target.value)} />
        <button className="btn pri big" disabled={busy || !password}>
          {busy ? "Checking…" : "Enter"}
        </button>
        {err && <p className="err">{err}</p>}
      </form>
    </div>
  );
}
