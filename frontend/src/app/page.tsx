"use client";

import { useEffect, useState } from "react";

type D = Record<string, any>;

export default function Home() {
  const [data, setData] = useState<D | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      const r = await fetch("/api/pipeline/refresh");
      setData(await r.json());
    } catch {}
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (!data)
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#05080f]">
        <p className="text-white/40 text-sm">
          Connecting to Quantinel pipeline...
        </p>
      </div>
    );

  const d = data.decision;
  const r = data.risk;
  const intel = data.intelligence;
  const branch = data.active_branch === "quantum" ? data.quantum : data.normal;
  const lastRec = branch.records[branch.records.length - 1];
  const eq = branch.scorecard.equity_curve.values;
  const n = data.normal.scorecard;
  const q = data.quantum.scorecard;

  const pct = (v: any) => (v == null ? "\u2014" : (v * 100).toFixed(1) + "%");
  const fm = (v: any) => (v == null ? "\u2014" : Number(v).toFixed(2));

  const tickers: string[] = data.tickers;
  const models: string[] = r.models || [];
  const logs: string[] = data.execution_logs || [];
  const themes: string[] = intel.themes || [];

  return (
    <div className="min-h-screen bg-[#05080f] text-[#e6ecf5] font-sans">
      <header className="sticky top-0 z-50 border-b border-white/5 bg-[#05080f]/80 backdrop-blur-xl">
        <div className="mx-auto flex h-14 max-w-[1600px] items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <span className="text-lg font-bold tracking-tight">
              <span className="text-blue-400">&#x25C8;</span> QUANTINEL
            </span>
            <span className="hidden text-xs text-white/30 sm:inline">
              Agentic Quantum Risk
            </span>
          </div>
          <button
            onClick={fetchData}
            disabled={loading}
            className="rounded-lg bg-blue-600 px-4 py-2 text-xs font-semibold tracking-wider hover:bg-blue-500 disabled:opacity-50"
          >
            {loading ? "RUNNING..." : "EXECUTE RUN"}
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-[1600px] space-y-4 p-6">
        {/* Decision + Intelligence */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Card title="Agent Decision Center">
            <div className="mb-4 flex items-center justify-between rounded-xl border border-white/5 bg-white/[0.02] p-4">
              <div>
                <Label>Winner</Label>
                <div className="mt-1 text-2xl font-bold">{d.winner}</div>
              </div>
              <div className="text-right">
                <Label>Confidence</Label>
                <div className="mt-1 text-lg font-mono text-blue-400">
                  {d.confidence}
                </div>
              </div>
            </div>
            <p className="mb-4 text-sm leading-relaxed text-white/60">
              {d.rationale}
            </p>
            <div className="flex flex-wrap gap-2">
              {tickers.map((t) => {
                const s = d.stances?.[t] || "FLAT";
                const w = lastRec?.weights?.[t] || 0;
                const cls =
                  s === "LONG"
                    ? "border-emerald-400/20 bg-emerald-400/10 text-emerald-400"
                    : s === "SHORT"
                      ? "border-red-400/20 bg-red-400/10 text-red-400"
                      : "border-white/10 bg-white/5 text-white/30";
                return (
                  <div
                    key={t}
                    className={
                      "flex items-center justify-between rounded-lg border px-3 py-2 " +
                      cls
                    }
                  >
                    <span className="text-xs font-semibold">{t}</span>
                    <span className="text-xs font-mono">
                      {(w * 100).toFixed(1)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </Card>

          <Card title="Intelligence Feed">
            <div className="mb-3 flex flex-wrap gap-2">
              {themes.map((t, i) => (
                <span
                  key={i}
                  className="rounded-full bg-amber-400/10 px-2.5 py-0.5 text-[11px] font-semibold text-amber-400"
                >
                  {t.toUpperCase()}
                </span>
              ))}
            </div>
            {tickers.map((t) => {
              const heads = intel.headlines?.[t] || [];
              const sent = intel.sentiment?.[t] || 0;
              const c =
                sent > 0.05
                  ? "text-emerald-400"
                  : sent < -0.05
                    ? "text-red-400"
                    : "text-white/50";
              if (!heads[0]) return null;
              return (
                <div
                  key={t}
                  className="mb-2 flex items-start gap-3 rounded-lg border border-white/5 bg-white/[0.02] px-3 py-2.5"
                >
                  <span className="text-[11px] font-bold text-white/40 w-14 shrink-0">
                    {t}
                  </span>
                  <span className="flex-1 text-xs text-white/70 line-clamp-2">
                    {heads[0]}
                  </span>
                  <span className={"text-xs font-mono shrink-0 " + c}>
                    {sent > 0 ? "+" : ""}
                    {sent.toFixed(2)}
                  </span>
                </div>
              );
            })}
          </Card>
        </div>

        {/* Risk */}
        <Card title="Risk Intelligence">
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <div>
              <div className="mb-2 flex justify-between border-b border-white/5 pb-2">
                <Label>Tail Risk 95%</Label>
                <span className="text-[11px] text-amber-400">
                  Disagreement:{" "}
                  <span className="font-mono text-white/70">
                    {fm(r.disagreement, 3)}
                  </span>
                </span>
              </div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-white/30">
                    <th className="pb-2 text-left">TICKER</th>
                    <th className="pb-2 text-right">VaR</th>
                    <th className="pb-2 text-right">CVaR</th>
                  </tr>
                </thead>
                <tbody>
                  {tickers.map((t) => (
                    <tr key={t} className="border-b border-white/[0.03]">
                      <td className="py-2 font-semibold">{t}</td>
                      <td className="py-2 text-right font-mono text-red-400">
                        {pct(r.var_95?.[t])}
                      </td>
                      <td className="py-2 text-right font-mono text-red-400">
                        {pct(r.cvar_95?.[t])}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div>
              <div className="mb-2 border-b border-white/5 pb-2">
                <Label>Calibration</Label>
              </div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-white/30">
                    <th className="pb-2 text-left">MODEL</th>
                    <th className="pb-2 text-right">BREACH</th>
                    <th className="pb-2 text-right">STATUS</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((m) => {
                    const br = r.breach_rates?.[m] || 0;
                    const pass = br <= 0.08;
                    return (
                      <tr key={m} className="border-b border-white/[0.03]">
                        <td className="py-2 font-semibold">
                          {m.toUpperCase()}
                        </td>
                        <td
                          className={
                            "py-2 text-right font-mono " +
                            (pass ? "text-emerald-400" : "text-red-400")
                          }
                        >
                          {(br * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 text-right">
                          <span
                            className={
                              "rounded-full px-2 py-0.5 text-[10px] font-semibold " +
                              (pass
                                ? "bg-emerald-400/10 text-emerald-400"
                                : "bg-red-400/10 text-red-400")
                            }
                          >
                            {pass ? "PASS" : "FAIL"}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </Card>

        {/* Comparison + Equity */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Card title="Normal vs Quantum Simulation">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-white/30">
                  <th className="pb-2 text-left">METRIC</th>
                  <th className="pb-2 text-right">NORMAL</th>
                  <th className="pb-2 text-right">QUANTUM</th>
                </tr>
              </thead>
              <tbody>
                <Row a="Sharpe" b={fm(n.sharpe)} c={fm(q.sharpe)} />
                <Row
                  a="Return"
                  b={pct(n.total_return)}
                  c={pct(q.total_return)}
                />
                <Row
                  a="Dir Acc"
                  b={(n.directional_accuracy * 100).toFixed(1) + "%"}
                  c={(q.directional_accuracy * 100).toFixed(1) + "%"}
                />
              </tbody>
            </table>
            <div className="mt-3 text-xs text-white/30">
              Active:{" "}
              <span className="font-semibold text-blue-400">
                {data.active_branch?.toUpperCase()}
              </span>
            </div>
          </Card>

          <Card>
            <div className="mb-3 flex gap-4 text-xs">
              <span className="font-semibold text-emerald-400">
                Sharpe: {fm(branch.scorecard.sharpe)}
              </span>
              <span className="font-semibold text-emerald-400">
                Return: {pct(branch.scorecard.total_return)}
              </span>
            </div>
            <div className="h-[140px] rounded-lg border border-white/5 p-3">
              <svg
                width="100%"
                height="100%"
                viewBox="0 0 400 116"
                preserveAspectRatio="none"
              >
                <path
                  d={(() => {
                    const v = eq;
                    const mn = Math.min(...v);
                    const mx = Math.max(...v);
                    const rng = mx - mn || 1;
                    return v
                      .map(
                        (val: number, i: number) =>
                          (i === 0 ? "M" : "L") +
                          ((i / (v.length - 1)) * 400).toFixed(1) +
                          "," +
                          (116 - ((val - mn) / rng) * 116).toFixed(1),
                      )
                      .join(" ");
                  })()}
                  fill="none"
                  stroke="#38bdf8"
                  strokeWidth="2"
                />
              </svg>
            </div>
          </Card>
        </div>

        {/* Logs */}
        <Card title="Pipeline Logs">
          <div className="max-h-[200px] space-y-0.5 overflow-auto rounded-lg border border-white/5 bg-white/[0.02] p-3 font-mono text-[11px] text-white/40">
            {logs.map((l, i) => (
              <div key={i}>{l}</div>
            ))}
          </div>
        </Card>
      </main>
    </div>
  );
}

function Card({
  title,
  children,
}: {
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-white/5 bg-white/[0.02] p-5">
      {title && (
        <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.15em] text-white/30">
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <div className="text-[10px] uppercase text-white/30">{children}</div>;
}

function Row({ a, b, c }: { a: string; b: string; c: string }) {
  return (
    <tr className="border-b border-white/[0.03]">
      <td className="py-2.5 text-white/50">{a}</td>
      <td className="py-2.5 text-right font-mono">{b}</td>
      <td className="py-2.5 text-right font-mono">{c}</td>
    </tr>
  );
}
