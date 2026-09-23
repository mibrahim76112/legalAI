export default function Stepper({ steps, at }: { steps: string[]; at: number }) {
  return (
    <ol className="stepper">
      {steps.map((s, i) => (
        <li key={s} className={i < at ? "done" : i === at ? "now" : ""}>
          <span className="n">{i < at ? "✓" : i + 1}</span>
          <span className="l">{s}</span>
        </li>
      ))}
    </ol>
  );
}
