import Rail from "./Rail";

export default function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="app">
      <Rail />
      <div className="main">{children}</div>
    </div>
  );
}
