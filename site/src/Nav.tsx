import { useEffect, useState } from "react";

type Item = readonly [string, string];

function Logo() {
  return (
    <svg className="logo" width="26" height="26" viewBox="0 0 24 24" aria-hidden="true" fill="none">
      <rect x="1" y="1" width="22" height="22" rx="6" fill="currentColor" />
      <path d="M5 17c3-1 4-6 7-7s4-3 7-5" stroke="var(--bg)" strokeWidth="2" strokeLinecap="round" />
      <circle cx="19" cy="5" r="1.8" fill="var(--hi)" />
    </svg>
  );
}

export default function Header({ sections, active }: { sections: readonly Item[]; active: string }) {
  const [open, setOpen] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let raf = 0;
    const update = () => {
      raf = 0;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      setProgress(max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0);
    };
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(update); };
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => { window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); if (raf) cancelAnimationFrame(raf); };
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <header className={open ? "open" : ""}>
      <a className="brand" href="#about" onClick={() => setOpen(false)}><Logo /><strong>Customer spend forecast</strong></a>
      <button className="burger" aria-expanded={open} aria-controls="site-nav" aria-label={open ? "Close menu" : "Open menu"} onClick={() => setOpen((o) => !o)}>
        <span /><span /><span />
      </button>
      <nav id="site-nav" aria-label="Sections">
        {sections.map(([id, l]) => (
          <a key={id} href={`#${id}`} className={active === id ? "cur" : ""} aria-current={active === id ? "true" : undefined} onClick={() => setOpen(false)}>{l}</a>
        ))}
      </nav>
      <div className="progress" role="progressbar" aria-label="Reading progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress * 100)}>
        <i style={{ transform: `scaleX(${progress})` }} />
      </div>
    </header>
  );
}
