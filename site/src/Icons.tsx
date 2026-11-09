// Small inline icons (no dependency). Drawn with currentColor so they follow the theme.
const base = { width: 26, height: 26, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.7, strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true } as const;

export const IconFit = () => (<svg {...base}><path d="M3 20h18" /><path d="M5 16l4-5 4 3 6-8" /><circle cx="9" cy="11" r="1.3" /><circle cx="13" cy="14" r="1.3" /><circle cx="19" cy="6" r="1.3" /></svg>);
export const IconError = () => (<svg {...base}><path d="M3 9h18v6H3z" /><path d="M7 9v3M11 9v2M15 9v3M19 9v2" /></svg>);
export const IconTop = () => (<svg {...base}><path d="M3 5h18l-7 8v6l-4-2v-4z" /></svg>);

// Decorative background for the hero: a rising curve and scattered customer dots.
export function HeroArt() {
  const dots = [[60, 210], [95, 180], [140, 195], [185, 150], [225, 160], [265, 110], [310, 120], [350, 80], [395, 70], [430, 50], [120, 225], [205, 205], [290, 175], [375, 125]];
  return (
    <svg className="heroArt" viewBox="0 0 480 260" aria-hidden="true" preserveAspectRatio="xMaxYMid meet">
      <path d="M10 250 C 80 150, 170 70, 470 20" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
      <path d="M10 250 L 470 20" fill="none" stroke="currentColor" strokeWidth="1" strokeDasharray="4 6" opacity=".6" />
      {dots.map(([x, y], i) => <circle key={i} cx={x} cy={y} r={i % 4 === 0 ? 5 : 3.2} fill="currentColor" opacity={i % 4 === 0 ? 0.9 : 0.5} />)}
    </svg>
  );
}

export const IconProtect = () => (<svg {...base}><path d="M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6z" /><path d="M9 12l2 2 4-4" /></svg>);
export const IconWinBack = () => (<svg {...base}><path d="M4 12a8 8 0 1 0 3-6.2" /><path d="M4 4v4h4" /></svg>);
export const IconGrow = () => (<svg {...base}><path d="M12 21v-9" /><path d="M12 12C12 8 9 6 5 6c0 4 2 6 7 6z" /><path d="M12 14c0-3 2-5 6-5 0 3-2 5-6 5z" /></svg>);
export const IconLow = () => (<svg {...base}><circle cx="12" cy="12" r="9" /><path d="M8 12h8" /></svg>);
