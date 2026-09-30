import { IconContext } from '@phosphor-icons/react'
import clsx from 'clsx'
import { Suspense, useEffect, useState, type ReactNode } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'

type Theme = 'system' | 'light' | 'dark'

function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    const t = document.documentElement.dataset.theme
    return t === 'light' || t === 'dark' ? t : 'system'
  })
  useEffect(() => {
    if (theme === 'system') delete document.documentElement.dataset.theme
    else document.documentElement.dataset.theme = theme
    try {
      if (theme === 'system') localStorage.removeItem('netrikan-theme')
      else localStorage.setItem('netrikan-theme', theme)
    } catch {
      /* storage unavailable */
    }
  }, [theme])
  return [theme, setTheme] as const
}

/** Scroll to the top on route change. Braces matter: scrollTo() returns a Promise in current Chrome, and an effect must not return one. */
function useScrollTopOnNavigate() {
  const { pathname } = useLocation()
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [pathname])
}

function Mark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <rect width="32" height="32" rx="8" style={{ fill: 'var(--ink)' }} />
      <path d="M5 16c3-6 7-9 11-9s8 3 11 9c-3 6-7 9-11 9S8 22 5 16z" fill="none" style={{ stroke: 'var(--page)' }} strokeWidth="2.2" />
      <circle cx="16" cy="16" r="4" style={{ fill: 'var(--accent)' }} />
    </svg>
  )
}

const TOP = [
  { to: '/', label: 'Overview', end: true },
  { to: '/dashboard', label: 'Dashboard', end: false },
]

function TopBar() {
  const [theme, setTheme] = useTheme()
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-page/85 backdrop-blur-md">
      <div className="mx-auto flex h-15 max-w-[1400px] items-center gap-6 px-4 sm:px-6 lg:px-10">
        <Link to="/" className="flex shrink-0 items-center gap-2.5 rounded-md focus-visible:outline-2 focus-visible:outline-accent">
          <Mark className="size-7" />
          <span className="text-[15px] font-semibold tracking-[-0.01em]">Netrikan</span>
        </Link>
        <nav aria-label="Primary" className="flex items-center gap-1">
          {TOP.map((t) => (
            <NavLink
              key={t.to}
              to={t.to}
              end={t.end}
              className={({ isActive }) =>
                clsx(
                  'rounded-md px-3 py-1.5 text-[13.5px] font-medium transition-colors duration-200 focus-visible:outline-2 focus-visible:outline-accent',
                  isActive ? 'bg-surface-2 text-ink' : 'text-ink-2 hover:text-ink',
                )
              }
            >
              {t.label}
            </NavLink>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-3">
          <label className="hidden items-center gap-2 text-[12.5px] text-muted sm:flex">
            Theme
            <select value={theme} onChange={(e) => setTheme(e.target.value as Theme)}
              className="h-8 rounded-md border border-line bg-surface pr-7 pl-2 text-[12.5px] text-ink-2 outline-none hover:border-line-strong focus-visible:ring-2 focus-visible:ring-accent">
              <option value="system">System</option>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </label>
          <a href="/api/docs" target="_blank" rel="noreferrer" className="text-[12.5px] text-ink-2 underline-offset-4 hover:text-ink hover:underline">
            API docs
          </a>
        </div>
      </div>
    </header>
  )
}

export function SiteLayout({ children }: { children: ReactNode }) {
  useScrollTopOnNavigate()
  return (
    <IconContext.Provider value={{ weight: 'regular' }}>
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-surface focus:px-3 focus:py-2">
        Skip to content
      </a>
      <TopBar />
      <main id="main">{children}</main>
    </IconContext.Provider>
  )
}

// Grouped like the Dashboard home: Home | Forecast | Respond | Evidence
const SECTIONS = [
  [{ to: '/dashboard', label: 'Home', end: true }],
  [
    { to: '/dashboard/dapt', label: 'DAPT2020' },
    { to: '/dashboard/zeek', label: 'ZeekData24' },
    { to: '/dashboard/labs/cic17', label: 'CIC-IDS2017' },
    { to: '/dashboard/labs/ctu13', label: 'CTU-13' },
  ],
  [{ to: '/dashboard/response', label: 'Response' }],
  [{ to: '/dashboard/zero-shot', label: 'Zero-shot' }],
]

/** The Dashboard tab: a dense analyst app with its own section bar. */
export function DashboardLayout() {
  return (
    <>
      <div className="sticky top-15 z-20 border-b border-line bg-page/85 backdrop-blur-md">
        <nav aria-label="Dashboard sections" className="mx-auto flex max-w-[1400px] items-center overflow-x-auto px-4 [scrollbar-width:none] sm:px-6 lg:px-10">
          {SECTIONS.map((group, g) => (
            <div key={g} className="flex shrink-0 items-center gap-1">
              {g > 0 && <span aria-hidden className="mx-2 h-4 w-px bg-line-strong" />}
              {group.map((s) => (
                <NavLink
                  key={s.to}
                  to={s.to}
                  end={'end' in s ? s.end : false}
                  className={({ isActive }) =>
                    clsx(
                      '-mb-px shrink-0 border-b-2 px-3 py-2.5 text-[13px] font-medium transition-colors duration-200',
                      isActive ? 'border-accent text-ink' : 'border-transparent text-muted hover:text-ink',
                    )
                  }
                >
                  {s.label}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
      </div>
      <div className="mx-auto max-w-[1400px] space-y-6 px-4 py-6 sm:px-6 lg:px-10 lg:py-8">
        <Suspense fallback={<div className="skeleton h-64 w-full" />}>
          <Outlet />
        </Suspense>
      </div>
    </>
  )
}
