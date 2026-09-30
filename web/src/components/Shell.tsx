import { IconContext, WarningDiamond } from '@phosphor-icons/react'
import clsx from 'clsx'
import { motion, MotionConfig } from 'motion/react'
import { Suspense, useEffect, useState, type ReactNode } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { useResolvedTheme } from '../lib/theme'
import { EYE_PROPS, HEADER_BG, REPLAY_INTRO } from './brand'
import EvilEye from './EvilEye'

type Theme = 'system' | 'light' | 'dark'

function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    const t = document.documentElement.dataset.themeChoice
    return t === 'light' || t === 'system' ? t : 'dark' // dark is the default
  })
  useEffect(() => {
    if (theme === 'system') delete document.documentElement.dataset.theme
    else document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem('netrikan-theme', theme) // 'system' is stored too, so it isn't replaced by the dark default
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

const TOP = [
  { to: '/', label: 'Overview', end: true },
  { to: '/dashboard', label: 'Dashboard', end: false },
]

function TopBar() {
  const [theme, setTheme] = useTheme()
  const shown = useResolvedTheme()
  // On the Overview page the intro eye (pages/Landing.tsx) flies into this slot as you scroll, so the header keeps the slot empty.
  const intro = useLocation().pathname === '/'
  return (
    <header className="sticky top-0 z-30 border-b border-line" style={{ background: HEADER_BG[shown] }}>
      <div className="mx-auto grid h-15 max-w-[1400px] grid-cols-[1fr_auto_1fr] items-center gap-4 px-4 sm:px-6 lg:px-10">
        <div className="flex min-w-0 items-center gap-5">
          <Link to="/" className="hidden shrink-0 rounded-md text-[15px] font-semibold tracking-[-0.01em] focus-visible:outline-2 focus-visible:outline-accent sm:block">
            Netrikan
          </Link>
          <nav aria-label="Primary" className="flex items-center gap-1">
            {TOP.map((t) => (
              <NavLink
                key={t.to}
                to={t.to}
                end={t.end}
                className={({ isActive }) =>
                  clsx(
                    'relative rounded-md px-3 py-1.5 text-[13.5px] font-medium transition-colors duration-200 focus-visible:outline-2 focus-visible:outline-accent',
                    isActive ? 'text-ink' : 'text-ink-2 hover:text-ink',
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    {isActive && <motion.span layoutId="top-tab" className="absolute inset-0 -z-10 rounded-md bg-surface-2" transition={{ type: 'spring', stiffness: 400, damping: 34 }} />}
                    {t.label}
                  </>
                )}
              </NavLink>
            ))}
          </nav>
        </div>
        <Link
          to="/"
          id="eye-slot"
          aria-label={intro ? 'Replay the Netrikan intro' : 'Netrikan home'}
          onClick={(e) => {
            if (!intro) return
            e.preventDefault() // already on Overview: replay the big-eye intro instead
            window.dispatchEvent(new Event(REPLAY_INTRO))
          }}
          className="block h-15 w-36 overflow-hidden rounded-md focus-visible:outline-2 focus-visible:outline-accent"
        >
          {!intro && <EvilEye className="h-full w-full" {...EYE_PROPS} backgroundColor={HEADER_BG[shown]} lightMode={shown === 'light'} {...(shown === 'light' && { eyeColor: '#111111', intensity: 2.8, glowIntensity: 0.5 })} />}
        </Link>
        <div className="flex items-center justify-end gap-3">
          <label className="hidden items-center gap-2 text-[12.5px] text-muted md:flex">
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

function DemoBanner() {
  return (
    <div className="flex items-center justify-center gap-2 border-b border-amber-500/30 bg-amber-500/10 px-4 py-2 text-[12.5px] text-amber-600 dark:text-amber-400">
      <WarningDiamond weight="fill" className="size-3.5 shrink-0" />
      <span>Frontend demo — backend offline. Live packet analysis and file uploads are unavailable.</span>
    </div>
  )
}

export function SiteLayout({ children }: { children: ReactNode }) {
  useScrollTopOnNavigate()
  return (
    <MotionConfig reducedMotion="user">
    <IconContext.Provider value={{ weight: 'regular' }}>
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-surface focus:px-3 focus:py-2">
        Skip to content
      </a>
      <TopBar />
      <DemoBanner />
      <main id="main">{children}</main>
    </IconContext.Provider>
    </MotionConfig>
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
  const { pathname } = useLocation()
  return (
    <>
      <div className="sticky top-15 z-20 border-b border-line bg-page/90 backdrop-blur-md">
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
                    clsx('relative shrink-0 px-3 py-2.5 text-[13px] font-medium transition-colors duration-200', isActive ? 'text-ink' : 'text-muted hover:text-ink')
                  }
                >
                  {({ isActive }) => (
                    <>
                      {s.label}
                      {isActive && <motion.span layoutId="section-underline" className="absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-accent" transition={{ type: 'spring', stiffness: 420, damping: 36 }} />}
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
      </div>
      <div className="mx-auto max-w-[1400px] space-y-6 px-4 py-6 sm:px-6 lg:px-10 lg:py-8">
        <Suspense fallback={<div className="skeleton h-64 w-full" />}>
          <motion.div key={pathname} className="space-y-6" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}>
            <Outlet />
          </motion.div>
        </Suspense>
      </div>
    </>
  )
}
