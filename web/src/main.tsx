import '@fontsource-variable/geist'
import '@fontsource-variable/geist-mono'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { lazy, StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Navigate, Route, Routes, useLocation, useParams } from 'react-router-dom'
import { DashboardLayout, SiteLayout } from './components/Shell'
import './index.css'
import LandingPage from './pages/Landing'
import NotFound from './pages/NotFound'

// Dashboard pages load on demand, so the landing page stays small.
const DashboardHome = lazy(() => import('./pages/DashboardHome'))
const DaptPage = lazy(() => import('./pages/Dapt'))
const ZeekPage = lazy(() => import('./pages/Zeek'))
const ResponsePage = lazy(() => import('./pages/Response'))
const CorpusPage = lazy(() => import('./pages/Corpus'))
const ZeroShotPage = lazy(() => import('./pages/ZeroShot'))

const qc = new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false } } })

/** Old top-level URLs (/dapt, /response?...) keep working: they move under /dashboard with their query string. */
function Moved({ to }: { to: string }) {
  const { search } = useLocation()
  const { name } = useParams()
  return <Navigate to={`${to.replace(':name', name ?? '')}${search}`} replace />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <SiteLayout>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/dashboard" element={<DashboardLayout />}>
              <Route index element={<DashboardHome />} />
              <Route path="dapt" element={<DaptPage />} />
              <Route path="zeek" element={<ZeekPage />} />
              <Route path="response" element={<ResponsePage />} />
              <Route path="labs/:name" element={<CorpusPage />} />
              <Route path="zero-shot" element={<ZeroShotPage />} />
            </Route>
            <Route path="/dapt" element={<Moved to="/dashboard/dapt" />} />
            <Route path="/zeek" element={<Moved to="/dashboard/zeek" />} />
            <Route path="/response" element={<Moved to="/dashboard/response" />} />
            <Route path="/labs/:name" element={<Moved to="/dashboard/labs/:name" />} />
            <Route path="/zero-shot" element={<Moved to="/dashboard/zero-shot" />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </SiteLayout>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
)
