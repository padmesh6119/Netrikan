import { Link } from 'react-router-dom'

export default function NotFound() {
  return (
    <section className="mx-auto flex min-h-[60dvh] max-w-[1400px] flex-col justify-center px-4 sm:px-6 lg:px-10">
      <h1 className="text-[32px] font-semibold tracking-[-0.02em]">This page does not exist.</h1>
      <p className="mt-2 max-w-[55ch] text-ink-2">The link may point to an old address. The analyst views now live under the Dashboard tab.</p>
      <div className="mt-6 flex gap-3">
        <Link to="/dashboard" className="btn btn-primary">Dashboard</Link>
        <Link to="/" className="btn btn-quiet">Overview</Link>
      </div>
    </section>
  )
}
