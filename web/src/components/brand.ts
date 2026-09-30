import type { EvilEyeProps } from './EvilEye'

/** Header background per theme (the user's EvilEye background in dark; the light page colour in light). */
export const HEADER_BG = { light: '#f3f4f7', dark: '#000000' } as const

/** The Netrikan eye as specified; shared by the header logo and the landing intro so they look identical. */
export const EYE_PROPS: EvilEyeProps = {
  eyeColor: '#FF6F37',
  intensity: 1.5,
  pupilSize: 0.6,
  irisWidth: 0.25,
  glowIntensity: 0.35,
  scale: 0.8,
  noiseScale: 1.0,
  pupilFollow: 1.0,
  flameSpeed: 1.0,
  transparent: true,
}

/** The header slot's aspect ratio (width / height); the intro eye keeps it so it lands on the slot exactly. */
export const EYE_ASPECT = 2.4

/** Fired when the header eye is clicked on the Overview page: the intro replays (pages/Landing.tsx listens). */
export const REPLAY_INTRO = 'netrikan:replay-intro'
