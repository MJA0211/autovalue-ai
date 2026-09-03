const DEFAULT_API_BASE_URL = import.meta.env.DEV ? 'http://localhost:8000' : ''

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(/\/$/, '')

export async function fetchApiHealth(signal) {
  const response = await fetch(`${apiBaseUrl}/health/live`, {
    headers: { Accept: 'application/json' },
    signal,
  })

  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`)
  }

  return response.json()
}
