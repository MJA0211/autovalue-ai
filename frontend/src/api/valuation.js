const DEFAULT_API_BASE_URL = import.meta.env.DEV ? 'http://localhost:8000' : ''

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(/\/$/, '')

async function parseResponse(response) {
  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    const message = payload?.detail?.message || payload?.detail || 'The request could not be completed.'
    throw new Error(typeof message === 'string' ? message : 'The request could not be completed.')
  }
  return payload
}

export async function fetchModelStatus(signal) {
  const response = await fetch(`${apiBaseUrl}/api/v1/model`, {
    headers: { Accept: 'application/json' },
    signal,
  })
  return parseResponse(response)
}

export async function createValuation(vehicle, clientId, signal) {
  const response = await fetch(`${apiBaseUrl}/api/v1/valuations`, {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      'X-AutoValue-Client': clientId,
    },
    body: JSON.stringify(vehicle),
    signal,
  })
  return parseResponse(response)
}

export async function fetchRecentPredictions(clientId, signal) {
  const response = await fetch(`${apiBaseUrl}/api/v1/predictions/recent`, {
    headers: { Accept: 'application/json', 'X-AutoValue-Client': clientId },
    signal,
  })
  const payload = await parseResponse(response)
  return payload.predictions
}
