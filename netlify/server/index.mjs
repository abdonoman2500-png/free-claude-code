// Free Claude Code — web edition.
// Runs on Netlify as a long-running Node server and routes chat/proxy requests
// to models served through Netlify AI Gateway (no provider API keys needed).
import { timingSafeEqual } from 'node:crypto'
import { Hono } from 'hono'
import Anthropic from '@anthropic-ai/sdk'
import OpenAI from 'openai'
import { GoogleGenAI } from '@google/genai'

export const config = { path: ['/api/*', '/v1/*'] }

const anthropic = new Anthropic()
const openai = new OpenAI()
const gemini = new GoogleGenAI({})

// Models available through Netlify AI Gateway.
const MODELS = {
  anthropic: ['claude-sonnet-5-5', 'claude-opus-5-5', 'claude-haiku-5-5', 'claude-fable-5-2', 'claude-sonnet-4-6', 'claude-opus-4-8'],
  openai: ['gpt-5.5', 'gpt-5.4-mini', 'gpt-5.3-codex', 'gpt-5.2', 'gpt-5-mini'],
  gemini: ['gemini-3.8-flash', 'gemini-3.1-pro-preview', 'gemini-3.5-flash-lite', 'gemini-flash-latest'],
}
const DEFAULT_CLAUDE = 'claude-sonnet-5-5'
const providerOf = (model) =>
  Object.keys(MODELS).find((p) => MODELS[p].includes(model)) ?? null

// Claude Code sends its own tier names (e.g. claude-opus-4-1-20250805); map
// any Claude-family name onto an available gateway model of the same tier.
function resolveClaudeModel(model = '') {
  if (MODELS.anthropic.includes(model)) return model
  const m = model.toLowerCase()
  if (m.includes('opus')) return 'claude-opus-5-5'
  if (m.includes('haiku')) return 'claude-haiku-5-5'
  if (m.includes('fable')) return 'claude-fable-5-2'
  return DEFAULT_CLAUDE
}

// ---- Access control -------------------------------------------------------
// All endpoints spend the site's AI credits, so they require FCC_ACCESS_TOKEN.
function extractToken(c) {
  const auth = c.req.header('authorization') ?? ''
  if (auth.toLowerCase().startsWith('bearer ')) return auth.slice(7).trim()
  return (c.req.header('x-api-key') ?? c.req.header('x-goog-api-key') ?? '').trim()
}
function tokenOk(given) {
  const expected = process.env.FCC_ACCESS_TOKEN ?? ''
  if (!expected || !given) return false
  const a = Buffer.from(given)
  const b = Buffer.from(expected)
  return a.length === b.length && timingSafeEqual(a, b)
}

const app = new Hono()

app.onError((err, c) => {
  console.error(err)
  const status = typeof err?.status === 'number' ? err.status : 500
  return c.json({ type: 'error', error: { type: 'api_error', message: err?.message ?? 'Internal error' } }, status)
})

app.get('/api/health', (c) => c.json({ ok: true, configured: Boolean(process.env.FCC_ACCESS_TOKEN) }))

app.use('*', async (c, next) => {
  if (!process.env.FCC_ACCESS_TOKEN) {
    return c.json({ type: 'error', error: { type: 'configuration_error', message: 'FCC_ACCESS_TOKEN is not configured on this site.' } }, 503)
  }
  if (!tokenOk(extractToken(c))) {
    return c.json({ type: 'error', error: { type: 'authentication_error', message: 'Invalid or missing access token.' } }, 401)
  }
  await next()
})

app.get('/api/models', (c) => c.json({ models: MODELS, default: DEFAULT_CLAUDE }))

app.get('/v1/models', (c) => {
  const data = Object.entries(MODELS).flatMap(([owned_by, ids]) =>
    ids.map((id) => ({ id, object: 'model', type: 'model', display_name: id, owned_by, created: 0 })),
  )
  return c.json({ object: 'list', data, has_more: false, first_id: data[0]?.id, last_id: data.at(-1)?.id })
})

// ---- Browser chat (simple SSE: {t: text} / {error} / {done}) --------------
app.post('/api/chat', async (c) => {
  const { model, messages, system } = await c.req.json()
  const provider = providerOf(model)
  if (!provider) return c.json({ error: { message: `Unknown model: ${model}` } }, 400)
  if (!Array.isArray(messages) || messages.length === 0) return c.json({ error: { message: 'messages required' } }, 400)
  const clean = messages
    .filter((m) => (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string' && m.content)
    .slice(-40)

  const encoder = new TextEncoder()
  const stream = new ReadableStream({
    async start(controller) {
      const send = (obj) => controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`))
      try {
        if (provider === 'anthropic') {
          const s = await anthropic.messages.create({
            model, max_tokens: 8192, stream: true, messages: clean, ...(system ? { system } : {}),
          })
          for await (const ev of s) {
            if (ev.type === 'content_block_delta' && ev.delta.type === 'text_delta') send({ t: ev.delta.text })
          }
        } else if (provider === 'openai') {
          const s = await openai.chat.completions.create({
            model, stream: true,
            messages: [...(system ? [{ role: 'system', content: system }] : []), ...clean],
          })
          for await (const ch of s) {
            const t = ch.choices?.[0]?.delta?.content
            if (t) send({ t })
          }
        } else {
          const s = await gemini.models.generateContentStream({
            model,
            contents: clean.map((m) => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.content }] })),
            ...(system ? { config: { systemInstruction: system } } : {}),
          })
          for await (const ch of s) if (ch.text) send({ t: ch.text })
        }
        send({ done: true })
      } catch (err) {
        console.error(err)
        send({ error: err?.message ?? 'Request failed' })
      } finally {
        controller.close()
      }
    },
  })
  return new Response(stream, {
    headers: { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache, no-transform' },
  })
})

// ---- Anthropic Messages API proxy (for Claude Code) -----------------------
// Point Claude Code at this site: ANTHROPIC_BASE_URL=https://<site>  ANTHROPIC_AUTH_TOKEN=<token>
const passthrough = (res) =>
  new Response(res.body, {
    status: res.status,
    headers: {
      'content-type': res.headers.get('content-type') ?? 'application/json',
      'cache-control': 'no-cache, no-transform',
    },
  })

app.post('/v1/messages', async (c) => {
  const body = await c.req.json()
  body.model = resolveClaudeModel(body.model)
  const res = await anthropic.messages.create(body).asResponse()
  return passthrough(res)
})

app.post('/v1/messages/count_tokens', async (c) => {
  const body = await c.req.json()
  body.model = resolveClaudeModel(body.model)
  const res = await anthropic.messages.countTokens(body).asResponse()
  return passthrough(res)
})

// ---- OpenAI-compatible APIs ------------------------------------------------
app.post('/v1/chat/completions', async (c) => {
  const body = await c.req.json()
  if (!MODELS.openai.includes(body.model)) body.model = MODELS.openai[0]
  const res = await openai.chat.completions.create(body).asResponse()
  return passthrough(res)
})

app.post('/v1/responses', async (c) => {
  const body = await c.req.json()
  if (!MODELS.openai.includes(body.model)) body.model = MODELS.openai[0]
  const res = await openai.responses.create(body).asResponse()
  return passthrough(res)
})

app.all('*', (c) => c.json({ type: 'error', error: { type: 'not_found_error', message: 'Not found' } }, 404))

export default app
