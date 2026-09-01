import assert from 'node:assert/strict'
import { test } from 'node:test'
import { generateAnswersWithOpenAICompatibleApi } from '../../../../src/services/apis/openai-api.mjs'
import { createFakePort } from '../../helpers/port.mjs'
import { createMockSseResponse } from '../../helpers/sse-response.mjs'

const AIMLAPI_ATTRIBUTION_HEADER_NAMES = [
  'HTTP-Referer',
  'X-Title',
  'X-AIMLAPI-Source',
  'X-AIMLAPI-Partner-ID',
]
const OPENROUTER_ONLY_HEADER_NAMES = ['X-OpenRouter-Title', 'X-OpenRouter-Categories']

function createConfig(overrides = {}) {
  return {
    maxConversationContextLength: 3,
    maxResponseTokenLength: 256,
    temperatureOverrideEnabled: false,
    temperature: 1,
    ...overrides,
  }
}

async function captureRequest(t, config, session) {
  let capturedInput
  let capturedInit
  t.mock.method(globalThis, 'fetch', async (input, init) => {
    capturedInput = input
    capturedInit = init
    return createMockSseResponse([
      'data: {"choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}]}\n\n',
    ])
  })

  await generateAnswersWithOpenAICompatibleApi(createFakePort(), 'CurrentQ', session, config)

  return { capturedInput, capturedInit }
}

function assertAimlapiAttribution(headers) {
  assert.equal(headers['HTTP-Referer'], 'https://github.com/ChatGPTBox-dev/chatGPTBox')
  assert.equal(headers['X-Title'], 'ChatGPTBox')
  assert.equal(headers['X-AIMLAPI-Source'], 'agent/chatgptbox')
  assert.equal(headers['X-AIMLAPI-Partner-ID'], 'part_suAsqzeIpC2a5yS8K1leCNtf')
}

function assertNoAimlapiAttribution(headers) {
  for (const headerName of AIMLAPI_ATTRIBUTION_HEADER_NAMES) {
    assert.equal(Object.hasOwn(headers, headerName), false)
  }
}

test('adds app attribution headers for built-in AI/ML API requests', async (t) => {
  const config = createConfig({
    providerSecrets: {
      aiml: 'aiml-test-key',
    },
  })
  const session = {
    modelName: 'aiml_openai_gpt_5_5',
    conversationRecords: [],
    isRetry: false,
  }

  const { capturedInput, capturedInit } = await captureRequest(t, config, session)

  assert.equal(capturedInput, 'https://api.aimlapi.com/v1/chat/completions')
  assert.equal(capturedInit.headers.Authorization, 'Bearer aiml-test-key')
  assertAimlapiAttribution(capturedInit.headers)

  // Attribution belongs in the headers; it must never leak into the payload.
  const body = JSON.parse(capturedInit.body)
  for (const headerName of AIMLAPI_ATTRIBUTION_HEADER_NAMES) {
    assert.equal(Object.hasOwn(body, headerName), false)
  }
})

test('keeps OpenRouter-specific headers off AI/ML API requests', async (t) => {
  const config = createConfig({
    providerSecrets: {
      aiml: 'aiml-test-key',
    },
  })
  const session = {
    modelName: 'aiml_openai_gpt_5_5',
    conversationRecords: [],
    isRetry: false,
  }

  const { capturedInit } = await captureRequest(t, config, session)

  for (const headerName of OPENROUTER_ONLY_HEADER_NAMES) {
    assert.equal(Object.hasOwn(capturedInit.headers, headerName), false)
  }
})

test('keeps AI/ML API headers off OpenRouter requests', async (t) => {
  const config = createConfig({
    providerSecrets: {
      openrouter: 'sk-or-test',
    },
  })
  const session = {
    modelName: 'openRouter_auto',
    conversationRecords: [],
    isRetry: false,
  }

  const { capturedInit } = await captureRequest(t, config, session)

  assert.equal(Object.hasOwn(capturedInit.headers, 'X-AIMLAPI-Source'), false)
  assert.equal(Object.hasOwn(capturedInit.headers, 'X-AIMLAPI-Partner-ID'), false)
  assert.equal(Object.hasOwn(capturedInit.headers, 'X-Title'), false)
})

test('adds app attribution headers for custom providers that call AI/ML API directly', async (t) => {
  const config = createConfig({
    customOpenAIProviders: [
      {
        id: 'direct-aimlapi',
        name: 'Direct AI/ML API',
        baseUrl: 'https://api.aimlapi.com/v1',
        chatCompletionsPath: '/chat/completions',
        completionsPath: '/completions',
        enabled: true,
      },
    ],
    providerSecrets: {
      'direct-aimlapi': 'direct-key',
    },
  })
  const session = {
    modelName: 'customModel',
    conversationRecords: [],
    isRetry: false,
    apiMode: {
      groupName: 'customApiModelKeys',
      itemName: 'customModel',
      isCustom: true,
      providerId: 'direct-aimlapi',
      customName: 'openai/gpt-5-5',
      customUrl: '',
      apiKey: '',
      active: true,
    },
  }

  const { capturedInput, capturedInit } = await captureRequest(t, config, session)

  assert.equal(capturedInput, 'https://api.aimlapi.com/v1/chat/completions')
  assert.equal(capturedInit.headers.Authorization, 'Bearer direct-key')
  assertAimlapiAttribution(capturedInit.headers)
})

test('omits AI/ML API attribution headers for third-party proxy endpoints', async (t) => {
  const config = createConfig({
    customOpenAIProviders: [
      {
        id: 'aimlapi-proxy',
        name: 'AI/ML API Proxy',
        baseUrl: 'https://proxy.example.com/v1',
        chatCompletionsPath: '/chat/completions',
        completionsPath: '/completions',
        sourceProviderId: 'aiml',
        enabled: true,
      },
    ],
    providerSecrets: {
      'aimlapi-proxy': 'proxy-key',
    },
  })
  const session = {
    modelName: 'customModel',
    conversationRecords: [],
    isRetry: false,
    apiMode: {
      groupName: 'customApiModelKeys',
      itemName: 'customModel',
      isCustom: true,
      providerId: 'aimlapi-proxy',
      customName: 'openai/gpt-5-5',
      customUrl: '',
      apiKey: '',
      active: true,
    },
  }

  const { capturedInput, capturedInit } = await captureRequest(t, config, session)

  assert.equal(capturedInput, 'https://proxy.example.com/v1/chat/completions')
  assert.equal(capturedInit.headers.Authorization, 'Bearer proxy-key')
  assertNoAimlapiAttribution(capturedInit.headers)
})
