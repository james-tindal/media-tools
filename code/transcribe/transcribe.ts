import { createClient } from '@deepgram/sdk'
import fs from 'node:fs'
import { writeFile } from 'node:fs/promises'
import secrets from 'code/config/read-secrets'
import logger from './logger'
import path from 'node:path'
import { Readable } from 'node:stream'


const deepgram = createClient(secrets.deepgram)

export async function transcribe(sourcePath: string) {
  const readable = Readable.from(readSource(sourcePath))

  const { result, error } = await deepgram.listen.prerecorded.transcribeFile(readable)
  if (error)
    throw error

  console.log('received transcript')
  const { transcript } = result.results.channels[0].alternatives[0]

  const outputPath = getOutputPath(sourcePath)
  writeFile(outputPath, transcript)
}

async function* readSource(sourcePath: string) {
  logger.sending.start()
  const stream = fs.createReadStream(sourcePath)
  for await (const chunk of stream) {
    logger.sending.chunk(chunk)
    yield chunk
  }
  logger.sending.end()
}

function getOutputPath (sourcePath: string) {
  const { base, ...p } = path.parse(sourcePath)
  return path.format({ ...p, ext: '.txt' })
}
