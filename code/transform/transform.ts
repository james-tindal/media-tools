import { GoogleGenAI } from '@google/genai'
import secrets from 'code/config/read-secrets'
import { readFile } from 'node:fs/promises'
import { Config } from './configSchema'
import { createWriteStream } from 'node:fs'
import path from 'node:path'

const google = new GoogleGenAI({ apiKey: secrets.googleAiStudio })

export default async function transform(sourcePath: string, config: Config) {
  const contents = await readFile(sourcePath, 'utf-8')
  const response = await google.models.generateContentStream({
    contents,
    model: 'gemini-2.5-flash',
    config: { systemInstruction: config.systemInstruction },
  })
  const outputPath = getOutputPath(sourcePath, config.output)
  const outputFile = createWriteStream(outputPath)
  for await (const chunk of response)
    outputFile.write(chunk.text)
  outputFile.close()
}

function getOutputPath(sourcePath: string, config: Config['output']) {
  const { dir, name, root } = path.parse(sourcePath)
  const { method, argument } = config.pathTransform || {}

  return path.format({
    dir:
      method === 'folder' ? argument :
      method === 'appendFolder' ? dir + argument : dir,
    name: method === 'appendFile' ? name + argument : name,
    ext: config.extension,
    root,
  })
}
