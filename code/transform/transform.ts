import { GoogleGenAI } from '@google/genai'
import secrets from 'code/config/read-secrets'
import { readFile } from 'node:fs/promises'
import { Config } from './configSchema'
import { createWriteStream } from 'node:fs'
import path from 'node:path'
import { fileExists } from 'code/utilities'
import logUpdate from 'log-update'

const google = new GoogleGenAI({ apiKey: secrets.googleAiStudio })

export default async function transform(sourcePath: string, config: Config) {
  const outputPath = getOutputPath(sourcePath, config.output)
  const sourceExists = await fileExists(sourcePath)
  const outputExists = await fileExists(outputPath)
  if (!sourceExists) {
    console.log()
    console.log(`Source file not found: ${sourcePath}`)
    return
  }
  if (outputExists) {
    console.log()
    console.log('Destination file already exists. Not overwriting.')
    console.log(outputPath)
    return
  }

  console.log('\n' + outputPath)

  const contents = await readFile(sourcePath, 'utf-8')
  logUpdate('Sending request')
  const response = await google.models.generateContentStream({
    contents,
    model: 'gemini-2.5-flash',
    config: { systemInstruction: config.systemInstruction },
  })
  const outputFile = createWriteStream(outputPath)
  let chunkCount = 0
  for await (const chunk of response) {
    if (chunk === undefined) continue
    logUpdate(`Received chunk ${++chunkCount}`)
    outputFile.write(chunk.text)
  }
  outputFile.close()
  logUpdate.clear()
}

function getOutputPath(sourcePath: string, config: Config['output']) {
  const { dir, name, root } = path.parse(sourcePath)
  const { method, argument } = config.pathTransform || {}

  const currentFolderName = path.basename(process.cwd())

  return path.format({
    dir:
      method === 'folder' ? argument :
      method === 'appendFolder'
      ? dir === ''
        ? path.join('..', currentFolderName + argument)
        : dir + argument
      : dir,
    name: method === 'appendFile' ? name + argument : name,
    ext: config.extension,
    root,
  })
}
