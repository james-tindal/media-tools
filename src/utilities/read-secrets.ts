import { readFile } from 'node:fs/promises'
import { object, string } from 'valibot'
import * as valibot from 'valibot'
import * as ahocon from 'ahocon'
import { packageDirectory } from 'package-directory'
import { resolve } from 'node:path'

const rootDir = await packageDirectory({ cwd: import.meta.dirname })
const file = await readFile(resolve(rootDir!, 'secrets.conf'), 'utf-8')

const schema = object({
  deepgram: string(),
  googleAiStudio: string(),
})

const secrets = valibot.parse(schema, ahocon.parse(file))

export default secrets
