import { readFile } from 'node:fs/promises'
import { object, string } from 'valibot'
import * as valibot from 'valibot'
import * as ahocon from 'ahocon'

const file = await readFile('secrets.conf', 'utf-8')

const schema = object({
  deepgram: string(),
  googleAiStudio: string(),
})

const secrets = valibot.parse(schema, ahocon.parse(file))

export default secrets
