import { readdir, readFile } from 'node:fs/promises'
import * as valibot from 'valibot'
import * as ahocon from 'ahocon'
import { packageDirectory } from 'package-directory'
import { resolve } from 'node:path'
import { configSchema } from './schema'

const rootDir = await packageDirectory({ cwd: import.meta.dirname })
if (!rootDir) throw Error('Couldn\'t find package.json root directory')

const profilesDir = resolve(rootDir, 'profiles')
export const profileNames =
  (await readdir(profilesDir))
  .filter(fileName => fileName.endsWith('.conf'))

export async function getProfile(profileName: string) {
  const file = await readFile(resolve(profilesDir, `${profileName}.conf`), 'utf-8')
  return valibot.parse(configSchema, ahocon.parse(file))
}
