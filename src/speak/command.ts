import { spawn } from 'node:child_process'
import { resolve } from 'node:path'
import { Args, Command, Flags } from '@oclif/core'

const float = Flags.custom<number>({
  parse: async input => {
    const value = Number(input)
    if (!Number.isFinite(value)) throw new Error(`Expected a number but received: ${input}`)
    return value
  },
})

export default class Speak extends Command {
  static override args = {
    file: Args.file({ description: 'Text file to read aloud', exists: true, required: true }),
  }

  static override description = 'Read a text file aloud with Supertonic'

  static override examples = [
    '<%= config.bin %> <%= command.id %> document.txt',
  ]

  static override flags = {
    voice: Flags.string({ default: 'M1', description: 'Supertonic voice style' }),
    lang: Flags.string({ description: 'Language code' }),
    speed: float({ default: 1.05, description: 'Speech speed' }),
    steps: Flags.integer({ default: 8, description: 'Number of synthesis steps' }),
    'max-chunk-length': Flags.integer({ default: 300, description: 'Maximum text chunk length' }),
    'silence-duration': float({ default: 0.3, description: 'Silence between chunks in seconds' }),
    buffer: Flags.integer({ default: 3, description: 'Number of synthesised chunks to buffer ahead' }),
  }

  public async run(): Promise<void> {
    const { args, flags } = await this.parse(Speak)
    const projectPath = import.meta.dirname
    const scriptPath = resolve(projectPath, 'speak.py')
    const uvArgs = [
      'run',
      '--project', projectPath,
      'python',
      scriptPath,
      args.file,
      '--voice', flags.voice,
      '--speed', String(flags.speed),
      '--steps', String(flags.steps),
      '--max-chunk-length', String(flags['max-chunk-length']),
      '--silence-duration', String(flags['silence-duration']),
      '--buffer', String(flags.buffer),
    ]

    if (flags.lang) uvArgs.push('--lang', flags.lang)

    let result: { code: number | null, signal: NodeJS.Signals | null }
    try {
      result = await new Promise((resolveResult, reject) => {
        const child = spawn('uv', uvArgs, { stdio: 'inherit' })
        child.once('error', reject)
        child.once('close', (code, signal) => resolveResult({ code, signal }))
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      this.error(`Unable to start uv: ${message}`)
    }

    if (result.code !== 0) {
      const reason = result.signal ? `signal ${result.signal}` : `exit code ${result.code}`
      this.error(`Speak failed with ${reason}`)
    }
  }
}
