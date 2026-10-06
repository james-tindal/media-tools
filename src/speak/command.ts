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
  static override strict = false

  static override args = {
    file: Args.string({ description: 'Markdown/text file or directory; accepts multiple paths', required: true }),
  }

  static override description = 'Read Markdown or text aloud, or generate audio with Supertonic'

  static override examples = [
    '<%= config.bin %> <%= command.id %> document.txt',
    '<%= config.bin %> <%= command.id %> document.txt --output speech.wav',
    '<%= config.bin %> <%= command.id %> document.md --voice F5 --output speech.opus --format opus:24',
    '<%= config.bin %> <%= command.id %> chapters/ --voice F5 --output-dir audio/',
  ]

  static override flags = {
    output: Flags.file({ char: 'o', description: 'Write one input to a WAV or Opus file instead of playing it' }),
    'output-dir': Flags.string({ description: 'Write one Opus file per input to this directory' }),
    format: Flags.string({ description: 'Opus encoding: opus or opus:N, where N is the bitrate in kbps (default: 24)' }),
    overwrite: Flags.boolean({ description: 'Replace existing audio after synthesis and verification succeed' }),
    voice: Flags.string({ default: 'M1', description: 'Supertonic voice style' }),
    lang: Flags.string({ description: 'Language code' }),
    speed: float({ default: 1.05, description: 'Speech speed' }),
    steps: Flags.integer({ default: 8, description: 'Number of synthesis steps' }),
    'max-chunk-length': Flags.integer({ default: 300, description: 'Maximum text chunk length' }),
    'silence-duration': float({ default: 0.3, description: 'Silence between chunks in seconds' }),
    buffer: Flags.integer({ default: 3, description: 'Number of synthesised chunks to buffer ahead' }),
  }

  public async run(): Promise<void> {
    const { argv, flags } = await this.parse(Speak)
    if (!argv.every(value => typeof value === 'string')) this.error('Expected file or directory paths')
    const projectPath = import.meta.dirname
    const scriptPath = resolve(projectPath, 'speak.py')
    const uvArgs = [
      'run',
      '--project', projectPath,
      'python',
      scriptPath,
      ...argv,
      '--voice', flags.voice,
      '--speed', String(flags.speed),
      '--steps', String(flags.steps),
      '--max-chunk-length', String(flags['max-chunk-length']),
      '--silence-duration', String(flags['silence-duration']),
      '--buffer', String(flags.buffer),
    ]

    if (flags.lang) uvArgs.push('--lang', flags.lang)
    if (flags.output) uvArgs.push('--output', flags.output)
    if (flags['output-dir']) uvArgs.push('--output-dir', flags['output-dir'])
    if (flags.format) uvArgs.push('--format', flags.format)
    if (flags.overwrite) uvArgs.push('--overwrite')

    const executable = process.platform === 'darwin' ? 'caffeinate' : 'uv'
    const spawnArgs = process.platform === 'darwin' ? ['-i', 'uv', ...uvArgs] : uvArgs
    let result: { code: number | null, signal: NodeJS.Signals | null }
    try {
      result = await new Promise((resolveResult, reject) => {
        const child = spawn(executable, spawnArgs, { stdio: 'inherit' })
        child.once('error', reject)
        child.once('close', (code, signal) => resolveResult({ code, signal }))
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      this.error(`Unable to start ${executable}: ${message}`)
    }

    if (result.code !== 0) {
      const reason = result.signal ? `signal ${result.signal}` : `exit code ${result.code}`
      this.error(`Speak failed with ${reason}`)
    }
  }
}
