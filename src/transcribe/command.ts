import { Args, Command } from '@oclif/core'
import { transcribe } from './transcribe'

export default class Transcribe extends Command {
  static override strict = false
  static override args = {
    files: Args.string({ description: 'files to read', required: true }),
  }
  static override description = 'Transcribe audio to text'
  static override examples = [
    '<%= config.bin %> <%= command.id %>',
  ]
  public async run(): Promise<void> {
    const { argv } = await this.parse(Transcribe)
    const sourcePaths = argv as string[]

    for (const sourcePath of sourcePaths)
      await transcribe(sourcePath)
  }
}
