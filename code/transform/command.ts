import { Args, Command, Flags } from '@oclif/core'
import transform from './transform'
import { getProfile, profileNames } from 'code/config/read-profiles'


export default class Transform extends Command {
  static override strict = false
  static override args = {
    file: Args.string({ description: 'Files to read', required: true }),
  }
  static override description = 'Send files to LLM'
  static override examples = [
    '<%= config.bin %> <%= command.id %>',
  ]
  static override flags = {
    profile: Flags.option({
      char: 'p',
      options: profileNames,
      description: 'Which config profile to use?',
      required: true,
    })(),
  }

  public async run(): Promise<void> {
    const { argv, flags: { profile }} = await this.parse(Transform)
    const sourcePaths = argv as string[]
    const config = await getProfile(profile)

    for (const sourcePath of sourcePaths)
      await transform(sourcePath, config.transform)
  }
}
