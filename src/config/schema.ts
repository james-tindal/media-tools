import { InferOutput, object } from 'valibot'
import { schema as transform } from 'src/transform/configSchema'

export type Config = InferOutput<typeof configSchema>
export const configSchema = object({
  transform,
})
