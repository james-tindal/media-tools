import { InferOutput, object } from 'valibot'
import { schema as transform } from 'code/transform/configSchema'

export type Config = InferOutput<typeof configSchema>
export const configSchema = object({
  transform,
})
