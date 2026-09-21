import { object, optional, string, pipe, union, intersect, transform, InferOutput } from 'valibot'

const extension = object({ extension: optional(string(), 'md') })
const appendFile = object({ appendFile: string() })
const appendFolder = object({ appendFolder: string() })
const folder = object({ folder: string() })
const empty = object({})
const pathTransform = pipe(
  union([appendFile, appendFolder, folder, empty]),
  transform(member => {
    for (const [method, argument] of Object.entries(member) as [['appendFile' | 'appendFolder' | 'folder', string]])
      return { method, argument }
  }),
  transform(pathTransform => ({ pathTransform }))
)
const output = intersect([extension, pathTransform])

export type Config = InferOutput<typeof schema>
export const schema = object({
  systemInstruction: string(),
  output: optional(output, {}),
})
