import { access } from 'fs/promises'


export function attempt<Good, Bad>(good: () => Good, bad: (error: unknown) => Bad) {
  try {
    return good()
  } catch (error) {
    return bad(error)
  }
}

export const fileExists = (path: string) =>
  access(path).then(() => true, () => false)
