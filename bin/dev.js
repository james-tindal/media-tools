#!/usr/bin/env node

import {execute} from '@oclif/core'
import {register} from 'tsx/esm/api'

register()
await execute({dir: import.meta.url})
