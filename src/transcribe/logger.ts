
function Logger() {
  return {
    sending: {
      start() { console.log('sending.start') },
      chunk(chunk: any) { console.log('sending chunk. length:', chunk.length) },
      end() { console.log('sending.end') },
    }
  }
}

const logger = Logger()
export default logger
